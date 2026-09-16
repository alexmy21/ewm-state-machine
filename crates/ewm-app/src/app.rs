//! The [UM] harness — the stateless driver loop of the state machine.
//!
//! ```text
//! read tip (H(t-1), the cache — from StateCache)
//!   → ingest incoming tokens → S(t) (the emerging state — in StateCache)
//!   → D/R/N vs H(t-1)
//!   → commit H(t) = (S(t), H(t-1), D, R, N)
//!   → advance head (the tip)
//! ```
//!
//! The driver owns only the store handle; S(t) and H(t-1) live in the
//! [`StateCache`](crate::state::StateCache), outside the [UM], so the state
//! is shareable and a crashed [UM] is replaced by a fresh one over the same
//! tip. Uncommitted work is reprocessed (idempotent by IICA).
//!
//! - **ingest** — one streaming pass through `ewm-git::Ingestor` (complete,
//!   single-touch, n-seed channels; the state store's bit-TF);
//! - **S(t)** — the cumulative working set in the cache, presented as a
//!   `context-tree` Merkle tree;
//! - **D/R/N** — tree-level diff of the leaves plus the bit-level
//!   `CommitView` from `ewm-git`;
//! - **commit / advance head** — one commit per turn; a pass that brings no
//!   new bits is skipped (idempotent, no-change).

use crate::state::{ids_message, ids_message_structural, StateCache, TurnRecord};
use crate::encoder::Encoder;
use context_tree::Leaf;
use ewm_boolring::RingStats;
use ewm_git::{
    view, CommitView, IngestSink, LatticeState, ObjectId, ObjectStore, Repository, StoreError,
};
use hllset_contracts::token::{parse_token_id, token_in_bytes, TokenId};
use hllset_core::HLLSet;
use hllset_morphisms::{ingest, materialize};

/// The harness's inscription choice, explicit per NEXT_SESSION §3.7.
///
/// This app path uses the `tid{n}` encoding (nanoLM/cortex). The 4-byte LE
/// encoding is exercised in `token.rs` and available to pipelines that pick
/// it; every test states which one it uses.
pub const APP_ENCODING_NAME: &str = "tid";

/// The result of one loop step.
#[derive(Clone, Debug)]
pub struct TurnOutcome {
    /// The commit produced by this turn (`None` = no-change pass).
    pub commit: Option<ObjectId>,
    /// The current head (the tip) after the step.
    pub head: Option<ObjectId>,
    /// S(t) as a Merkle tree over the turn leaves.
    pub tree: context_tree::ContextTree,
    /// Tree-level D/R/N from the previous tree to this one.
    pub diff: context_tree::TreeDiff,
    /// Bit-level Noether view of the head commit (G1). Its `S(t)` is the
    /// cumulative working set — the same snapshot as `tree` at bit level.
    pub commit_view: Option<CommitView>,
    /// The ordered restoration of this turn via the default morphisms.
    pub full_image: Vec<Vec<u8>>,
    /// The same restoration parsed back to token ids — the hand-back to
    /// the host (the host receives its own encodings, in order).
    pub restored_ids: Vec<TokenId>,
    /// The Boolean-ring statistics of this turn's original HLLSet against
    /// the moving window (linear novelty, span membership, dimension).
    pub ring_stats: RingStats,
}

/// App errors.
#[derive(Debug)]
pub enum AppError {
    Store(StoreError),
    Other(String),
}

impl std::fmt::Display for AppError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AppError::Store(e) => write!(f, "store error: {e}"),
            AppError::Other(e) => write!(f, "{e}"),
        }
    }
}

impl std::error::Error for AppError {}

impl From<StoreError> for AppError {
    fn from(e: StoreError) -> Self {
        AppError::Store(e)
    }
}

/// Sink that ignores per-pass originals (the harness registers the committed
/// cumulative state instead; per-pass provenance is a later increment).
struct NoopSink;

impl IngestSink for NoopSink {
    fn on_original(&mut self, _hllset: &HLLSet, _sha1: String) {}
}

/// The [UM] — stateless and disposable.
///
/// It owns only the handle to the persistent store. S(t) and H(t-1) live in
/// the [`StateCache`], which the caller passes in for each turn; the driver
/// keeps nothing between steps. Dropping the driver loses no state.
pub struct StateMachine<S: ObjectStore> {
    repo: Repository<S>,
}

impl<S: ObjectStore> StateMachine<S> {
    /// A fresh harness over `store` (no head yet).
    pub fn new(store: S) -> Self {
        Self {
            repo: Repository::new(store),
        }
    }

    /// Recovery: open the store, read the head. The cache is restored
    /// separately via [`StateCache::restore`] — the [UM] itself stays empty.
    pub fn open(store: S) -> Self {
        Self {
            repo: Repository::open(store),
        }
    }

    pub fn repo(&self) -> &Repository<S> {
        &self.repo
    }

    pub fn repo_mut(&mut self) -> &mut Repository<S> {
        &mut self.repo
    }

    pub fn head(&self) -> Option<&ObjectId> {
        self.repo.head()
    }

    /// Run one loop step over a turn of **host encodings**: quantize them
    /// through the shared [`Encoder`], then run the ordinary turn. This is
    /// the LLM side-car loop — encodings in, ordered encodings out.
    pub fn run_turn_encoded(
        &mut self,
        cache: &mut StateCache,
        encoder: &dyn Encoder,
        encodings: &[Vec<f32>],
    ) -> Result<TurnOutcome, AppError> {
        let ids = encoder.encode(encodings);
        self.run_turn(cache, &ids)
    }

    /// Run one loop step over a turn of `tid{n}`-encoded token ids, against
    /// the shared [`StateCache`].
    pub fn run_turn(
        &mut self,
        cache: &mut StateCache,
        ids: &[TokenId],
    ) -> Result<TurnOutcome, AppError> {
        let bytes: Vec<Vec<u8>> = ids.iter().map(|&n| token_in_bytes(n)).collect();

        // 1. ingest — complete, single-touch; accumulates into the cache.
        let pass = cache.ingestor.ingest_stream(bytes.iter(), &mut NoopSink);
        let turn_g1 = pass.channels[0].clone();

        // 2. S(t) — the cumulative working set (lattice join of all turns).
        for (i, channel) in pass.channels.iter().enumerate() {
            cache.working[i] = cache.working[i].union(channel);
        }

        // 3. Push the ring first: a basis change is a structural event and
        //    commits even when the pass brings no new bits.
        let gen_before = cache.ring.generation();
        let ring_stats = cache.ring.push(&turn_g1);
        let basis_changed = cache.ring.generation() != gen_before;

        // 4. Commit — one per turn. Skipped only when the pass brings no
        //    new bits AND the ring basis did not change; a basis change
        //    alone commits a structurally identical state under a
        //    `basis-change` message so the event is durable in the DAG.
        let state = LatticeState {
            g1: cache.working[0].clone(),
            g2: cache.working[1].clone(),
            g3: cache.working[2].clone(),
            tf: cache.current_tf(),
        };
        let parents: Vec<ObjectId> = cache.tip.clone().into_iter().collect();
        let new_bits = self.repo.new_bits(&state);
        let message = if new_bits == 0 && basis_changed {
            ids_message_structural(ids)
        } else {
            ids_message(ids)
        };
        let commit = if new_bits == 0 && !basis_changed {
            None
        } else {
            Some(self.repo.commit(&state, &parents, &message)?)
        };

        // 5. Advance the shared state: record the turn, insert the leaf,
        //    derive tree D/R/N, move the tip.
        let leaf_h = turn_g1.content_key();
        cache.turns.push(TurnRecord {
            ids: ids.to_vec(),
            g1: turn_g1.clone(),
            commit: commit.clone(),
        });
        let prev_tree = cache.tree.clone();
        cache.tree = cache.tree.insert(Leaf {
            h: leaf_h,
            views: Vec::new(),
        });
        let diff = prev_tree.diff(&cache.tree);
        cache.turn += 1;
        cache.tip = self.repo.head().cloned();

        // 5. Bit-level Noether view of the head (the head *is* the pointer).
        let head = self.repo.head().cloned();
        let commit_view = match &head {
            Some(cid) => Some(view(&self.repo, cid)?),
            None => None,
        };

        // 6. Default morphisms round-trip: the turn's ordered full image,
        //    parsed back to ids as the hand-back to the host.
        let full_image = materialize(&ingest(&bytes));
        let restored_ids: Vec<TokenId> = full_image
            .iter()
            .filter_map(|b| parse_token_id(b))
            .collect();

        Ok(TurnOutcome {
            commit,
            head,
            tree: cache.tree.clone(),
            diff,
            commit_view,
            full_image,
            restored_ids,
            ring_stats,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::RING_CAPACITY;
    use ewm_git::MemoryStore;

    #[test]
    fn no_change_pass_inside_window_is_skipped() {
        let mut um = StateMachine::new(MemoryStore::default());
        let mut cache = StateCache::empty();

        let first = um.run_turn(&mut cache, &[1, 2, 3]).unwrap();
        assert!(first.commit.is_some(), "new bits commit");
        let second = um.run_turn(&mut cache, &[1, 2, 3]).unwrap();
        assert!(
            second.commit.is_none(),
            "same ids inside the window: no new bits, no basis change"
        );
    }

    #[test]
    fn basis_change_commits_even_without_new_bits() {
        let mut um = StateMachine::new(MemoryStore::default());
        let mut cache = StateCache::empty();

        // Fill the ring window (64 originals) with distinct turns.
        for i in 0..RING_CAPACITY as u32 {
            let out = um.run_turn(&mut cache, &[i]).unwrap();
            assert!(out.commit.is_some(), "turn {i} brings new bits");
        }

        // Turn 64 repeats tid0. No new bits (the cumulative working set
        // already contains tid0), but the window has evicted turn 0, so the
        // basis is recomputed → structural commit.
        let out = um.run_turn(&mut cache, &[0]).unwrap();
        let commit = out
            .commit
            .as_ref()
            .expect("a basis change alone must still commit");
        let msg = um.repo.read_commit(commit).unwrap().message;
        assert!(
            msg.contains("basis-change"),
            "structural commit carries the basis-change suffix: {msg}"
        );
        assert_eq!(
            msg.split(';').next().unwrap(),
            "ids=0",
            "ids stay parseable for recovery"
        );
    }

    #[test]
    fn structural_message_parses_like_a_normal_one() {
        let ids = vec![7u32, 9];
        let structural = ids_message_structural(&ids);
        assert_eq!(
            crate::state::parse_ids_message(&structural),
            Some(ids.clone()),
            "recovery ignores the ;basis-change suffix"
        );
        assert_eq!(crate::state::parse_ids_message(&ids_message(&ids)), Some(ids));
    }
}
