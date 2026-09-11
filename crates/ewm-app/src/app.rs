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

use crate::state::{ids_message, StateCache, TurnRecord};
use context_tree::Leaf;
use ewm_git::{
    view, CommitView, IngestSink, LatticeState, ObjectId, ObjectStore, Repository, StoreError,
};
use hllset_contracts::token::{token_in_bytes, TokenId};
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

        // 3. commit — one commit per turn; skipped when no new bits arrive.
        let state = LatticeState {
            g1: cache.working[0].clone(),
            g2: cache.working[1].clone(),
            g3: cache.working[2].clone(),
            tf: cache.current_tf(),
        };
        let parents: Vec<ObjectId> = cache.tip.clone().into_iter().collect();
        let message = ids_message(ids);
        let commit = if self.repo.new_bits(&state) == 0 {
            None
        } else {
            Some(self.repo.commit(&state, &parents, &message)?)
        };

        // 4. Advance the shared state: record the turn, insert the leaf,
        //    derive tree D/R/N, move the tip.
        let leaf_h = turn_g1.content_key();
        cache.turns.push(TurnRecord {
            ids: ids.to_vec(),
            g1: turn_g1,
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

        // 6. Default morphisms round-trip: the turn's ordered full image.
        let full_image = materialize(&ingest(&bytes));

        Ok(TurnOutcome {
            commit,
            head,
            tree: cache.tree.clone(),
            diff,
            commit_view,
            full_image,
        })
    }
}
