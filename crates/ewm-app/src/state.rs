//! The shared state layer — S(t) and H(t-1) live here, **outside the [UM]**.
//!
//! The [UM] is stateless and disposable. Everything it needs between steps is
//! kept in the [`StateCache`]:
//!
//! - **S(t)** — the emerging working state (work in progress): the cumulative
//!   per-channel working set, the live ingestor, and the turn presentation
//!   tree;
//! - **H(t-1)** — the previous committed state, the cache: the tip pointer,
//!   the head's channel states (the base of `working`), and the TF baseline.
//!
//! The cache is a plain value, so it is shareable between processing units
//! and rebuildable from the persistent store ([`StateCache::restore`]). On
//! crash the cache is discarded and restored from `ewm-git` — the committed
//! stack is untouched; only uncommitted work in progress is reprocessed
//! (idempotent by IICA).

use context_tree::{ContextTree, Leaf};
use ewm_boolring::BoolWindow;
use ewm_git::{BitTf, Ingestor, ObjectId, ObjectStore, Repository};
use hllset_contracts::token::{token_in_bytes, TokenId};
use hllset_core::{HLLSet, TFVec};

/// Capacity of the Boolean-ring window over the original turn HLLSets.
/// The ring is a moving window over ingested originals, bounded by the
/// cache size (docs/BOOLRING.md).
pub const RING_CAPACITY: usize = 64;

/// One recorded turn: the token collection and its seed-0 sketch.
#[derive(Clone, Debug)]
pub struct TurnRecord {
    /// The turn's token ids (`tid{n}` encoding).
    pub ids: Vec<TokenId>,
    /// `ingest(L)` at seed 0 — the projection of the token collection
    /// (the correspondence rule's forward direction).
    pub g1: HLLSet,
    /// The commit this turn produced (`None` when the pass brought no new
    /// bits and was skipped as idempotent).
    pub commit: Option<ObjectId>,
}

/// The shared state layer — the cache between the [UM] and the store.
///
/// Held outside the [UM]; the driver receives `&mut StateCache` for a turn
/// and keeps nothing between turns.
pub struct StateCache {
    /// S(t): the emerging working set, per channel (the commit payload).
    pub working: [HLLSet; 3],
    /// The live n-seed ingestor (token TF + bit-TF accumulators since the
    /// cache was created or restored).
    pub ingestor: Ingestor,
    /// H(t-1): the TF baseline restored from the tip.
    pub tf_base: TFVec,
    /// S(t) presentation: the Merkle tree over the turn leaves.
    pub tree: ContextTree,
    /// The turn leaves (presentation; rebuilt from commit messages on
    /// restore).
    pub turns: Vec<TurnRecord>,
    /// H(t-1): the previous committed head (the tip). The cache needed to
    /// commit changes.
    pub tip: Option<ObjectId>,
    /// Number of turns processed since the cache was created or restored.
    pub turn: u64,
    /// The Boolean ring as a moving window over the original turn HLLSets
    /// (insertion order = ingestion order; bounded by [`RING_CAPACITY`]).
    pub ring: BoolWindow,
}

impl StateCache {
    /// An empty cache for a fresh store (no tip yet).
    pub fn empty() -> Self {
        Self {
            working: std::array::from_fn(|_| HLLSet::new()),
            ingestor: Ingestor::new(&[0, 1, 2]),
            tf_base: TFVec::new(),
            tree: ContextTree::empty(),
            turns: Vec::new(),
            tip: None,
            turn: 0,
            ring: BoolWindow::new(RING_CAPACITY),
        }
    }

    /// Restore the cache from the persistent store: read the tip,
    /// dereference its states and TF, and rebuild the presentation from the
    /// commit messages. A read, not a replay.
    pub fn restore<S: ObjectStore>(repo: &Repository<S>) -> Self {
        let mut turns: Vec<TurnRecord> = Vec::new();
        let mut leaves: Vec<Leaf> = Vec::new();
        if let Ok(log) = repo.log() {
            for cid in &log {
                if let Ok(commit) = repo.read_commit(cid) {
                    if let Some(ids) = parse_ids_message(&commit.message) {
                        let g1 = HLLSet::from_tokens(ids.iter().map(|&n| token_in_bytes(n)));
                        leaves.push(Leaf {
                            h: g1.content_key(),
                            views: Vec::new(),
                        });
                        turns.push(TurnRecord {
                            ids,
                            g1,
                            commit: Some(cid.clone()),
                        });
                    }
                }
            }
        }

        let working = match repo.head() {
            Some(head) => repo
                .states(head)
                .unwrap_or_else(|_| std::array::from_fn(|_| HLLSet::new())),
            None => std::array::from_fn(|_| HLLSet::new()),
        };

        let tf_base = repo
            .head()
            .and_then(|head| repo.state_tf(head).ok())
            .and_then(|bit_tf| TFVec::from_bytes(&bit_tf.to_bytes()))
            .unwrap_or_default();

        let tree = ContextTree::build(leaves);
        let turn = turns.len() as u64;
        let tip = repo.head().cloned();

        // Rebuild the Boolean ring from the restored originals in commit
        // order — the deterministic basis of the window.
        let mut ring = BoolWindow::new(RING_CAPACITY);
        for record in &turns {
            ring.push(&record.g1);
        }

        Self {
            working,
            ingestor: Ingestor::new(&[0, 1, 2]),
            tf_base,
            tree,
            turns,
            tip,
            turn,
            ring,
        }
    }

    /// The full monotone TF: the restored baseline (H(t-1)) plus the
    /// ingestor's since-restore accumulation (TF stored, never reset).
    pub fn current_tf(&self) -> BitTf {
        let mut values = self.tf_base.values.clone();
        for (i, v) in values.iter_mut().enumerate() {
            *v += self.ingestor.bit_tf().values[i];
        }
        BitTf::from_tfvec(TFVec::from_values(values).expect("32768 entries"))
    }

    /// S(t) presentation — the Merkle tree over the turn leaves.
    pub fn tree(&self) -> &ContextTree {
        &self.tree
    }

    /// The recorded turns.
    pub fn turns(&self) -> &[TurnRecord] {
        &self.turns
    }

    /// Number of turns processed.
    pub fn turn_count(&self) -> u64 {
        self.turn
    }

    /// The cumulative working set S(t) (G1), derived from the turn leaves.
    pub fn working_g1(&self) -> &HLLSet {
        &self.working[0]
    }
}

/// The commit message carries the turn's token ids — the durable record the
/// recovery path reads to rebuild the token-level presentation.
pub(crate) fn ids_message(ids: &[TokenId]) -> String {
    let csv = ids
        .iter()
        .map(u32::to_string)
        .collect::<Vec<_>>()
        .join(",");
    format!("ids={csv}")
}

/// A structural commit (ring basis changed, no new bits): the ids plus a
/// `basis-change` suffix. The suffix makes the commit content unique (so the
/// event is durable in the DAG) while staying parseable by
/// [`parse_ids_message`].
pub(crate) fn ids_message_structural(ids: &[TokenId]) -> String {
    format!("{};basis-change", ids_message(ids))
}

/// Parse the turn ids from a commit message (`ids=1,2,3` with an optional
/// `;reason` suffix introduced by structural commits).
pub(crate) fn parse_ids_message(message: &str) -> Option<Vec<TokenId>> {
    let head = message.strip_prefix("ids=")?.split(';').next()?;
    let mut ids = Vec::new();
    for part in head.split(',') {
        let part = part.trim();
        if part.is_empty() {
            continue;
        }
        let id: TokenId = part.parse().ok()?;
        ids.push(id);
    }
    if ids.is_empty() {
        return None;
    }
    Some(ids)
}
