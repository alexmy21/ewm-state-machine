//! `StateSnapshot` — the explorer's read-only view of the S(t) run-time area.
//!
//! The [UM] exports this after a turn (or on demand) so a separate tool can
//! explore and monitor the state machine without holding the harness. The
//! snapshot covers the first of the three locations:
//!
//! ```text
//! 1. S(t) run-time   — this file (working set, tree, turns, tip)
//! 2. cache           — designed (Arrow), shown as stubs until implemented
//! 3. persistent      — explored directly by ewm-sm-explore over ewm-git
//! ```

use serde::{Deserialize, Serialize};

use crate::state::{StateCache, TurnRecord, RING_CAPACITY};
use crate::StateMachine;
use ewm_git::ObjectStore;

/// The S(t) run-time snapshot of one moment in the state machine.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct StateSnapshot {
    /// The layer this snapshot projects.
    pub layer: String,
    /// The tip (H(t-1) link into the persistent layer).
    pub tip: Option<String>,
    /// Number of turns processed.
    pub turn_count: u64,
    /// Active bits of the working set per channel (G1, G2, G3).
    pub working_bits: [u64; 3],
    /// S(t) presentation: the Merkle root over the turn leaves.
    pub tree_root: String,
    /// Number of leaves in the S(t) tree.
    pub tree_leaves: usize,
    /// The turn records (ids, G1 leaf key, commit).
    pub turns: Vec<TurnSnapshot>,
    /// Number of entries in the TF baseline restored from the tip.
    pub tf_base_entries: usize,
    /// The designed cache layer (Arrow RecordBatches) — stubs until
    /// implemented; the names are pinned in docs/ARROW_CACHE.md.
    pub cache: CacheStub,
    /// The Boolean-ring window over the original turn HLLSets
    /// (docs/BOOLRING.md): deterministic GF(2) span of the ingestion order.
    pub ring: RingSnapshot,
}

/// The Boolean-ring window projection.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct RingSnapshot {
    /// The window capacity (cache-bounded).
    pub capacity: usize,
    /// Originals currently in the window.
    pub window_len: usize,
    /// GF(2) span dimension of the window — the context width.
    pub dimension: usize,
}

/// One turn in the S(t) presentation.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct TurnSnapshot {
    /// Turn index (0-based).
    pub turn: u64,
    /// The turn's token ids (`tid{n}` encoding).
    pub ids: Vec<u32>,
    /// The turn leaf key — `h:<sha1>` of the seed-0 sketch.
    pub g1_key: String,
    /// The commit this turn produced (`None` = no-change pass).
    pub commit: Option<String>,
}

/// The cache layer as designed — not yet implemented.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct CacheStub {
    /// Fixed status until the Arrow-backed cache lands.
    pub status: String,
    /// The batch names pinned by docs/ARROW_CACHE.md §4.
    pub batches: Vec<String>,
}

impl StateSnapshot {
    /// Serialize to pretty JSON (the explorer's `snapshot` input format).
    pub fn to_json(&self) -> String {
        serde_json::to_string_pretty(self).expect("StateSnapshot serializes")
    }
}

impl TurnSnapshot {
    fn from_record(turn: u64, record: &TurnRecord) -> Self {
        Self {
            turn,
            ids: record.ids.clone(),
            g1_key: record.g1.content_key(),
            commit: record.commit.as_ref().map(|c| c.to_string()),
        }
    }
}

impl<S: ObjectStore> StateMachine<S> {
    /// Export the S(t) run-time area for the explorer. Read-only.
    pub fn snapshot(&self, cache: &StateCache) -> StateSnapshot {
        let turns = cache
            .turns
            .iter()
            .enumerate()
            .map(|(i, record)| TurnSnapshot::from_record(i as u64, record))
            .collect();

        StateSnapshot {
            layer: "S(t) run-time".to_string(),
            tip: cache.tip.as_ref().map(|t| t.to_string()),
            turn_count: cache.turn,
            working_bits: [
                cache.working[0].popcount(),
                cache.working[1].popcount(),
                cache.working[2].popcount(),
            ],
            tree_root: cache.tree.root().to_string(),
            tree_leaves: cache.tree.leaves().len(),
            turns,
            tf_base_entries: cache.tf_base.values.len(),
            ring: RingSnapshot {
                capacity: RING_CAPACITY,
                window_len: cache.ring.window_len(),
                dimension: cache.ring.dimension(),
            },
            cache: CacheStub {
                status: "designed (Arrow) — not implemented".to_string(),
                batches: vec![
                    "hllset_lut".to_string(),
                    "ng:G1".to_string(),
                    "ng:G2".to_string(),
                    "ng:G3".to_string(),
                    "ns:G1".to_string(),
                    "ns:G2".to_string(),
                    "ns:G3".to_string(),
                    "tf_table".to_string(),
                    "tf_vec".to_string(),
                    "tree_leaves".to_string(),
                    "tree_levels".to_string(),
                    "g1".to_string(),
                    "g2".to_string(),
                    "g3".to_string(),
                    "turns".to_string(),
                    "MANIFEST".to_string(),
                ],
            },
        }
    }
}
