//! The [UM] harness — the stateless driver loop of the state machine.
//!
//! ```text
//! read tip (H(t-1), the cache)
//!   → ingest incoming tokens → S(t) (the emerging state)
//!   → D/R/N vs H(t-1)
//!   → commit H(t) = (S(t), H(t-1), D, R, N)
//!   → advance head (the tip)
//! ```
//!
//! The driver holds no processing state between steps: the only durable
//! state is the store + the head pointer. If the [UM] crashes, the stack is
//! intact — a fresh [UM] reads the same tip and resumes. Uncommitted work is
//! simply reprocessed (idempotent by IICA).
//!
//! - **ingest** — one streaming pass through `ewm-git::Ingestor` (complete,
//!   single-touch, n-seed channels; the state store's bit-TF);
//! - **S(t)** — the cumulative working set: the lattice join of every turn's
//!   channel sketches, presented as a `context-tree` Merkle tree;
//! - **D/R/N** — tree-level diff of the leaves plus the bit-level
//!   `CommitView` from `ewm-git`;
//! - **commit / advance head** — one commit per turn; a pass that brings no
//!   new bits is skipped (idempotent, no-change).

use context_tree::{ContextTree, Leaf, TreeDiff};
use ewm_git::{
    view, BitTf, CommitView, IngestSink, Ingestor, LatticeState, ObjectId, ObjectStore,
    Repository, StoreError,
};
use hllset_contracts::token::{token_in_bytes, TokenId};
use hllset_core::{HLLSet, TFVec};
use hllset_morphisms::{ingest, materialize};

/// The harness's inscription choice, explicit per NEXT_SESSION §3.7.
///
/// This app path uses the `tid{n}` encoding (nanoLM/cortex). The 4-byte LE
/// encoding is exercised in `token.rs` and available to pipelines that pick
/// it; every test states which one it uses.
pub const APP_ENCODING_NAME: &str = "tid";

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

/// The result of one loop step.
#[derive(Clone, Debug)]
pub struct TurnOutcome {
    /// The commit produced by this turn (`None` = no-change pass).
    pub commit: Option<ObjectId>,
    /// The current head (the tip) after the step.
    pub head: Option<ObjectId>,
    /// S(t) as a Merkle tree over the turn leaves.
    pub tree: ContextTree,
    /// Tree-level D/R/N from the previous tree to this one.
    pub diff: TreeDiff,
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

/// The [UM] harness: token turns in, commits out. Stateless between steps —
/// it can be dropped and reopened from the store (`StateMachine::open` reads
/// the head, never replays).
pub struct StateMachine<S: ObjectStore> {
    repo: Repository<S>,
    ingestor: Ingestor,
    turns: Vec<TurnRecord>,
    tree: ContextTree,
    /// The cumulative working set `S(t)`, per channel (the commit payload).
    working: [HLLSet; 3],
    /// TF baseline restored from the head commit on open (the ingestor's
    /// own accumulator counts only since process start; the sum is the full
    /// monotone TF).
    tf_base: TFVec,
    turn: u64,
}

impl<S: ObjectStore> StateMachine<S> {
    /// A fresh harness over `store` (no head yet).
    pub fn new(store: S) -> Self {
        Self {
            repo: Repository::new(store),
            ingestor: Ingestor::new(&[0, 1, 2]),
            turns: Vec::new(),
            tree: ContextTree::empty(),
            working: std::array::from_fn(|_| HLLSet::new()),
            tf_base: TFVec::new(),
            turn: 0,
        }
    }

    /// Recovery: open the store, read the head, dereference, resume — a
    /// read, not a replay.
    ///
    /// - the pointer = `repo.head()`;
    /// - the snapshot = the head commit's three channel states;
    /// - the presentation (turns + S(t) tree) is rebuilt from the commit
    ///   messages, which carry each turn's token ids.
    pub fn open(store: S) -> Self {
        let repo = Repository::open(store);

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

        Self {
            repo,
            ingestor: Ingestor::new(&[0, 1, 2]),
            turns,
            tree,
            working,
            tf_base,
            turn,
        }
    }

    pub fn repo(&self) -> &Repository<S> {
        &self.repo
    }

    pub fn repo_mut(&mut self) -> &mut Repository<S> {
        &mut self.repo
    }

    pub fn ingestor(&self) -> &Ingestor {
        &self.ingestor
    }

    pub fn turns(&self) -> &[TurnRecord] {
        &self.turns
    }

    /// S(t) — the working set as a Merkle tree over the turn leaves.
    pub fn tree(&self) -> &ContextTree {
        &self.tree
    }

    pub fn turn_count(&self) -> u64 {
        self.turn
    }

    pub fn head(&self) -> Option<&ObjectId> {
        self.repo.head()
    }

    /// The cumulative working set `S(t)` (G1), derived from the turn leaves.
    pub fn working_g1(&self) -> &HLLSet {
        &self.working[0]
    }

    /// Run one loop step over a turn of `tid{n}`-encoded token ids.
    pub fn run_turn(&mut self, ids: &[TokenId]) -> Result<TurnOutcome, AppError> {
        let bytes: Vec<Vec<u8>> = ids.iter().map(|&n| token_in_bytes(n)).collect();

        // 1. ingest — complete, single-touch (mutates the ingestor's TF only).
        let pass = self.ingestor.ingest_stream(bytes.iter(), &mut NoopSink);
        let turn_g1 = pass.channels[0].clone();

        // 2. S(t) — the cumulative working set (lattice join of all turns).
        for (i, channel) in pass.channels.iter().enumerate() {
            self.working[i] = self.working[i].union(channel);
        }

        // 3. commit — one commit per turn; skipped when no new bits arrive.
        let state = LatticeState {
            g1: self.working[0].clone(),
            g2: self.working[1].clone(),
            g3: self.working[2].clone(),
            tf: self.current_tf(),
        };
        let parents: Vec<ObjectId> = self.repo.head().cloned().into_iter().collect();
        let message = ids_message(ids);
        let commit = if self.repo.new_bits(&state) == 0 {
            None
        } else {
            Some(self.repo.commit(&state, &parents, &message)?)
        };

        let leaf_h = turn_g1.content_key();
        self.turns.push(TurnRecord {
            ids: ids.to_vec(),
            g1: turn_g1,
            commit: commit.clone(),
        });

        // 4. S(t) presentation — insert the turn leaf, derive tree D/R/N.
        let prev_tree = self.tree.clone();
        self.tree = self.tree.insert(Leaf {
            h: leaf_h,
            views: Vec::new(),
        });
        let diff = prev_tree.diff(&self.tree);
        self.turn += 1;

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
            tree: self.tree.clone(),
            diff,
            commit_view,
            full_image,
        })
    }

    /// The full monotone TF: the restored baseline plus the ingestor's
    /// since-start accumulation (TF stored, never reset).
    fn current_tf(&self) -> BitTf {
        let mut values = self.tf_base.values.clone();
        for (i, v) in values.iter_mut().enumerate() {
            *v += self.ingestor.bit_tf().values[i];
        }
        BitTf::from_tfvec(TFVec::from_values(values).expect("32768 entries"))
    }
}

/// The commit message carries the turn's token ids — the durable record the
/// recovery path reads to rebuild the token-level presentation.
fn ids_message(ids: &[TokenId]) -> String {
    let csv = ids
        .iter()
        .map(u32::to_string)
        .collect::<Vec<_>>()
        .join(",");
    format!("ids={csv}")
}

/// Parse the turn ids from a commit message (`ids=1,2,3`).
fn parse_ids_message(message: &str) -> Option<Vec<TokenId>> {
    let csv = message.strip_prefix("ids=")?;
    let mut ids = Vec::new();
    for part in csv.split(',') {
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
