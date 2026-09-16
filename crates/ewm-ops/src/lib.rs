//! # ewm-ops — the operational graph
//!
//! The third generalization of the EWM state machine: a **content-addressed
//! operational graph** in which the state-machine lattice and its processor
//! network are two physically separated sides of one bipartite graph, tied
//! together by SHA1 CIDs.
//!
//! ```text
//! ValueStore : SHA1 → HLLSet          the lattice side (values)
//! OpTable    : SHA1 → Expression      the processor side (programs)
//! Edges      : Port → Target          directed, by reference; may be cyclic
//! Dispatcher : the stack pop machine  fires [UM]s when their in-ports fill
//! ```
//!
//! - **Values** are HLLSets addressed by their content key (`h:<sha1>`).
//! - **Programs** are DSL expressions addressed by the SHA1 of their source
//!   (`p:<sha1>`). A [UM] is therefore *any expression*, not a persistent
//!   file — persistence is optional and orthogonal.
//! - **Edges** reference values by CID, never by copy; the same value can be
//!   consumed by many programs (fan-out is by reference).
//! - The **dispatcher** is a stack pop machine with two token kinds:
//!   `Value(cid)` (route to every consumer) and `Fire(op)` (run the program).
//!   The LIFO stack makes the traversal — hence the whole fire sequence —
//!   deterministic for a given graph.
//!
//! The lattice side remains acyclic by construction (it only contains
//! values, ordered by ⊆). The operational side may contain directed cycles:
//! a cycle is a feedback loop the dispatcher traverses; "time" is simply the
//! fire count. Use [`Dispatcher::run`] for acyclic graphs and
//! [`Dispatcher::run_limited`] for graphs with feedback.
//!
//! Everything follows the IICA mantra: same expression bytes → same program
//! CID; same input CIDs → same output CIDs; new values are only ever added,
//! never mutated; reconfiguration adds nodes and edges, never edits them.

pub mod boot;
#[cfg(feature = "git")]
pub mod commit;
pub mod dispatch;
pub mod dsl;
pub mod expr;
pub mod graph;

pub use boot::{boot_cid, BootStore};
#[cfg(feature = "git")]
pub use commit::commit_fire_log;
pub use dispatch::{CommitPoint, CommitReason, Dispatcher, FireLog, FireRecord, Token};
pub use dsl::{compile_boot, BootProgram};
pub use expr::{op_cid, Expression, EvalError, Word};
pub use graph::{Edge, OpGraph, OpSpec, OpTable, Port, Target, ValueCid, ValueStore};
