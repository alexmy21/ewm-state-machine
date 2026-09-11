//! # ewm-app — the [UM] harness
//!
//! The stateless driver loop of the state machine. The [UM] (processing unit)
//! is disposable: it reads the tip, ingests the incoming token turn, proposes
//! S(t), computes D/R/N against H(t-1), commits, and advances the head.
//!
//! ```text
//!         ┌────────────────────────────┐
//!         │  token source (stub/ollama)│  fire-and-forget, never blocking
//!         └────────────┬───────────────┘
//!                      │ token turns
//!         ┌────────────▼───────────────┐
//!         │  ewm-app ([UM] harness)    │  stateless; runs the state loop
//!         │  read tip → ingest → S(t)  │
//!         │  → D/R/N → commit → head   │
//!         └────────────┬───────────────┘
//!                      │ CIDs
//!         ┌────────────▼───────────────┐
//!         │  ewm-git (the state stack) │  content-addressed store + tip
//!         └────────────────────────────┘
//! ```
//!
//! Recovery is a read, not a replay: [`StateMachine::open`] reads the head,
//! dereferences the snapshots, and rebuilds the presentation from the commit
//! messages. A crashed [UM] is replaced by a fresh one over the same tip.

pub mod app;
pub mod llm;
pub mod state;
pub mod token;

pub use app::{AppError, StateMachine, TurnOutcome, APP_ENCODING_NAME};
pub use llm::{OllamaLlm, StubLlm, TurnSource};
pub use state::{StateCache, TurnRecord};
pub use token::TokenEncoding;

// ── Direct access to the two morphisms ─────────────────────────────────────
// The default application-level interface of the hllset foundation. The DSL
// remains the escape hatch for custom HLLSet building; when nothing special
// is needed, an application calls `ingest` / `materialize` directly.
pub use hllset_morphisms::{
    ingest, ingest_key, ingest_with_pad, materialize, materialize_no_order, materialize_with,
    unordered_tokens, HllsetLut, Ingested, MaterializeOptions, Order, PAD,
};
