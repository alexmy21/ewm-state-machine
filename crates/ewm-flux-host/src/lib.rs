//! # ewm-flux-host — the Flux/MMDiT side-car host adapter
//!
//! The side-car for a generative-image host (aarambh-vision-studio's
//! rectified-flow MMDiT sampler). Until the host's inference runtime
//! stabilizes, this crate drives the loop against a **synthetic,
//! random-weight MMDiT shim** and closes the whole Phase-1 loop here:
//!
//! ```text
//! synthetic MMDiT shim (random weights, rectified-flow ODE)
//!        │  per-step latent state S(t)   (seq_len × dim, batch 1)
//!        ▼
//! SidecarProbe (CodebookProbe)  — the only host-specific step
//!        │  shared codebook → tid{n} ids → default morphism
//!        ▼
//! HLLSet S(t) → H(t-1), D/R/N, Boolean-ring window, warnings
//!        │
//!        ▼
//! two-score eval:  loop accuracy   = latent token restoration
//!                  decode quality  = latent reconstruction (cosine / MSE)
//! ```
//!
//! The design mirrors the Qwen-Drive Phase-1 bench
//! (docs/ASSIGNMENT_QWENDRIVE.md): the side-car is host-agnostic by design —
//! everything downstream of the [`SidecarProbe`] HLLSet runs on bit sets, and
//! only the probe knows how to read a tensor (docs/notes/flux-notes.md).

pub mod probe;
pub mod shim;
pub mod sidecar;

pub use ewm_app::CodebookEncoder;
pub use probe::{CodebookProbe, DecodeRecord, LatentState, SidecarProbe};
pub use shim::{Disturbance, SyntheticFluxHost};
pub use sidecar::{
    run, run_states, sidecar_series, sidecar_series_with_window, warnings, BasisHistory,
    BasisSnapshot, DrnRecord, FinalProjection, FluxConfig, FluxReport, Projection, RingRecord,
    Scores, SidecarFrame, SidecarSeries, StepFrame, Warnings,
};

/// Capacity of the Boolean-ring window over the original per-step HLLSets —
/// the same window contract as the [UM] cache (docs/BOOLRING.md).
pub const RING_CAPACITY: usize = ewm_app::RING_CAPACITY;
