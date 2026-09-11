//! `hllset-morphisms` — the two morphisms with their operational contracts.
//!
//! **Ingest (complete, single touch).** For each token, in one pass and
//! without leaving the module:
//!
//! 1. compute the 3 seeded hashes (n-seed encodings; the n-gram regime is
//!    the same shape with channel-selected seeds),
//! 2. set the atom in the corresponding HLLSet,
//! 3. insert the token into the corresponding LUT fiber,
//! 4. increment TF.
//!
//! **Materialize (LUT-first, keep every reference).** For each active bit
//! of the sketch, collect candidate tokens from **all pointed LUTs** across
//! all encodings; a bit with several candidates restores **all** of them.
//! Collisions are normal in large token collections, so TF is never used to
//! filter — this is probabilistic restoration. TF exists for ranking only.
//!
//! # Application-level default interface
//!
//! The DSL is the place for custom HLLSet building; when nothing special is
//! needed, applications call the default [`ingest`] and [`materialize`]
//! directly (see [`api`] for the operational contracts):
//!
//! - [`ingest`] — an ordered token collection, processed end-to-end with
//!   one start pad + two end pads and 1-/2-/3-gram encoding (three hashes
//!   point at every token, one per channel);
//! - [`materialize`] — LUT-first over the three channels, returning the
//!   restored tokens in their original order by default ([`Order::NoOrder`]
//!   for the plain set).
//!
//! The low-level modules remain available at [`ingest::Ingest`] (n-seed
//! single-touch) and [`materialize::materialize`] (LUT-first pairs).

pub mod api;
pub mod conv;
pub mod grid;
pub mod hllset_lut;
pub mod ingest;
pub mod materialize;
pub mod scheme;
pub mod tensor;
pub mod tf;

pub use api::{
    gate, ingest, ingest_key, ingest_with_pad, materialize, materialize_beam, materialize_no_order,
    materialize_with, unordered_tokens, Ingested, MaterializeOptions, Order, CHANNELS,
    CHANNEL_NAMES, CHANNEL_SEEDS, PAD,
};
pub use conv::{channel_name, seed, ConvSpec};
pub use grid::{
    ingest_grid, ingest_grid_with_pad, materialize_grid, materialize_grid_beam,
    materialize_grid_no_order, Grid, GridIngested, GRID_BORDER, GRID_MAX_N,
};
pub use hllset_lut::{HllsetLut, UNNAMED};
pub use ingest::{Ingest, N_SEEDS, SEEDS};
pub use scheme::{lut_channel, lut_name, lut_scheme, NG, NS};
pub use tensor::{
    ingest_tensor, ingest_tensor_with_pad, materialize_tensor, materialize_tensor_beam,
    materialize_tensor_no_order, Tensor, TensorIngested, TENSOR_BORDER, TENSOR_MAX_N,
};
pub use tf::TfTable;
