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
//! **Materialize (LUT-first, TF only for ambiguity).** For each active bit
//! of the sketch, collect candidate tokens from **all pointed LUTs** across
//! all encodings; only when a bit resolves to more than one candidate does
//! the LUT's TF break the tie. TF is never the starting point — normally
//! people start with TF; this module never does.
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
pub mod hllset_lut;
pub mod ingest;
pub mod materialize;
pub mod scheme;
pub mod tf;

pub use api::{
    ingest, ingest_key, ingest_with_pad, materialize, materialize_no_order, materialize_with,
    unordered_tokens, Ingested, MaterializeOptions, Order, CHANNELS, CHANNEL_NAMES, CHANNEL_SEEDS,
    PAD,
};
pub use hllset_lut::{HllsetLut, UNNAMED};
pub use ingest::{Ingest, N_SEEDS, SEEDS};
pub use scheme::{lut_channel, lut_name, lut_scheme, NG, NS};
pub use tf::TfTable;
