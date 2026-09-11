//! Core HLLSet algebra engine.
//!
//! This crate provides the fundamental HLLSet data structure along with:
//!
//! - **Set operations**: union (∪), intersection (∩), difference (\), XOR (⊕)
//! - **Cardinality estimation**: Horvitz-Thompson estimator for bitmap registers
//! - **Hashing**: MurmurHash3 (seeded/unseeded) for token inscription
//! - **Content addressing**: deterministic SHA-1 keys (`h:`, `c:`) for idempotent storage
//! - **BSS morphisms**: Bell State Similarity — inclusion, exclusion, and morphism checks
//! - **Serialization**: Roaring bitmap compression with serde support
//! - **TFVec**: bit-level term frequency vector — monotonic CRDT
//! - **Commit**: lattice evolution record — D/R/N decomposition

pub mod core;

pub use core::bss;
pub use core::cardinality;
pub use core::commit::Commit;
pub use core::content_addr;
pub use core::hashing;
pub use core::hllset::{HLLSet, BITS_PER_REG, M, P};
pub use core::operations;
pub use core::serialization;
pub use core::tfvec::TFVec;
