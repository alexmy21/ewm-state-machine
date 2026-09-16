//! # ewm-cache — the Arrow-backed extended cache
//!
//! Implementation of docs/ARROW_CACHE.md: Apache Arrow is the **cache-layer
//! substrate**, never the state representation. HLLSets stay Roaring-native;
//! Arrow IPC files form the **extended cache** (spill/restore) for the
//! derived, rebuildable cache objects.
//!
//! ```text
//! <cache_dir>/
//! ├── MANIFEST.arrow        # schema_version, tip, created_at, name → sha1
//! ├── objects/
//! │   ├── <sha1>.hllset     # Roaring bytes of G1/G2/G3, named by content ID
//! │   └── <sha1>.tfvec      # TFVec bytes, named by its content ID
//! └── tables/
//!     ├── hllset_lut.arrow  # append-only, context-scoped
//!     ├── ng:G1.arrow …     # token LUTs (bit, token), append-only
//!     ├── tree_leaves.arrow / tree_levels.arrow
//!     └── turns.arrow
//! ```
//!
//! Rules (ARROW_CACHE.md §5):
//!
//! 1. one IPC file per batch — partial updates rewrite only the changed batch;
//! 2. content-addressed objects live under their SHA1;
//! 3. `MANIFEST.arrow` is the atomic commit point of a cache snapshot;
//! 4. the schema version is the soldered [`ARROW_CACHE_SCHEMA_VERSION`];
//! 5. restore reads the manifest, verifies SHA1s, then rebuilds native objects;
//! 6. the cache directory carries a content key `c:<sha1 of manifest bytes>`.
//!
//! Everything the cache holds is derived data: any corrupt batch can be
//! recomputed from the `ewm-git` store. Arrow changes no `h:`/`t:` keys.

pub mod manifest;
pub mod schema;
pub mod storage;

pub use hllset_contracts::ARROW_CACHE_SCHEMA_VERSION;
pub use manifest::{content_key, Manifest};
pub use storage::{CacheError, ExtendedCache, Snapshot};
