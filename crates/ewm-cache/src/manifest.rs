//! The cache manifest — `MANIFEST.arrow`, the atomic commit point of a
//! cache snapshot (ARROW_CACHE.md §5 rule 3).
//!
//! The manifest is a single Arrow batch of `(key: Utf8, value: Utf8)` rows,
//! canonically sorted by key:
//!
//! ```text
//! schema_version → ARROW_CACHE_SCHEMA_VERSION
//! tip            → the commit tip the snapshot was taken at
//! created_at     → RFC3339 timestamp
//! object:G1      → <sha1>   (Roaring bytes of the G1 HLLSet)
//! object:G2      → <sha1>
//! object:G3      → <sha1>
//! object:tf_vec  → <sha1>   (TFVec bytes)
//! table:<name>   → <sha1 of the table's IPC bytes>
//! ```
//!
//! The cache content key is `c:<sha1 of the manifest IPC bytes>`.

use std::collections::BTreeMap;
use std::sync::Arc;

use arrow::array::StringArray;
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use hllset_contracts::{sha1_hex, ARROW_CACHE_SCHEMA_VERSION};
use serde::{Deserialize, Serialize};

use crate::schema::{from_ipc, to_ipc};
use crate::storage::CacheError;

/// The logical names that are HLLSet objects (everything else under
/// `object:` is a TFVec).
pub const HLLSET_OBJECTS: [&str; 3] = ["G1", "G2", "G3"];

/// The TFVec logical name.
pub const TFVEC_OBJECT: &str = "tf_vec";

/// A cache snapshot manifest.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Manifest {
    pub schema_version: u32,
    pub tip: String,
    pub created_at: String,
    /// Logical name (e.g. `G1`, `tf_vec`) → object SHA1.
    pub objects: BTreeMap<String, String>,
    /// Table name (e.g. `hllset_lut`) → SHA1 of the table's IPC bytes.
    pub tables: BTreeMap<String, String>,
}

impl Manifest {
    /// A manifest for the current soldered schema version.
    pub fn new(tip: impl Into<String>, created_at: impl Into<String>) -> Self {
        Self {
            schema_version: ARROW_CACHE_SCHEMA_VERSION,
            tip: tip.into(),
            created_at: created_at.into(),
            objects: BTreeMap::new(),
            tables: BTreeMap::new(),
        }
    }

    /// The canonical rows of the manifest batch.
    pub fn rows(&self) -> Vec<(String, String)> {
        let mut rows = vec![
            (
                "schema_version".to_string(),
                self.schema_version.to_string(),
            ),
            ("tip".to_string(), self.tip.clone()),
            ("created_at".to_string(), self.created_at.clone()),
        ];
        for (name, sha1) in &self.objects {
            rows.push((format!("object:{name}"), sha1.clone()));
        }
        for (name, sha1) in &self.tables {
            rows.push((format!("table:{name}"), sha1.clone()));
        }
        rows.sort();
        rows
    }

    /// Serialize to the canonical `MANIFEST.arrow` bytes.
    pub fn to_ipc(&self) -> Result<Vec<u8>, CacheError> {
        let schema = Schema::new(vec![
            Field::new("key", DataType::Utf8, false),
            Field::new("value", DataType::Utf8, false),
        ]);
        let rows = self.rows();
        let key = StringArray::from(rows.iter().map(|r| r.0.as_str()).collect::<Vec<_>>());
        let value = StringArray::from(rows.iter().map(|r| r.1.as_str()).collect::<Vec<_>>());
        let batch = RecordBatch::try_new(Arc::new(schema), vec![Arc::new(key), Arc::new(value)])?;
        to_ipc(&batch)
    }

    /// Parse the canonical manifest batch.
    pub fn from_ipc(bytes: &[u8]) -> Result<Self, CacheError> {
        let batch = from_ipc(bytes)?;
        let key = batch
            .column(0)
            .as_any()
            .downcast_ref::<StringArray>()
            .ok_or(CacheError::BadManifest("key column is not Utf8".into()))?;
        let value = batch
            .column(1)
            .as_any()
            .downcast_ref::<StringArray>()
            .ok_or(CacheError::BadManifest("value column is not Utf8".into()))?;

        let mut schema_version = None;
        let mut tip = String::new();
        let mut created_at = String::new();
        let mut objects = BTreeMap::new();
        let mut tables = BTreeMap::new();

        for i in 0..batch.num_rows() {
            let k = key.value(i);
            let v = value.value(i);
            match k {
                "schema_version" => {
                    schema_version = Some(v.parse::<u32>().map_err(|_| {
                        CacheError::BadManifest(format!("schema_version not a u32: {v}"))
                    })?)
                }
                "tip" => tip = v.to_string(),
                "created_at" => created_at = v.to_string(),
                _ if k.starts_with("object:") => {
                    objects.insert(k[7..].to_string(), v.to_string());
                }
                _ if k.starts_with("table:") => {
                    tables.insert(k[6..].to_string(), v.to_string());
                }
                other => {
                    return Err(CacheError::BadManifest(format!("unknown key {other}")));
                }
            }
        }

        Ok(Self {
            schema_version: schema_version.ok_or_else(|| {
                CacheError::BadManifest("missing schema_version".into())
            })?,
            tip,
            created_at,
            objects,
            tables,
        })
    }
}

/// The cache content key: `c:<sha1 of the manifest IPC bytes>`.
pub fn content_key(manifest_ipc_bytes: &[u8]) -> String {
    format!("c:{}", sha1_hex(manifest_ipc_bytes))
}
