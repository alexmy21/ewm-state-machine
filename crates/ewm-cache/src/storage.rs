//! The extended cache directory (ARROW_CACHE.md §5): content-addressed
//! objects under `objects/`, one IPC file per table under `tables/`, and
//! the `MANIFEST.arrow` atomic commit point.

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::io;
use std::path::{Path, PathBuf};

use arrow::error::ArrowError;
use arrow::record_batch::RecordBatch;
use hllset_contracts::{sha1_hex, ARROW_CACHE_SCHEMA_VERSION};
use hllset_core::HLLSet;
use hllset_lut::LutIndex;

use crate::manifest::{content_key, Manifest, HLLSET_OBJECTS, TFVEC_OBJECT};
use crate::schema::{from_ipc, lut_batch, lut_rows, to_ipc};

/// Cache errors.
#[derive(Debug, thiserror::Error)]
pub enum CacheError {
    #[error("io: {0}")]
    Io(#[from] io::Error),
    #[error("arrow: {0}")]
    Arrow(#[from] ArrowError),
    #[error("bad manifest: {0}")]
    BadManifest(String),
    #[error("sha mismatch for {what} {name}: manifest {expected}, found {found}")]
    ShaMismatch {
        what: String,
        name: String,
        expected: String,
        found: String,
    },
    #[error("missing {what}: {name}")]
    Missing { what: String, name: String },
    #[error("schema version {found} != soldered {ARROW_CACHE_SCHEMA_VERSION}")]
    SchemaVersion { found: u32 },
    #[error("empty IPC stream")]
    EmptyIpc,
    #[error("multi-batch IPC stream (expected one)")]
    MultiBatchIpc,
}

/// A validated cache snapshot: the manifest plus the native objects and the
/// table batches it references.
#[derive(Clone, Debug)]
pub struct Snapshot {
    pub manifest: Manifest,
    /// `G1`, `G2`, `G3` by logical name.
    pub hllsets: BTreeMap<String, HLLSet>,
    /// `tf_vec` values, if the snapshot references one.
    pub tf_vec: Option<Vec<f64>>,
    /// Table name → batch.
    pub tables: BTreeMap<String, RecordBatch>,
}

/// The Arrow IPC extended cache on disk.
pub struct ExtendedCache {
    dir: PathBuf,
}

impl ExtendedCache {
    /// Open (or create) a cache directory with the §5 layout.
    pub fn open(dir: impl AsRef<Path>) -> io::Result<Self> {
        let dir = dir.as_ref().to_path_buf();
        fs::create_dir_all(dir.join("objects"))?;
        fs::create_dir_all(dir.join("tables"))?;
        Ok(Self { dir })
    }

    pub fn dir(&self) -> &Path {
        &self.dir
    }

    // ── objects ─────────────────────────────────────────────────────────

    /// Store an HLLSet under its content SHA1 (`objects/<sha1>.hllset`);
    /// returns the SHA1.
    pub fn put_hllset(&self, set: &HLLSet) -> io::Result<String> {
        let bytes = set.to_bytes();
        let sha1 = sha1_hex(&bytes);
        fs::write(self.dir.join("objects").join(format!("{sha1}.hllset")), &bytes)?;
        Ok(sha1)
    }

    /// Load an HLLSet object by SHA1.
    pub fn get_hllset(&self, sha1: &str) -> io::Result<Option<HLLSet>> {
        let path = self.dir.join("objects").join(format!("{sha1}.hllset"));
        if !path.exists() {
            return Ok(None);
        }
        let bytes = fs::read(path)?;
        Ok(HLLSet::from_bytes(&bytes))
    }

    /// Store a TFVec under its content SHA1 (`objects/<sha1>.tfvec`);
    /// returns the SHA1. Requires 32,768 values.
    pub fn put_tfvec(&self, values: &[f64]) -> io::Result<String> {
        assert_eq!(values.len(), 32768, "tf_vec must have 32768 entries");
        let mut bytes = Vec::with_capacity(values.len() * 8);
        for v in values {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        let sha1 = sha1_hex(&bytes);
        fs::write(self.dir.join("objects").join(format!("{sha1}.tfvec")), &bytes)?;
        Ok(sha1)
    }

    /// Load a TFVec object by SHA1.
    pub fn get_tfvec(&self, sha1: &str) -> io::Result<Option<Vec<f64>>> {
        let path = self.dir.join("objects").join(format!("{sha1}.tfvec"));
        if !path.exists() {
            return Ok(None);
        }
        let bytes = fs::read(path)?;
        Ok(Some(
            bytes
                .chunks_exact(8)
                .map(|c| f64::from_le_bytes(c.try_into().expect("8 bytes")))
                .collect(),
        ))
    }

    // ── tables ──────────────────────────────────────────────────────────

    /// Write one table batch to `tables/<name>.arrow`; returns the SHA1 of
    /// the IPC bytes.
    pub fn write_table(&self, name: &str, batch: &RecordBatch) -> Result<String, CacheError> {
        let bytes = to_ipc(batch)?;
        let sha1 = sha1_hex(&bytes);
        fs::write(self.dir.join("tables").join(format!("{name}.arrow")), &bytes)?;
        Ok(sha1)
    }

    /// Read one table batch.
    pub fn read_table(&self, name: &str) -> Result<Option<RecordBatch>, CacheError> {
        let path = self.dir.join("tables").join(format!("{name}.arrow"));
        if !path.exists() {
            return Ok(None);
        }
        Ok(Some(from_ipc(&fs::read(path)?)?))
    }

    /// Merge one ingest's token LUT into the shared reverse LUT table
    /// `tables/<name>.arrow` (append-only union, deduplicated by
    /// `(bit, token)`). Returns the SHA1 of the new IPC bytes.
    ///
    /// The shared table accumulates across ingests, so materialization can
    /// run against it without a live `Ingest`/`Ingested` — the shared LUT.
    pub fn merge_lut(&self, name: &str, index: &LutIndex) -> Result<String, CacheError> {
        let mut rows: BTreeSet<(u32, Vec<u8>)> = BTreeSet::new();
        if let Some(batch) = self.read_table(name)? {
            rows.extend(lut_rows(&batch));
        }
        rows.extend(index.rows());
        let rows: Vec<(u32, Vec<u8>)> = rows.into_iter().collect();
        self.write_table(name, &lut_batch(&rows)?)
    }

    /// Materialize an HLLSet against a shared token-LUT table — the shared
    /// reverse LUT: every candidate referenced by an active bit is kept
    /// (probabilistic restoration, no TF filtering, no live ingest).
    pub fn materialize_lut(
        &self,
        hllset: &HLLSet,
        table: &str,
    ) -> Result<BTreeSet<Vec<u8>>, CacheError> {
        let batch = self.read_table(table)?.ok_or_else(|| CacheError::Missing {
            what: "table".into(),
            name: table.to_string(),
        })?;
        let mut fibers: BTreeMap<u32, BTreeSet<Vec<u8>>> = BTreeMap::new();
        for (bit, token) in lut_rows(&batch) {
            fibers.entry(bit).or_default().insert(token);
        }
        let mut out = BTreeSet::new();
        for addr in hllset.bit_addresses() {
            if let Some(tokens) = fibers.get(&addr.bit()) {
                out.extend(tokens.iter().cloned());
            }
        }
        Ok(out)
    }

    // ── manifest ────────────────────────────────────────────────────────

    /// Seal a snapshot: write `MANIFEST.arrow` and return `(manifest, c:<sha1>)`.
    pub fn seal(&self, manifest: &Manifest) -> Result<(Manifest, String), CacheError> {
        if manifest.schema_version != ARROW_CACHE_SCHEMA_VERSION {
            return Err(CacheError::SchemaVersion {
                found: manifest.schema_version,
            });
        }
        let bytes = manifest.to_ipc()?;
        let cid = content_key(&bytes);
        fs::write(self.dir.join("MANIFEST.arrow"), &bytes)?;
        Ok((manifest.clone(), cid))
    }

    /// Read the sealed manifest, if present.
    pub fn open_manifest(&self) -> Result<Option<Manifest>, CacheError> {
        let path = self.dir.join("MANIFEST.arrow");
        if !path.exists() {
            return Ok(None);
        }
        Ok(Some(Manifest::from_ipc(&fs::read(path)?)?))
    }

    /// Verify a manifest: schema version, then every referenced object and
    /// table must exist and hash to its manifest SHA1 (ARROW_CACHE.md §5
    /// rule 3).
    pub fn validate(&self, manifest: &Manifest) -> Result<(), CacheError> {
        if manifest.schema_version != ARROW_CACHE_SCHEMA_VERSION {
            return Err(CacheError::SchemaVersion {
                found: manifest.schema_version,
            });
        }
        for (name, sha1) in &manifest.objects {
            if HLLSET_OBJECTS.contains(&name.as_str()) {
                let path = self.dir.join("objects").join(format!("{sha1}.hllset"));
                let bytes = fs::read(&path)
                    .map_err(|_| CacheError::Missing {
                        what: "object".into(),
                        name: name.clone(),
                    })?;
                let found = sha1_hex(&bytes);
                if found != *sha1 {
                    return Err(CacheError::ShaMismatch {
                        what: "object".into(),
                        name: name.clone(),
                        expected: sha1.clone(),
                        found,
                    });
                }
            } else if name == TFVEC_OBJECT {
                let path = self.dir.join("objects").join(format!("{sha1}.tfvec"));
                let bytes = fs::read(&path)
                    .map_err(|_| CacheError::Missing {
                        what: "object".into(),
                        name: name.clone(),
                    })?;
                let found = sha1_hex(&bytes);
                if found != *sha1 {
                    return Err(CacheError::ShaMismatch {
                        what: "object".into(),
                        name: name.clone(),
                        expected: sha1.clone(),
                        found,
                    });
                }
            } else {
                return Err(CacheError::BadManifest(format!(
                    "unknown object logical name {name}"
                )));
            }
        }
        for (name, sha1) in &manifest.tables {
            let path = self.dir.join("tables").join(format!("{name}.arrow"));
            let bytes = fs::read(&path).map_err(|_| CacheError::Missing {
                what: "table".into(),
                name: name.clone(),
            })?;
            let found = sha1_hex(&bytes);
            if found != *sha1 {
                return Err(CacheError::ShaMismatch {
                    what: "table".into(),
                    name: name.clone(),
                    expected: sha1.clone(),
                    found,
                });
            }
        }
        Ok(())
    }

    /// Restore a snapshot: validate, then load every referenced object and
    /// table into native form.
    pub fn snapshot(&self, manifest: &Manifest) -> Result<Snapshot, CacheError> {
        self.validate(manifest)?;
        let mut hllsets = BTreeMap::new();
        let mut tf_vec = None;
        for (name, sha1) in &manifest.objects {
            if HLLSET_OBJECTS.contains(&name.as_str()) {
                let set = self
                    .get_hllset(sha1)?
                    .ok_or_else(|| CacheError::Missing {
                        what: "object".into(),
                        name: name.clone(),
                    })?;
                hllsets.insert(name.clone(), set);
            } else if name == TFVEC_OBJECT {
                tf_vec = self.get_tfvec(sha1)?.or_else(|| None);
            }
        }
        let mut tables = BTreeMap::new();
        for name in manifest.tables.keys() {
            let batch = self.read_table(name)?.ok_or_else(|| CacheError::Missing {
                what: "table".into(),
                name: name.clone(),
            })?;
            tables.insert(name.clone(), batch);
        }
        Ok(Snapshot {
            manifest: manifest.clone(),
            hllsets,
            tf_vec,
            tables,
        })
    }
}
