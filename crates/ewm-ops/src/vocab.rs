//! The boot vocabulary — a content-addressed named dictionary.
//!
//! The DSL compiler resolves names at compile time; the vocabulary is the
//! dictionary it produced. It is itself a content-addressed value: the
//! canonical text (sorted `op:name p:<cid>` / `value:name h:<cid>` rows)
//! hashes to `v:<sha1>`. Same script → same vocabulary → same `v:` CID.

use std::collections::BTreeMap;

use hllset_contracts::sha1_hex;
use serde::{Deserialize, Serialize};

use crate::graph::{OpCid, ValueCid};

/// A named dictionary: program names → program CIDs, value names → value CIDs.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Vocabulary {
    /// `name → p:<sha1>`.
    pub ops: BTreeMap<String, OpCid>,
    /// `name → h:<sha1>`.
    pub values: BTreeMap<String, ValueCid>,
}

impl Vocabulary {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn is_empty(&self) -> bool {
        self.ops.is_empty() && self.values.is_empty()
    }

    /// The canonical text form: one `op:name cid` / `value:name cid` per
    /// line, sorted — the bytes whose SHA1 is the vocabulary CID.
    pub fn canonical(&self) -> String {
        let mut rows: Vec<String> = Vec::new();
        for (name, cid) in &self.ops {
            rows.push(format!("op:{name} {cid}"));
        }
        for (name, cid) in &self.values {
            rows.push(format!("value:{name} {cid}"));
        }
        rows.sort();
        rows.join("\n")
    }

    /// The vocabulary content key: `v:<sha1 of the canonical bytes>`.
    pub fn cid(&self) -> String {
        format!("v:{}", sha1_hex(self.canonical().as_bytes()))
    }

    /// Merge another vocabulary (entries in `other` win on name conflicts,
    /// so a later definition shadows an earlier one).
    pub fn merge(&mut self, other: Vocabulary) {
        self.ops.extend(other.ops);
        self.values.extend(other.values);
    }
}
