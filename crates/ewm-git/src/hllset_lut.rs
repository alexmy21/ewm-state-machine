//! HLLSet-LUT and bit-TF — the two frequency objects of the lattice state.
//!
//! Ranking consistency: tokens rank by TF, registers rank by bit-TF, and
//! original HLLSets rank by **TH** (touch count). No lattice degree is
//! needed — the union of all original HLLSets is the lattice **Top**, the
//! original HLLSets are the natural **bottom** (atoms), and compound
//! HLLSets are never stored explicitly.
//!
//! - `HllsetLut` — `<SHA1, TH>`: how often a given original HLLSet was
//!   part of a compound HLLSet or used in any other way (each touch
//!   counts). Managed exactly like the token/catalog LUTs: append-only,
//!   idempotent registration, monotonic TH.
//! - `BitTf` — the 32K integer vector holding TF for each bit of the
//!   lattice top (the union of all original HLLSets). Monotonic CRDT;
//!   each touch of an HLLSet increments its set bits.

use std::collections::HashMap;

use hllset_core::core::tfvec::TFVec;
use hllset_core::HLLSet;

use crate::object::ObjectId;

/// `<SHA1, TH>` — the touch-count registry of original HLLSets.
#[derive(Clone, Debug, Default)]
pub struct HllsetLut {
    map: HashMap<ObjectId, u64>,
}

impl HllsetLut {
    pub fn new() -> Self {
        Self::default()
    }

    /// Intern an original HLLSet (append-only, idempotent).
    pub fn register(&mut self, id: &ObjectId) {
        self.map.entry(id.clone()).or_insert(0);
    }

    /// One touch: the HLLSet participated in a compound HLLSet or was used
    /// in some other way. Monotonic.
    pub fn touch(&mut self, id: &ObjectId) {
        *self.map.entry(id.clone()).or_insert(0) += 1;
    }

    /// Touch many original HLLSets at once.
    pub fn touch_many<'a>(&mut self, ids: impl IntoIterator<Item = &'a ObjectId>) {
        for id in ids {
            self.touch(id);
        }
    }

    /// Touch-count of an original HLLSet (0 if never seen).
    pub fn th(&self, id: &ObjectId) -> u64 {
        self.map.get(id).copied().unwrap_or(0)
    }

    /// Number of registered original HLLSets.
    pub fn len(&self) -> usize {
        self.map.len()
    }

    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }

    /// Ranked entries, highest TH first (ties: address order).
    pub fn ranked(&self) -> Vec<(ObjectId, u64)> {
        let mut entries: Vec<(ObjectId, u64)> = self.map.iter().map(|(k, v)| (k.clone(), *v)).collect();
        entries.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
        entries
    }
}

/// The 32K bit-TF vector over the lattice top.
#[derive(Clone, Debug, PartialEq)]
pub struct BitTf {
    inner: TFVec,
}

impl Default for BitTf {
    fn default() -> Self {
        Self::new()
    }
}

impl BitTf {
    /// A zeroed bit-TF vector (32768 entries).
    pub fn new() -> Self {
        Self { inner: TFVec::new() }
    }

    /// Wrap an existing [`TFVec`].
    pub fn from_tfvec(inner: TFVec) -> Self {
        Self { inner }
    }

    /// One touch of an HLLSet: every set bit gets +1 TF.
    pub fn touch(&mut self, hllset: &HLLSet) {
        self.inner.increment_from_hllset(hllset, 1.0);
    }

    /// TF of one bit position.
    pub fn tf(&self, bit: usize) -> f64 {
        self.inner.get(bit)
    }

    /// Total TF across all bits.
    pub fn total(&self) -> f64 {
        self.inner.total()
    }

    /// Number of bit positions with nonzero TF.
    pub fn active_bits(&self) -> usize {
        self.inner.values.iter().filter(|&&v| v > 0.0).count()
    }

    /// Monotonic CRDT merge (element-wise max/sum per TFVec semantics).
    pub fn merge(&mut self, other: &BitTf) {
        self.inner.merge(&other.inner);
    }

    /// Serialize the full vector (262148 bytes).
    pub fn to_bytes(&self) -> Vec<u8> {
        self.inner.to_bytes()
    }

    pub fn from_bytes(bytes: &[u8]) -> Option<Self> {
        TFVec::from_bytes(bytes).map(|inner| Self { inner })
    }

    pub fn len(&self) -> usize {
        self.inner.len()
    }

    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hll(tokens: &[&str]) -> HLLSet {
        HLLSet::from_tokens(tokens.iter())
    }

    #[test]
    fn hllset_lut_registers_idempotently_and_counts_touches() {
        let id = ObjectId::of_bytes(b"hllset");
        let mut lut = HllsetLut::new();
        lut.register(&id);
        lut.register(&id);
        assert_eq!(lut.len(), 1);
        assert_eq!(lut.th(&id), 0);
        lut.touch(&id);
        lut.touch(&id);
        assert_eq!(lut.th(&id), 2, "each touch counts");
    }

    #[test]
    fn hllset_lut_ranks_by_th() {
        let a = ObjectId::of_bytes(b"a");
        let b = ObjectId::of_bytes(b"b");
        let mut lut = HllsetLut::new();
        lut.touch(&a);
        lut.touch(&b);
        lut.touch(&b);
        let ranked = lut.ranked();
        assert_eq!(ranked[0], (b, 2));
        assert_eq!(ranked[1], (a, 1));
    }

    #[test]
    fn bit_tf_touches_are_monotonic_and_merge() {
        let mut tf = BitTf::new();
        let set = hll(&["a", "b"]);
        tf.touch(&set);
        tf.touch(&set);
        assert_eq!(tf.active_bits(), set.popcount() as usize);
        assert_eq!(tf.total(), 2.0 * set.popcount() as f64);

        let mut tf2 = BitTf::new();
        tf2.touch(&hll(&["b", "c"]));
        tf.merge(&tf2);
        assert!(tf.total() >= 2.0 * set.popcount() as f64, "merge is monotonic");
    }

    #[test]
    fn bit_tf_roundtrip() {
        let mut tf = BitTf::new();
        tf.touch(&hll(&["round", "trip"]));
        let bytes = tf.to_bytes();
        let restored = BitTf::from_bytes(&bytes).unwrap();
        assert_eq!(restored, tf);
    }
}
