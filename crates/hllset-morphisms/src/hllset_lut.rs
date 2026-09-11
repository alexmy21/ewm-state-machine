//! HllsetLut — the `<SHA1, TH>` registry of created HLLSets.
//!
//! The ingest side effect that preserves the HLLSets an application created:
//! every original HLLSet produced by ingestion is registered here by its
//! content key (`h:<sha1>`) and touch-counted each time it is used. Ranking
//! consistency across the collection: tokens rank by TF, registers rank by
//! bit-TF, and original HLLSets rank by **TH** (touch count).
//!
//! Managed exactly like the token LUTs: append-only, idempotent
//! registration, monotonic TH (a CRDT — pointwise max on merge).

use std::collections::BTreeMap;

/// The touch-count registry of created HLLSets, keyed by content key.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct HllsetLut {
    map: BTreeMap<String, u64>,
}

impl HllsetLut {
    pub fn new() -> Self {
        Self::default()
    }

    /// Intern an original HLLSet (append-only, idempotent). A fresh
    /// registration starts at TH = 0; [`touch`](Self::touch) counts uses.
    pub fn register(&mut self, key: &str) {
        self.map.entry(key.to_string()).or_insert(0);
    }

    /// One touch: the HLLSet participated in a compound HLLSet or was used in
    /// some other way. Monotonic.
    pub fn touch(&mut self, key: &str) {
        *self.map.entry(key.to_string()).or_insert(0) += 1;
    }

    /// Touch many keys at once.
    pub fn touch_many<'a>(&mut self, keys: impl IntoIterator<Item = &'a str>) {
        for key in keys {
            self.touch(key);
        }
    }

    /// Touch-count of a key (0 if never seen).
    pub fn th(&self, key: &str) -> u64 {
        self.map.get(key).copied().unwrap_or(0)
    }

    /// Number of registered HLLSets.
    pub fn len(&self) -> usize {
        self.map.len()
    }

    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }

    /// Ranked entries, highest TH first (ties: key order).
    pub fn ranked(&self) -> Vec<(String, u64)> {
        let mut entries: Vec<(String, u64)> =
            self.map.iter().map(|(k, v)| (k.clone(), *v)).collect();
        entries.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
        entries
    }

    /// Merge another registry (CRDT join: pointwise max of touch counts).
    pub fn merge(&mut self, other: &Self) {
        for (key, &th) in &other.map {
            let slot = self.map.entry(key.clone()).or_insert(0);
            *slot = (*slot).max(th);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn register_is_idempotent_and_touch_is_monotonic() {
        let mut lut = HllsetLut::new();
        lut.register("h:aaa");
        lut.register("h:aaa");
        assert_eq!(lut.len(), 1);
        assert_eq!(lut.th("h:aaa"), 0);
        lut.touch("h:aaa");
        lut.touch("h:aaa");
        assert_eq!(lut.th("h:aaa"), 2, "each touch counts");
    }

    #[test]
    fn ranks_by_th_then_key() {
        let mut lut = HllsetLut::new();
        lut.touch("h:a");
        lut.touch("h:b");
        lut.touch("h:b");
        let ranked = lut.ranked();
        assert_eq!(ranked[0], ("h:b".to_string(), 2));
        assert_eq!(ranked[1], ("h:a".to_string(), 1));
    }

    #[test]
    fn merge_is_pointwise_max() {
        let mut a = HllsetLut::new();
        a.touch("h:x");
        a.touch("h:x");
        a.touch("h:y");
        let mut b = HllsetLut::new();
        b.touch("h:x");
        b.touch("h:y");
        b.touch("h:y");
        b.touch("h:y");
        a.merge(&b);
        assert_eq!(a.th("h:x"), 2, "max(2, 1)");
        assert_eq!(a.th("h:y"), 3, "max(1, 3)");
    }
}
