//! HllsetLut — the `<name, SHA1, TH>` registry of created HLLSets.
//!
//! The ingest side effect that preserves the HLLSets an application created:
//! every original HLLSet produced by ingestion is registered here by its
//! content key (`h:<sha1>`) and touch-counted each time it is used. Ranking
//! consistency across the collection: tokens rank by TF, registers rank by
//! bit-TF, and original HLLSets rank by **TH** (touch count).
//!
//! Named HLLSets (G1, G2, G3, …) are registered under their name: the name
//! is a label on the entry, the key is the identity. A named HLLSet is
//! immutable — every update creates a **new** HLLSet (new key); the old one
//! stays registered.
//!
//! Managed exactly like the token LUTs: append-only, idempotent
//! registration, monotonic TH (a CRDT — pointwise max on merge).

use std::collections::BTreeMap;

/// The touch-count registry of created HLLSets, keyed by `(name, content key)`.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct HllsetLut {
    map: BTreeMap<(String, String), u64>,
}

/// The name used for entries registered without an explicit name.
pub const UNNAMED: &str = "";

impl HllsetLut {
    pub fn new() -> Self {
        Self::default()
    }

    /// Intern an original HLLSet under a name (append-only, idempotent).
    /// A fresh registration starts at TH = 0; [`touch_named`](Self::touch_named)
    /// counts uses.
    pub fn register_named(&mut self, name: &str, key: &str) {
        self.map
            .entry((name.to_string(), key.to_string()))
            .or_insert(0);
    }

    /// One touch of a named HLLSet. Monotonic.
    pub fn touch_named(&mut self, name: &str, key: &str) {
        *self.map
            .entry((name.to_string(), key.to_string()))
            .or_insert(0) += 1;
    }

    /// Touch-count of a named HLLSet (0 if never seen).
    pub fn th_named(&self, name: &str, key: &str) -> u64 {
        self.map
            .get(&(name.to_string(), key.to_string()))
            .copied()
            .unwrap_or(0)
    }

    /// Unnamed convenience: register under [`UNNAMED`].
    pub fn register(&mut self, key: &str) {
        self.register_named(UNNAMED, key);
    }

    /// Unnamed convenience: touch under [`UNNAMED`].
    pub fn touch(&mut self, key: &str) {
        self.touch_named(UNNAMED, key);
    }

    /// Unnamed convenience: TH under [`UNNAMED`].
    pub fn th(&self, key: &str) -> u64 {
        self.th_named(UNNAMED, key)
    }

    /// Touch many unnamed keys at once.
    pub fn touch_many<'a>(&mut self, keys: impl IntoIterator<Item = &'a str>) {
        for key in keys {
            self.touch(key);
        }
    }

    /// Number of registered `(name, key)` entries.
    pub fn len(&self) -> usize {
        self.map.len()
    }

    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }

    /// Ranked entries as `(name, key, th)`, highest TH first (ties: name,
    /// then key order).
    pub fn ranked(&self) -> Vec<(String, String, u64)> {
        let mut entries: Vec<(String, String, u64)> = self
            .map
            .iter()
            .map(|((name, key), v)| (name.clone(), key.clone(), *v))
            .collect();
        entries.sort_by(|a, b| b.2.cmp(&a.2).then(a.0.cmp(&b.0)).then(a.1.cmp(&b.1)));
        entries
    }

    /// Merge another registry (CRDT join: pointwise max of touch counts).
    pub fn merge(&mut self, other: &Self) {
        for ((name, key), &th) in &other.map {
            let slot = self
                .map
                .entry((name.clone(), key.clone()))
                .or_insert(0);
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
    fn named_entries_are_separate_from_unnamed() {
        let mut lut = HllsetLut::new();
        lut.register_named("G1", "h:abc");
        lut.touch_named("G1", "h:abc");
        lut.touch("h:abc");
        assert_eq!(lut.len(), 2, "(G1, h:abc) and (\"\", h:abc) are distinct");
        assert_eq!(lut.th_named("G1", "h:abc"), 1);
        assert_eq!(lut.th("h:abc"), 1);
    }

    #[test]
    fn ranks_by_th_then_name_then_key() {
        let mut lut = HllsetLut::new();
        lut.touch("h:a");
        lut.touch("h:b");
        lut.touch("h:b");
        let ranked = lut.ranked();
        assert_eq!(ranked[0], (UNNAMED.to_string(), "h:b".to_string(), 2));
        assert_eq!(ranked[1], (UNNAMED.to_string(), "h:a".to_string(), 1));
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
