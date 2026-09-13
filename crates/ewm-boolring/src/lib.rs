//! GF(2) Boolean-ring structure over HLLSets — context-index spike.
//!
//! An HLLSet is a subset of the bit plane, hence a vector over GF(2).
//! Symmetric difference (Δ) is addition; intersection (∩) is multiplication:
//!
//! ```text
//! A ∪ B = A Δ B Δ (A ∩ B)        x + x = 0,   x · x = x
//! ```
//!
//! [`BoolBasis`] holds a **reduced row-echelon** GF(2) basis of the span of
//! an inserted collection. Every set in the span has unique coordinates
//! (which basis elements XOR to it); a set outside the span leaves a
//! **residual** — its linear novelty, the part not expressible from the
//! context so far.
//!
//! Scope: this is a *local* context index. The basis depends on insertion
//! order and pivot choice — it is not a global content address.

use hllset_core::HLLSet;

/// `A Δ B` — GF(2) addition.
pub fn symmetric_difference(a: &HLLSet, b: &HLLSet) -> HLLSet {
    a.difference(b).union(&b.difference(a))
}

/// The lowest set bit (the GF(2) pivot candidate), if any.
pub fn lowest_bit(set: &HLLSet) -> Option<u32> {
    set.bit_addresses().into_iter().map(|a| a.bit()).min()
}

/// The result of inserting one set into the basis.
#[derive(Clone, Debug)]
pub enum InsertResult {
    /// The set is in the span; `coords` lists the basis indices that XOR to
    /// it (in elimination order; toggling is safe).
    InSpan { coords: Vec<usize> },
    /// The set was outside the span; it was reduced and the remainder was
    /// added as a new basis element with `pivot`. `rotation_count` is the
    /// number of existing basis elements that were re-pivoted (XORed with the
    /// residual to clear the new pivot bit) — the basis-change component that
    /// is a *rotation* rather than an extension.
    Added {
        residual: HLLSet,
        pivot: u32,
        rotation_count: usize,
    },
}

/// A reduced row-echelon GF(2) basis over HLLSets.
#[derive(Clone, Debug, Default)]
pub struct BoolBasis {
    pub basis: Vec<HLLSet>,
    pub pivots: Vec<u32>,
}

impl BoolBasis {
    pub fn new() -> Self {
        Self::default()
    }

    /// The dimension of the span (number of independent directions seen).
    pub fn dimension(&self) -> usize {
        self.basis.len()
    }

    /// Reduce a set against the basis. Returns the residual (empty if the
    /// set is in the span) and the basis indices used during elimination.
    pub fn reduce(&self, set: &HLLSet) -> (HLLSet, Vec<usize>) {
        let mut cur = set.clone();
        let mut used = Vec::new();
        loop {
            let p = lowest_bit(&cur);
            let Some(p) = p else {
                return (cur, used);
            };
            let j = self.pivots.iter().position(|&q| q == p);
            match j {
                Some(j) => {
                    cur = symmetric_difference(&cur, &self.basis[j]);
                    used.push(j);
                }
                None => return (cur, used),
            }
        }
    }

    /// The part of `set` outside the current span (linear novelty).
    pub fn residual(&self, set: &HLLSet) -> HLLSet {
        self.reduce(set).0
    }

    /// The unique GF(2) coordinates of a set in the span; `None` when the
    /// set is outside the span.
    pub fn coordinates(&self, set: &HLLSet) -> Option<Vec<bool>> {
        let (residual, used) = self.reduce(set);
        if !residual.is_empty() {
            return None;
        }
        let mut coords = vec![false; self.basis.len()];
        for j in used {
            coords[j] = !coords[j];
        }
        Some(coords)
    }

    /// Insert one set, maintaining reduced row-echelon form.
    pub fn insert(&mut self, set: &HLLSet) -> InsertResult {
        let (residual, coords) = self.reduce(set);
        let p = lowest_bit(&residual);
        let Some(p) = p else {
            return InsertResult::InSpan { coords };
        };
        // Clear the new pivot from every existing basis element so each
        // pivot bit appears in exactly one basis element. This re-pivoting is
        // the *rotation* component of the basis change; count how many
        // existing directions it touches (the rest is pure extension).
        let mut rotation_count = 0usize;
        for b in self.basis.iter_mut() {
            if b.has_bit(p / 32, p % 32) {
                let cleared = symmetric_difference(b, &residual);
                *b = cleared;
                rotation_count += 1;
            }
        }
        self.pivots.push(p);
        self.basis.push(residual.clone());
        InsertResult::Added {
            residual,
            pivot: p,
            rotation_count,
        }
    }
}

/// Build a basis from an iterator of sets (insertion order determines the
/// basis; the span itself is order-independent).
pub fn span_basis<'a>(sets: impl IntoIterator<Item = &'a HLLSet>) -> BoolBasis {
    let mut basis = BoolBasis::new();
    for set in sets {
        basis.insert(set);
    }
    basis
}

#[cfg(test)]
mod tests {
    use super::*;

    fn set(bits: &[u32]) -> HLLSet {
        let mut h = HLLSet::new();
        for b in bits {
            h.add_bit(*b);
        }
        h
    }

    #[test]
    fn symmetric_difference_axioms() {
        let a = set(&[1, 2, 3]);
        let b = set(&[2, 3, 4]);
        assert!(symmetric_difference(&a, &a).is_empty(), "A Δ A = ∅");
        assert_eq!(
            symmetric_difference(&a, &HLLSet::new()).popcount(),
            a.popcount(),
            "A Δ ∅ = A"
        );
        assert_eq!(
            symmetric_difference(&a, &b).popcount(),
            2,
            "A Δ B = {{1, 4}}"
        );
    }

    #[test]
    fn union_decomposes_in_the_ring() {
        let a = set(&[1, 2, 3]);
        let b = set(&[2, 3, 4]);
        // A ∪ B = A Δ B Δ (A ∩ B)
        let lhs = a.union(&b);
        let rhs = symmetric_difference(&symmetric_difference(&a, &b), &a.intersection(&b));
        assert_eq!(lhs.popcount(), rhs.popcount());
        assert!(lhs.difference(&rhs).is_empty() && rhs.difference(&lhs).is_empty());
    }

    #[test]
    fn span_dimension_and_coordinates() {
        let a = set(&[1, 2, 3]);
        let b = set(&[2, 3, 4]);
        let ab = symmetric_difference(&a, &b); // {1, 4}

        let mut basis = BoolBasis::new();
        assert!(matches!(basis.insert(&a), InsertResult::Added { .. }));
        assert!(matches!(basis.insert(&b), InsertResult::Added { .. }));
        assert_eq!(basis.dimension(), 2);

        // A Δ B is in the span; reconstruct it from its coordinates.
        let coords = basis.coordinates(&ab).expect("A Δ B is in the span");
        let mut rebuilt = HLLSet::new();
        for (i, c) in coords.iter().enumerate() {
            if *c {
                rebuilt = symmetric_difference(&rebuilt, &basis.basis[i]);
            }
        }
        assert!(rebuilt.difference(&ab).is_empty() && ab.difference(&rebuilt).is_empty(),
                "coordinates reconstruct A Δ B");
    }

    #[test]
    fn residual_is_linear_novelty() {
        let a = set(&[1, 2, 3]);
        let b = set(&[2, 3, 4]);
        let c = set(&[5, 6]); // introduces two new independent directions

        let mut basis = BoolBasis::new();
        basis.insert(&a);
        basis.insert(&b);
        assert_eq!(basis.dimension(), 2);

        let res = basis.residual(&c);
        assert!(!res.is_empty(), "C has new directions");
        assert!(basis.coordinates(&c).is_none(), "C is outside the span");

        match basis.insert(&c) {
            InsertResult::Added { residual, .. } => {
                assert!(!residual.is_empty());
            }
            InsertResult::InSpan { .. } => panic!("C must be added"),
        }
        assert_eq!(basis.dimension(), 3);
        assert!(basis.coordinates(&c).is_some(), "C is now in the span");
    }

    #[test]
    fn span_is_order_independent_though_basis_is_not() {
        let a = set(&[1, 2, 3]);
        let b = set(&[2, 3, 4]);
        let mut b1 = BoolBasis::new();
        b1.insert(&a);
        b1.insert(&b);
        let mut b2 = BoolBasis::new();
        b2.insert(&b);
        b2.insert(&a);
        assert_eq!(b1.dimension(), b2.dimension(), "the span has one size");
        assert!(b1.coordinates(&a).is_some());
        assert!(b2.coordinates(&a).is_some());
    }

    #[test]
    fn insertion_reports_the_rotation_component() {
        // Pure extension: the new pivot bit is absent from every existing
        // basis element -> nothing rotates.
        let mut basis = BoolBasis::new();
        match basis.insert(&set(&[1, 2])) {
            InsertResult::Added { rotation_count, .. } => assert_eq!(rotation_count, 0),
            InsertResult::InSpan { .. } => panic!("first set is added"),
        }
        match basis.insert(&set(&[4, 5])) {
            InsertResult::Added { rotation_count, .. } => assert_eq!(rotation_count, 0),
            InsertResult::InSpan { .. } => panic!("disjoint set is added"),
        }

        // Rotation: B1 = {1, 3} (pivot 1), insert R = {3, 5} (new pivot 3).
        // B1 contains bit 3, so it is re-pivoted to {1, 5} — one touched
        // element.
        let mut basis = BoolBasis::new();
        basis.insert(&set(&[1, 3]));
        match basis.insert(&set(&[3, 5])) {
            InsertResult::Added {
                residual,
                rotation_count,
                ..
            } => {
                assert_eq!(rotation_count, 1, "B1 shares the new pivot bit");
                assert_eq!(residual.popcount(), 2, "R = {{3, 5}}");
            }
            InsertResult::InSpan { .. } => panic!("R is outside the span"),
        }
        assert_eq!(basis.dimension(), 2);

        // In-span pushes never rotate.
        let mut basis = BoolBasis::new();
        basis.insert(&set(&[1, 2]));
        match basis.insert(&set(&[1, 2])) {
            InsertResult::InSpan { .. } => {}
            InsertResult::Added { .. } => panic!("duplicate is in the span"),
        }
    }

    #[test]
    fn ring_stats_carry_rotation_mass() {
        let mut w = BoolWindow::new(8);
        let s1 = w.push(&set(&[1, 2]));
        assert_eq!(s1.rotation_count, 0);
        assert_eq!(s1.rotation_mass, 0);
        // {1, 3} shares the new pivot of a second set whose residual has a
        // new lowest bit that B1 contains -> one element re-pivots.
        let mut w = BoolWindow::new(8);
        w.push(&set(&[1, 3]));
        let s2 = w.push(&set(&[3, 5]));
        assert_eq!(s2.rotation_count, 1);
        assert_eq!(s2.rotation_mass, s2.rotation_count * s2.residual);
    }
}

/// Per-push statistics of the windowed ring.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RingStats {
    /// Popcount of the incoming set's residual against the window *before*
    /// insertion — its linear novelty.
    pub residual: u64,
    /// `true` when the incoming set was already expressible from the window.
    pub in_span: bool,
    /// Window-span dimension after insertion (and after any eviction).
    pub dimension: usize,
    /// Number of existing basis elements re-pivoted by the insertion (the
    /// rotation component of the basis change). Zero when `in_span` — an
    /// in-span push never changes the basis.
    pub rotation_count: u64,
    /// Total Hamming change of the old basis: `rotation_count * residual`.
    /// Together with `residual` (the extension, one new element) the whole
    /// basis content change is `(rotation_count + 1) * residual`.
    pub rotation_mass: u64,
}

/// The Boolean ring as a moving window over **original** HLLSets.
///
/// Originals enter in ingestion order (a queue); the basis is the RREF
/// span of the current window. Insertion is one Gaussian step; eviction
/// recomputes the basis from the remaining originals (cheap at cache
/// sizes). Because the sequence is fixed by ingestion, the basis is
/// **deterministic for the window** — the canonical-basis problem is
/// resolved by never comparing across different windows.
#[derive(Clone, Debug)]
pub struct BoolWindow {
    max_len: usize,
    originals: std::collections::VecDeque<HLLSet>,
    basis: BoolBasis,
}

impl BoolWindow {
    pub fn new(max_len: usize) -> Self {
        Self {
            max_len: max_len.max(1),
            originals: std::collections::VecDeque::new(),
            basis: BoolBasis::new(),
        }
    }

    pub fn window_len(&self) -> usize {
        self.originals.len()
    }

    pub fn dimension(&self) -> usize {
        self.basis.dimension()
    }

    pub fn basis(&self) -> &BoolBasis {
        &self.basis
    }

    pub fn originals(&self) -> impl Iterator<Item = &HLLSet> {
        self.originals.iter()
    }

    /// Push one original (in ingestion order); evict the oldest when the
    /// window exceeds `max_len` and recompute the basis.
    pub fn push(&mut self, set: &HLLSet) -> RingStats {
        let result = self.basis.insert(set);
        let (residual, in_span, rotation_count) = match &result {
            InsertResult::InSpan { .. } => (0u64, true, 0u64),
            InsertResult::Added {
                residual,
                rotation_count,
                ..
            } => (residual.popcount(), false, *rotation_count as u64),
        };
        self.originals.push_back(set.clone());
        let mut dimension = self.basis.dimension();
        if self.originals.len() > self.max_len {
            self.originals.pop_front();
            self.recompute();
            dimension = self.basis.dimension();
        }
        RingStats {
            residual,
            in_span,
            dimension,
            rotation_count,
            rotation_mass: rotation_count * residual,
        }
    }

    /// Recompute the basis from the current originals (after eviction).
    pub fn recompute(&mut self) {
        self.basis = BoolBasis::new();
        for set in &self.originals {
            self.basis.insert(set);
        }
    }

    /// The part of `set` outside the window span (linear novelty).
    pub fn residual(&self, set: &HLLSet) -> HLLSet {
        self.basis.residual(set)
    }

    /// The GF(2) coordinates of a set in the window span.
    pub fn coordinates(&self, set: &HLLSet) -> Option<Vec<bool>> {
        self.basis.coordinates(set)
    }
}

#[cfg(test)]
mod window_tests {
    use super::*;

    fn set(bits: &[u32]) -> HLLSet {
        let mut h = HLLSet::new();
        for b in bits {
            h.add_bit(*b);
        }
        h
    }

    #[test]
    fn same_sequence_gives_the_same_basis() {
        let a = set(&[1, 2, 3]);
        let b = set(&[2, 3, 4]);
        let c = set(&[3, 4, 5]);

        let mut w1 = BoolWindow::new(8);
        let mut w2 = BoolWindow::new(8);
        for s in [&a, &b, &c] {
            w1.push(s);
            w2.push(s);
        }
        assert_eq!(w1.basis().pivots, w2.basis().pivots,
            "a fixed ingestion order fixes the basis");
        assert_eq!(w1.dimension(), w2.dimension());
    }

    #[test]
    fn eviction_slides_the_window_and_recomputes() {
        let a = set(&[1, 2, 3]);
        let b = set(&[2, 3, 4]);
        let c = set(&[5, 6]); // independent of {a, b}? {5,6} vs span of a,b -> independent

        let mut w = BoolWindow::new(2);
        let s1 = w.push(&a);
        assert!(!s1.in_span && s1.residual == a.popcount());
        let s2 = w.push(&b);
        assert_eq!(w.window_len(), 2);
        let _ = s2;
        let s3 = w.push(&c); // evicts a; window = {b, c}
        assert_eq!(w.window_len(), 2);
        assert_eq!(s3.residual, c.popcount(), "c is outside the {{b}} span of the window");
        // a is gone: its residual against the window is nonempty.
        assert!(!w.residual(&a).is_empty(), "evicted original is outside the window span");
    }

    #[test]
    fn in_span_push_has_zero_novelty() {
        let a = set(&[1, 2, 3]);
        let mut w = BoolWindow::new(8);
        let s1 = w.push(&a);
        assert!(!s1.in_span);
        let s2 = w.push(&a);
        assert!(s2.in_span);
        assert_eq!(s2.residual, 0);
        assert_eq!(w.dimension(), 1);
    }
}
