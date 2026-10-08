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

    // ── Ring + lattice tools ────────────────────────────────────────────────

    /// Rebuild the set selected by GF(2) coordinates: the XOR of the marked
    /// basis elements. Inverse of [`Self::coordinates`] on the span.
    pub fn reconstruct(&self, coords: &[bool]) -> HLLSet {
        let mut x = HLLSet::new();
        for (c, b) in coords.iter().zip(&self.basis) {
            if *c {
                x = symmetric_difference(&x, b);
            }
        }
        x
    }

    /// The **cover** of the basis: the union of every basis element.
    ///
    /// `cover(basis) == union(originals)`. Every basis element is an XOR of
    /// generators, hence a subset of their union (`XOR ⊆ OR`); every generator
    /// is an XOR of basis elements, hence a subset of the cover. The cover is
    /// the bit plane the ring can see, so `x \ cover` is bit-level novelty no
    /// span member can contain.
    pub fn cover(&self) -> HLLSet {
        let mut c = HLLSet::new();
        for b in &self.basis {
            c = c.union(b);
        }
        c
    }

    /// The BSS soft key of `x` against the basis: `w_i = |x ∩ B_i| / |B_i|`.
    /// This is the **lattice measurement**: any HLLSet can be measured this
    /// way, in span or not.
    pub fn bss_vector(&self, x: &HLLSet) -> Vec<f64> {
        self.basis
            .iter()
            .map(|b| {
                let denom = b.popcount();
                if denom == 0 {
                    1.0
                } else {
                    x.intersection(b).popcount() as f64 / denom as f64
                }
            })
            .collect()
    }

    /// The **lattice projection** of `x` onto the basis cover. See
    /// [`CoverProjection`].
    pub fn project(&self, x: &HLLSet) -> CoverProjection {
        let cover = self.cover();
        CoverProjection {
            retained: x.intersection(&cover),
            novelty: x.difference(&cover),
            departed: cover.difference(x),
            cover,
        }
    }

    /// The **change of basis** from `from` to `self`: for each basis element of
    /// `from`, its GF(2) coordinates in `self` (`None` when it is outside this
    /// span — excluded by monotone growth). Column `i` is the image of
    /// `from.basis[i]`, so a coordinate vector `a` in `from` transports to
    /// `a · M` in `self`.
    pub fn change_of_basis(&self, from: &BoolBasis) -> Vec<Option<Vec<bool>>> {
        from.basis.iter().map(|b| self.coordinates(b)).collect()
    }

    /// The minimal **linear cover** of an in-span set: the support of its
    /// unique coordinates. `None` when the set is outside the span.
    pub fn linear_cover(&self, x: &HLLSet) -> Option<Vec<usize>> {
        self.coordinates(x).map(|coords| {
            coords
                .iter()
                .enumerate()
                .filter_map(|(i, &c)| c.then_some(i))
                .collect()
        })
    }

    /// A greedy **set cover** of `x` by basis elements: repeatedly take the
    /// basis element adding the most uncovered bits of `x`. Greedy is
    /// `H(d)`-approximate for the minimum union cover (`d` = dimension); the
    /// bits `x \ cover` cannot be covered by any basis subset.
    pub fn minimal_cover(&self, x: &HLLSet) -> Vec<usize> {
        let mut covered = HLLSet::new();
        let mut chosen: Vec<usize> = Vec::new();
        while !x.difference(&covered).is_empty() {
            let mut best: Option<(usize, u64)> = None;
            for (i, b) in self.basis.iter().enumerate() {
                if chosen.contains(&i) {
                    continue;
                }
                let gain = x.difference(&covered).intersection(b).popcount();
                if gain > 0 && best.map_or(true, |(_, g)| gain > g) {
                    best = Some((i, gain));
                }
            }
            match best {
                Some((i, _)) => {
                    covered = covered.union(&self.basis[i]);
                    chosen.push(i);
                }
                None => break, // only un-coverable bits remain
            }
        }
        chosen
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

/// The lattice projection of a set onto a basis **cover** (union of the basis
/// elements): the bits shared with the context, the bits wholly outside it,
/// and the context bits the set no longer carries. `retained` and `novelty`
/// are disjoint and `x = retained ⊔ novelty`.
#[derive(Clone, Debug)]
pub struct CoverProjection {
    /// `x ∩ cover` — the part of `x` the ring can see.
    pub retained: HLLSet,
    /// `x \ cover` — bit-level novelty: bits no span member can contain.
    pub novelty: HLLSet,
    /// `cover \ x` — context bits this set has departed from.
    pub departed: HLLSet,
    /// The cover itself (union of the basis elements).
    pub cover: HLLSet,
}

impl CoverProjection {
    /// The retained / novelty / departed popcounts `(R, N, D)`.
    pub fn popcounts(&self) -> (u64, u64, u64) {
        (
            self.retained.popcount(),
            self.novelty.popcount(),
            self.departed.popcount(),
        )
    }
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

/// The Boolean ring over **original** HLLSets.
///
/// Two roles are deliberately decoupled:
///
/// - the **generator basis** is *monotone* — every original ever pushed is a
///   permanent basis element, so the span only grows and a set that is
///   representable stays representable (`span(t) ⊆ span(t+1)`);
/// - the **visibility window** (`max_len`) is the most recent originals — the
///   *context* the window exposes. Eviction moves this window; it does not
///   remove a generator.
///
/// Insertion is one Gaussian step. Because the ingestion order is fixed, the
/// basis is deterministic for a sequence: coordinates are comparable inside a
/// run and across commits (the canonical-basis problem is resolved by fixing
/// the generators and their order, not by bounding them).
///
/// [`BoolWindow::windowed`] restores the legacy scene-bounded mode, where the
/// window *is* the generator set and eviction rebuilds the basis.
#[derive(Clone, Debug)]
pub struct BoolWindow {
    /// Visibility capacity: how many recent originals the window keeps.
    max_len: usize,
    /// The visible originals (most recent `max_len`), in ingestion order.
    originals: std::collections::VecDeque<HLLSet>,
    /// The generator basis. Monotone by default; rebuilt from the visible
    /// window on eviction only in [`BoolWindow::windowed`] mode.
    basis: BoolBasis,
    /// Total originals ever pushed — the generator history length.
    total: u64,
    /// Monotonic stamp of the basis **content**: bumped whenever the basis
    /// changes (extension/rotation on push, or rebuild in windowed mode). A
    /// soft key / coordinate vector measured at an older generation is stale
    /// for the current basis.
    generation: u64,
    /// `true`: eviction drops the oldest generator and rebuilds from the
    /// visible window (legacy). `false`: the basis is monotone.
    evict_basis: bool,
}

impl BoolWindow {
    /// A monotone ring with a visibility window of `max_len` originals.
    pub fn new(max_len: usize) -> Self {
        Self::with_mode(max_len, false)
    }

    /// The legacy windowed ring: generators and visibility are the same set,
    /// and eviction rebuilds the basis from the remaining window. Kept for
    /// scene-bounded novelty analysis; the production ring uses [`Self::new`].
    pub fn windowed(max_len: usize) -> Self {
        Self::with_mode(max_len, true)
    }

    fn with_mode(max_len: usize, evict_basis: bool) -> Self {
        Self {
            max_len: max_len.max(1),
            originals: std::collections::VecDeque::new(),
            basis: BoolBasis::new(),
            total: 0,
            generation: 0,
            evict_basis,
        }
    }

    /// Number of originals currently visible (bounded by the capacity).
    pub fn window_len(&self) -> usize {
        self.originals.len()
    }

    /// Total originals ever pushed — the length of the generator history.
    pub fn total(&self) -> u64 {
        self.total
    }

    /// `true` for the monotone (default) mode: generators are never dropped.
    pub fn is_monotone(&self) -> bool {
        !self.evict_basis
    }

    pub fn dimension(&self) -> usize {
        self.basis.dimension()
    }

    pub fn basis(&self) -> &BoolBasis {
        &self.basis
    }

    /// The basis-content generation. Same generation ⟹ same basis, so cached
    /// projections keyed by this stamp stay valid; a bump means every older
    /// projection is stale.
    pub fn generation(&self) -> u64 {
        self.generation
    }

    /// The visible originals, oldest first.
    pub fn originals(&self) -> impl Iterator<Item = &HLLSet> {
        self.originals.iter()
    }

    /// Push one original (in ingestion order). The basis grows monotonically;
    /// the oldest original leaves the *visibility* window when `max_len` is
    /// exceeded. In windowed mode only, eviction also rebuilds the basis from
    /// the remaining window.
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
        if matches!(result, InsertResult::Added { .. }) {
            self.generation += 1;
        }
        self.originals.push_back(set.clone());
        self.total += 1;
        let mut dimension = self.basis.dimension();
        if self.originals.len() > self.max_len {
            self.originals.pop_front();
            if self.evict_basis {
                self.rebuild_from_window();
            }
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

    /// Rebuild the basis from the visible window after eviction (windowed
    /// mode only; in monotone mode this is never reached because no generator
    /// is dropped). The basis content changes, so the generation is bumped.
    fn rebuild_from_window(&mut self) {
        self.basis = BoolBasis::new();
        for set in &self.originals {
            self.basis.insert(set);
        }
        self.generation += 1;
    }

    /// The part of `set` outside the span (linear novelty).
    pub fn residual(&self, set: &HLLSet) -> HLLSet {
        self.basis.residual(set)
    }

    /// The GF(2) coordinates of a set in the span.
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
    fn monotone_window_bounds_visibility_not_the_span() {
        let a = set(&[1, 2, 3]);
        let b = set(&[2, 3, 4]);
        let c = set(&[5, 6]);

        let mut w = BoolWindow::new(2);
        let s1 = w.push(&a);
        assert!(!s1.in_span && s1.residual == a.popcount());
        w.push(&b);
        assert_eq!(w.window_len(), 2);
        let s3 = w.push(&c); // moves the visibility window; the basis keeps a
        assert_eq!(w.window_len(), 2);
        assert_eq!(w.total(), 3);
        assert_eq!(s3.residual, c.popcount(), "c is outside the {{a, b}} span");
        assert!(w.residual(&a).is_empty(), "a stays in the monotone span");
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

    #[test]
    fn generation_bumps_exactly_when_the_basis_changes() {
        let a = set(&[1, 2, 3]);
        let b = set(&[2, 3, 4]);
        let mut w = BoolWindow::new(8);
        assert_eq!(w.generation(), 0, "empty window");

        w.push(&a); // Added -> basis changes
        assert_eq!(w.generation(), 1);

        w.push(&a); // in-span -> basis unchanged
        assert_eq!(w.generation(), 1);

        w.push(&b); // Added -> basis changes
        assert_eq!(w.generation(), 2);

        w.push(&a); // in-span under the monotone basis -> unchanged
        assert_eq!(w.generation(), 2);
    }

    #[test]
    fn monotone_window_keeps_evicted_generators() {
        let a = set(&[1, 2, 3]);
        let b = set(&[2, 3, 4]);
        let c = set(&[5, 6]);

        let mut w = BoolWindow::new(2); // visibility cap 2, monotone basis
        w.push(&a);
        w.push(&b);
        assert_eq!(w.window_len(), 2);
        w.push(&c); // a leaves the *visibility* window, not the basis
        assert_eq!(w.window_len(), 2);
        assert_eq!(w.total(), 3);
        assert!(w.is_monotone());
        assert_eq!(w.dimension(), 3, "the span keeps every generator");
        assert!(w.residual(&a).is_empty(), "a stays representable");
        assert!(w.coordinates(&a).is_some());
    }

    #[test]
    fn monotone_span_is_inclusion() {
        let sets = [
            set(&[1, 2, 3]),
            set(&[2, 3, 4]),
            set(&[3, 4, 5]),
            set(&[9, 10]),
        ];
        let mut w = BoolWindow::new(2);
        let mut seen: Vec<HLLSet> = Vec::new();
        for s in &sets {
            w.push(s);
            seen.push(s.clone());
            for prev in &seen {
                assert!(
                    w.coordinates(prev).is_some(),
                    "span(t) subset span(t+1): representable never leaves"
                );
            }
        }
    }

    #[test]
    fn windowed_mode_evicts_and_rebuilds() {
        let a = set(&[1, 2, 3]);
        let b = set(&[2, 3, 4]);
        let c = set(&[5, 6]);

        let mut w = BoolWindow::windowed(2);
        w.push(&a);
        w.push(&b);
        assert_eq!(w.window_len(), 2);
        w.push(&c); // evicts a and rebuilds the basis from {b, c}
        assert!(!w.is_monotone());
        assert_eq!(w.window_len(), 2);
        assert_eq!(w.total(), 3);
        assert_eq!(w.dimension(), 2, "basis is the visible window's span");
        assert!(!w.residual(&a).is_empty(), "evicted generator leaves the span");
    }

    #[test]
    fn windowed_eviction_bumps_generation() {
        let a = set(&[1, 2, 3]);
        let b = set(&[2, 3, 4]);
        let c = set(&[5, 6]);
        let mut w = BoolWindow::windowed(2);
        w.push(&a);
        w.push(&b);
        let before = w.generation();
        w.push(&c); // evicts and rebuilds -> generation bumps
        assert!(w.generation() > before);
    }
}

#[cfg(test)]
mod tool_tests {
    use super::*;

    fn set(bits: &[u32]) -> HLLSet {
        let mut h = HLLSet::new();
        for b in bits {
            h.add_bit(*b);
        }
        h
    }

    fn same(a: &HLLSet, b: &HLLSet) -> bool {
        a.difference(b).is_empty() && b.difference(a).is_empty()
    }

    #[test]
    fn cover_is_the_union_of_the_generators() {
        let a = set(&[1, 2, 3]);
        let b = set(&[3, 4]);
        let c = set(&[5]);
        let mut basis = BoolBasis::new();
        for s in [&a, &b, &c] {
            basis.insert(s);
        }
        let cover = basis.cover();
        let expected = a.union(&b).union(&c);
        assert!(cover.difference(&expected).is_empty());
        assert!(expected.difference(&cover).is_empty());
    }

    #[test]
    fn cover_identity_holds_regardless_of_insertion_order() {
        let a = set(&[1, 2, 3]);
        let b = set(&[2, 3, 4]);
        let c = set(&[4, 5]);
        let mut b1 = BoolBasis::new();
        for s in [&a, &b, &c] {
            b1.insert(s);
        }
        let mut b2 = BoolBasis::new();
        for s in [&c, &a, &b] {
            b2.insert(s);
        }
        let union = a.union(&b).union(&c);
        for basis in [&b1, &b2] {
            let cover = basis.cover();
            assert!(cover.difference(&union).is_empty() && union.difference(&cover).is_empty());
        }
    }

    #[test]
    fn bss_vector_measures_any_set() {
        // disjoint generators keep the basis elements literal: {1,2}, {3,4}.
        let a = set(&[1, 2]);
        let b = set(&[3, 4]);
        let mut basis = BoolBasis::new();
        basis.insert(&a);
        basis.insert(&b);
        let va = basis.bss_vector(&a);
        assert!((va[0] - 1.0).abs() < 1e-12);
        assert!((va[1] - 0.0).abs() < 1e-12);
        // x = {1,3} is half of each basis element.
        let vx = basis.bss_vector(&set(&[1, 3]));
        assert!((vx[0] - 0.5).abs() < 1e-12);
        assert!((vx[1] - 0.5).abs() < 1e-12);
    }

    #[test]
    fn projection_splits_into_disjoint_retained_and_novelty() {
        let a = set(&[1, 2, 3]);
        let b = set(&[3, 4]);
        let mut basis = BoolBasis::new();
        basis.insert(&a);
        basis.insert(&b);
        // cover = {1,2,3,4}; x = {3,5} -> retained {3}, novelty {5}.
        let x = set(&[3, 5]);
        let p = basis.project(&x);
        assert_eq!(p.retained.popcount(), 1);
        assert_eq!(p.novelty.popcount(), 1);
        // retained and novelty are disjoint and reassemble x
        assert!(p.retained.intersection(&p.novelty).is_empty());
        let reassembled = p.retained.union(&p.novelty);
        assert!(reassembled.difference(&x).is_empty() && x.difference(&reassembled).is_empty());
        // novelty is always inside the linear residual of x
        let residual = basis.residual(&x);
        assert!(p.novelty.difference(&residual).is_empty());
    }

    #[test]
    fn change_of_basis_transports_coordinates() {
        // basis_old from {a,b}; basis_new adds an independent direction c.
        let a = set(&[1, 2, 3]);
        let b = set(&[2, 3, 4]);
        let c = set(&[9, 10]);
        let mut old = BoolBasis::new();
        old.insert(&a);
        old.insert(&b);
        let mut new = BoolBasis::new();
        for s in [&a, &b, &c] {
            new.insert(s);
        }
        let m = new.change_of_basis(&old);
        assert_eq!(m.len(), old.dimension());
        assert!(m.iter().all(|c| c.is_some()), "monotone growth keeps the old basis");

        // transport a coordinate vector and check the same set comes back
        let x = a.intersection(&b).union(&set(&[1])); // some set in the old span
        let old_coords = old.coordinates(&x).expect("x is in the old span");
        let mut new_coords = vec![false; new.dimension()];
        for (j, col) in m.iter().enumerate() {
            if old_coords[j] {
                let col = col.as_ref().unwrap();
                for (i, &bit) in col.iter().enumerate() {
                    new_coords[i] ^= bit;
                }
            }
        }
        assert!(same(&new.reconstruct(&new_coords), &x), "a·M transports x exactly");
        assert_eq!(new.coordinates(&x).unwrap(), new_coords);
    }

    #[test]
    fn linear_and_minimal_cover() {
        let a = set(&[1, 2, 3]);
        let b = set(&[2, 3, 4]);
        let c = set(&[9, 10]);
        let mut basis = BoolBasis::new();
        for s in [&a, &b, &c] {
            basis.insert(s);
        }

        // `a` is in span: the linear cover is the coordinate support and it
        // reconstructs `a` exactly.
        let lin = basis.linear_cover(&a).expect("a is in span");
        assert_eq!(lin.len(), 2, "a = B0 Δ B1 in this basis");
        let mut coords = vec![false; basis.dimension()];
        for &i in &lin {
            coords[i] = true;
        }
        assert!(same(&basis.reconstruct(&coords), &a));

        // The greedy union cover of `a` needs the same two elements.
        let cov = basis.minimal_cover(&a);
        assert_eq!(cov.len(), 2);
        let mut covered = HLLSet::new();
        for &i in &cov {
            covered = covered.union(&basis.basis[i]);
        }
        assert!(a.difference(&covered).is_empty());

        // A set with a bit outside the cover: greedy covers what it can, and
        // the linear cover does not exist.
        let outside = set(&[1, 99]);
        let cov = basis.minimal_cover(&outside);
        let mut covered = HLLSet::new();
        for &i in &cov {
            covered = covered.union(&basis.basis[i]);
        }
        assert_eq!(outside.difference(&covered).popcount(), 1, "bit 99 is uncoverable");
        assert!(basis.linear_cover(&outside).is_none());
    }
}
