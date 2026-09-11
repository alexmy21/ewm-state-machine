//! `context-tree` — a Merkle tree over the working set of HLLSets
//! (LUT_VIEW.md §7, prototype notebook 17).
//!
//! - **Leaves** = HLLSet content keys (`h:<sha1>`) plus their per-LUT view
//!   keys (`(lut, v:<sha1>)` pairs) — views hang off their HLLSet.
//! - **Root** = the exact, content-addressed identity of the collection;
//!   it changes iff the leaf set (or its views) changes.
//! - **Operations are persistent and reversible** (every op returns a new
//!   tree; old roots stay valid) and form a **lattice**: `merge` (union),
//!   `intersection`, `difference`, with `diff` giving leaf-level D/R/N.
//!
//! This is the structure the working memory is organized as: the algebraic
//! `S(t)` in the Noether equation `H(t) = (S(t), H(t-1), D, R, N)`.

#![forbid(unsafe_code)]

use sha1::{Digest, Sha1};

/// SHA-1 over arbitrary bytes, hex-encoded.
pub fn sha1_hex(data: &[u8]) -> String {
    hex::encode(Sha1::digest(data))
}

/// A tree leaf: one HLLSet (`h`) and the views materialized from it, as
/// `(lut, view_key)` pairs.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Leaf {
    pub h: String,
    pub views: Vec<(String, String)>,
}

/// Leaf-level difference of two trees.
#[derive(Clone, PartialEq, Eq, Debug, Default)]
pub struct TreeDiff {
    pub added: Vec<String>,
    pub removed: Vec<String>,
    pub retained: Vec<String>,
}

fn leaf_hash(leaf: &Leaf) -> String {
    let mut data = vec![0x00u8];
    data.extend_from_slice(leaf.h.as_bytes());
    for (l, v) in &leaf.views {
        data.push(0u8);
        data.extend_from_slice(l.as_bytes());
        data.extend_from_slice(v.as_bytes());
    }
    sha1_hex(&data)
}

fn internal_hash(left: &str, right: &str) -> String {
    let mut data = vec![0x01u8];
    data.extend_from_slice(left.as_bytes());
    data.extend_from_slice(right.as_bytes());
    sha1_hex(&data)
}

/// A Merkle tree over the HLLSet working set.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct ContextTree {
    root: String,
    leaves: Vec<Leaf>,
    /// All Merkle levels; `levels.last()` is the single root digest.
    levels: Vec<Vec<String>>,
}

impl ContextTree {
    /// Build a canonical tree: each leaf's views are sorted, leaves are
    /// sorted by `h` and deduplicated by `h`. Equal leaf sets always produce
    /// the same root, regardless of insertion order.
    pub fn build(leaves: Vec<Leaf>) -> Self {
        let mut leaves = leaves;
        for leaf in &mut leaves {
            leaf.views.sort();
        }
        leaves.sort_by(|a, b| a.h.cmp(&b.h));
        leaves.dedup_by(|a, b| a.h == b.h);

        if leaves.is_empty() {
            let root = sha1_hex(b"empty-context");
            return Self {
                root: root.clone(),
                leaves,
                levels: vec![vec![root]],
            };
        }

        let mut levels: Vec<Vec<String>> = Vec::new();
        let mut level: Vec<String> = leaves.iter().map(leaf_hash).collect();
        levels.push(level.clone());
        while level.len() > 1 {
            let mut next: Vec<String> = Vec::new();
            for pair in level.chunks(2) {
                let right = pair.get(1).cloned().unwrap_or_else(|| pair[0].clone());
                next.push(internal_hash(&pair[0], &right));
            }
            levels.push(next.clone());
            level = next;
        }
        let root = level[0].clone();
        Self {
            root,
            leaves,
            levels,
        }
    }

    pub fn empty() -> Self {
        Self::build(Vec::new())
    }

    pub fn root(&self) -> &str {
        &self.root
    }

    pub fn leaves(&self) -> &[Leaf] {
        &self.leaves
    }

    /// The Merkle levels, leaf level first, root last.
    pub fn levels(&self) -> &[Vec<String>] {
        &self.levels
    }

    pub fn contains(&self, h: &str) -> bool {
        self.leaves
            .binary_search_by(|leaf| leaf.h.as_str().cmp(h))
            .is_ok()
    }

    /// Persistent insert: a new tree with `leaf` added (existing `h`
    /// replaced). The old tree remains valid.
    pub fn insert(&self, leaf: Leaf) -> Self {
        let mut leaves = self.leaves.clone();
        leaves.retain(|l| l.h != leaf.h);
        leaves.push(leaf);
        Self::build(leaves)
    }

    /// Persistent remove: a new tree without `h`. Removing an absent key
    /// returns an equal tree (idempotent).
    pub fn remove(&self, h: &str) -> Self {
        let leaves: Vec<Leaf> = self
            .leaves
            .iter()
            .filter(|leaf| leaf.h != h)
            .cloned()
            .collect();
        Self::build(leaves)
    }

    /// Lattice join: leaf-set union.
    pub fn merge(&self, other: &Self) -> Self {
        let mut leaves = self.leaves.clone();
        leaves.extend(other.leaves.clone());
        Self::build(leaves)
    }

    /// Lattice meet: leaves present in both trees.
    pub fn intersection(&self, other: &Self) -> Self {
        let leaves: Vec<Leaf> = self
            .leaves
            .iter()
            .filter(|leaf| other.contains(&leaf.h))
            .cloned()
            .collect();
        Self::build(leaves)
    }

    /// Lattice difference: leaves in `self` but not in `other`.
    pub fn difference(&self, other: &Self) -> Self {
        let leaves: Vec<Leaf> = self
            .leaves
            .iter()
            .filter(|leaf| !other.contains(&leaf.h))
            .cloned()
            .collect();
        Self::build(leaves)
    }

    /// Leaf-level D/R/N between `self` (previous) and `other` (current):
    /// `removed` = leaves that left, `added` = leaves that entered,
    /// `retained` = leaves present in both.
    pub fn diff(&self, other: &Self) -> TreeDiff {
        TreeDiff {
            added: other
                .leaves
                .iter()
                .filter(|leaf| !self.contains(&leaf.h))
                .map(|leaf| leaf.h.clone())
                .collect(),
            removed: self
                .leaves
                .iter()
                .filter(|leaf| !other.contains(&leaf.h))
                .map(|leaf| leaf.h.clone())
                .collect(),
            retained: self
                .leaves
                .iter()
                .filter(|leaf| other.contains(&leaf.h))
                .map(|leaf| leaf.h.clone())
                .collect(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn leaf(h: &str) -> Leaf {
        Leaf {
            h: h.to_string(),
            views: Vec::new(),
        }
    }

    fn tree(hs: &[&str]) -> ContextTree {
        ContextTree::build(hs.iter().map(|h| leaf(h)).collect())
    }

    #[test]
    fn build_is_canonical_and_deduplicates() {
        let a = ContextTree::build(vec![leaf("h:b"), leaf("h:a"), leaf("h:b")]);
        let b = ContextTree::build(vec![leaf("h:a"), leaf("h:b")]);
        assert_eq!(a, b, "same leaf set ⇒ same root");
        assert_eq!(a.leaves().len(), 2, "duplicate h deduplicated");
        assert_eq!(a.root(), b.root());
    }

    #[test]
    fn insert_and_remove_are_reversible() {
        let base = tree(&["h:a", "h:b"]);
        let with_c = base.insert(leaf("h:c"));
        assert!(with_c.contains("h:c"));
        assert_eq!(with_c.remove("h:c"), base, "insert∘remove = id");

        let removed = base.remove("h:b");
        assert!(!removed.contains("h:b"));
        assert_eq!(removed.insert(leaf("h:b")), base, "remove∘insert = id");
    }

    #[test]
    fn merge_is_a_lattice_join() {
        let a = tree(&["h:a", "h:b"]);
        let b = tree(&["h:b", "h:c"]);
        let c = tree(&["h:c", "h:d"]);

        // Idempotent / commutative / associative on canonical roots.
        assert_eq!(a.merge(&a), a, "idempotent");
        assert_eq!(a.merge(&b), b.merge(&a), "commutative");
        assert_eq!(
            a.merge(&b).merge(&c),
            a.merge(&b.merge(&c)),
            "associative"
        );
        assert_eq!(a.merge(&b).leaves().len(), 3, "union");
    }

    #[test]
    fn meet_join_and_difference_form_a_lattice() {
        let a = tree(&["h:a", "h:b"]);
        let b = tree(&["h:b", "h:c"]);

        assert_eq!(a.intersection(&b).leaves().len(), 1, "meet");
        assert_eq!(a.difference(&b).leaves().len(), 1, "A \\ B");
        assert_eq!(b.difference(&a).leaves().len(), 1, "B \\ A");
        // A = (A ∩ B) ∪ (A \ B)
        assert_eq!(
            a.intersection(&b).merge(&a.difference(&b)),
            a,
            "meet ∪ difference = self"
        );
    }

    #[test]
    fn diff_reports_leaf_level_drn() {
        let prev = tree(&["h:a", "h:b"]);
        let now = tree(&["h:a", "h:c"]);
        let diff = prev.diff(&now);

        assert_eq!(diff.added, vec!["h:c".to_string()]);
        assert_eq!(diff.removed, vec!["h:b".to_string()]);
        assert_eq!(diff.retained, vec!["h:a".to_string()]);
        // D ∩ N = ∅ by construction.
        assert!(diff.added.iter().all(|h| !diff.removed.contains(h)));
    }

    #[test]
    fn views_are_hashed_into_the_root() {
        let mut with_view = leaf("h:a");
        with_view.views = vec![("main".to_string(), "v:aaaa".to_string())];
        let plain = tree(&["h:a"]);
        let viewed = ContextTree::build(vec![with_view]);
        assert_ne!(plain.root(), viewed.root(), "views change the root");
    }

    #[test]
    fn empty_tree_has_a_stable_root() {
        assert_eq!(ContextTree::empty(), ContextTree::build(Vec::new()));
        assert_eq!(ContextTree::empty().root(), sha1_hex(b"empty-context"));
    }
}
