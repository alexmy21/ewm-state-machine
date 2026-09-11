//! The `H(t)` commit view — the HLLSet reading of one commit, per channel.
//!
//! For a commit `t` with parents `P`, on channel `GX`:
//!
//! ```text
//! H(t)   = (S(t), H(t-1), D, R, N)
//! S(t)   = GX(t) — the channel state at commit t
//! H(t-1) = ∪ { GX(p) : p ∈ P }                 (the lattice join of parents)
//! D      = H(t-1) \ S(t)                        (departed bits)
//! R      = H(t-1) ∩ S(t)                        (retained bits)
//! N      = S(t) \ H(t-1)                        (new bits)
//! ```
//!
//! Invariants (verified in tests):
//!
//! ```text
//! D ∪ R = H(t-1),   R ∪ N = S(t),   D ∩ N = ∅
//! ```
//!
//! All three channels are commit-linked, so the same commit yields three
//! views — one per GX.

use hllset_core::HLLSet;

use crate::object::{Gx, ObjectId};
use crate::repo::Repository;
use crate::store::{ObjectStore, Result};

/// The HLLSet view of a commit on one channel.
#[derive(Clone, Debug)]
pub struct CommitView {
    /// The commit address (the timer).
    pub commit: ObjectId,
    /// The channel this view reads (`G1`, `G2`, `G3`).
    pub channel: Gx,
    /// `S(t)` — the channel state at this commit.
    pub state: HLLSet,
    /// `H(t-1)` — the join of the parents' same-channel states (empty for root).
    pub parent_state: HLLSet,
    /// `D` — departed bits.
    pub departed: HLLSet,
    /// `R` — retained bits.
    pub retained: HLLSet,
    /// `N` — new bits.
    pub new: HLLSet,
    /// Parent addresses.
    pub parents: Vec<ObjectId>,
}

/// Compute the `H(t)` view of a commit on a specific channel.
pub fn view_channel<S: ObjectStore>(
    repo: &Repository<S>,
    commit_id: &ObjectId,
    gx: Gx,
) -> Result<CommitView> {
    let commit = repo.read_commit(commit_id)?;
    let state = repo.state_channel(commit_id, gx)?;

    let mut parent_state = HLLSet::new();
    for parent in &commit.parents {
        parent_state.merge(&repo.state_channel(parent, gx)?);
    }

    let departed = parent_state.difference(&state);
    let retained = parent_state.intersection(&state);
    let new = state.difference(&parent_state);

    Ok(CommitView {
        commit: commit_id.clone(),
        channel: gx,
        state,
        parent_state,
        departed,
        retained,
        new,
        parents: commit.parents,
    })
}

/// The `G1` view (single-seed convenience).
pub fn view<S: ObjectStore>(repo: &Repository<S>, commit_id: &ObjectId) -> Result<CommitView> {
    view_channel(repo, commit_id, Gx::G1)
}

/// All three channel views of a commit.
pub fn views<S: ObjectStore>(
    repo: &Repository<S>,
    commit_id: &ObjectId,
) -> Result<[CommitView; 3]> {
    let mut out = Vec::with_capacity(3);
    for gx in Gx::ALL {
        out.push(view_channel(repo, commit_id, gx)?);
    }
    Ok(out.try_into().expect("three channels"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hllset_lut::BitTf;
    use crate::repo::LatticeState;
    use crate::store::MemoryStore;

    fn hll(tokens: &[&str]) -> HLLSet {
        HLLSet::from_tokens(tokens.iter())
    }

    #[test]
    fn commit_view_decomposition_invariants() {
        let mut repo = Repository::new(MemoryStore::default());
        let root = repo
            .commit(&LatticeState::single(&hll(&["a", "b"])), &[], "root")
            .unwrap();
        let tip = repo
            .commit(&LatticeState::single(&hll(&["b", "c", "d"])), &[root.clone()], "tip")
            .unwrap();

        let v = view(&repo, &tip).unwrap();
        assert_eq!(v.parents, vec![root]);
        assert_eq!(v.channel, Gx::G1);

        // D ∪ R = H(t-1)
        assert_eq!(
            v.departed.union(&v.retained).popcount(),
            v.parent_state.popcount()
        );
        // R ∪ N = S(t)
        assert_eq!(
            v.retained.union(&v.new).popcount(),
            v.state.popcount()
        );
        // D ∩ N = ∅
        assert_eq!(v.departed.intersection(&v.new).popcount(), 0);
        // D = {a}, R = {b}, N = {c, d}
        assert_eq!(v.departed.popcount(), hll(&["a"]).popcount());
        assert_eq!(v.retained.popcount(), hll(&["b"]).popcount());
        assert_eq!(v.new.popcount(), hll(&["c", "d"]).popcount());
    }

    #[test]
    fn root_has_empty_parent_state() {
        let mut repo = Repository::new(MemoryStore::default());
        let root = repo
            .commit(&LatticeState::single(&hll(&["a"])), &[], "root")
            .unwrap();
        let v = view(&repo, &root).unwrap();
        assert_eq!(v.parent_state.popcount(), 0);
        assert_eq!(v.new.popcount(), v.state.popcount());
    }

    #[test]
    fn three_channel_views_are_commit_linked() {
        let mut repo = Repository::new(MemoryStore::default());
        let state = LatticeState {
            g1: hll(&["one"]),
            g2: hll(&["two"]),
            g3: hll(&["three"]),
            tf: BitTf::new(),
        };
        let c = repo.commit(&state, &[], "channels").unwrap();
        let vs = views(&repo, &c).unwrap();
        assert_eq!(vs[0].state.popcount(), state.g1.popcount());
        assert_eq!(vs[1].state.popcount(), state.g2.popcount());
        assert_eq!(vs[2].state.popcount(), state.g3.popcount());
        assert_eq!(vs[0].commit, c);
        assert_eq!(vs[1].commit, c);
        assert_eq!(vs[2].commit, c);
    }
}
