//! Integration tests: the full 2005-Git surface over memory and loose stores.

use ewm_git::{view, LatticeState, Gx, LooseStore, MemoryStore, ObjectStore, Repository};
use hllset_core::HLLSet;

fn hll(tokens: &[&str]) -> HLLSet {
    HLLSet::from_tokens(tokens.iter())
}

fn gs(tokens: &[&str]) -> LatticeState {
    LatticeState::single(&hll(tokens))
}

#[test]
fn full_lifecycle_over_memory_store() {
    let mut repo = Repository::new(MemoryStore::default());

    // Linear history: S0 ⊂ S1 ⊂ S2 … (monotone union)
    let c0 = repo.commit(&gs(&["a"]), &[], "start").unwrap();
    let c1 = repo.commit(&gs(&["a", "b"]), &[c0.clone()], "add b").unwrap();
    let c2 = repo.commit(&gs(&["a", "b", "c"]), &[c1.clone()], "add c").unwrap();

    let v2 = view(&repo, &c2).unwrap();
    assert_eq!(v2.departed.popcount(), 0, "monotone: nothing departs");
    assert_eq!(v2.retained.popcount(), hll(&["a", "b"]).popcount());
    assert_eq!(v2.new.popcount(), hll(&["c"]).popcount());

    // Branch + merge: the join is a derivation (no commit); record it
    // explicitly when topology matters.
    let side = repo.commit(&gs(&["a", "d"]), &[c0.clone()], "side").unwrap();
    let joined = repo.merge(&c2, &side).unwrap();
    let merged = repo.merge_commit(&c2, &side, "merge").unwrap();
    assert_eq!(joined.g1.popcount(), repo.state(&merged).unwrap().popcount());
    let mv = view(&repo, &merged).unwrap();
    assert_eq!(
        mv.state.popcount(),
        hll(&["a", "b", "c"]).union(&hll(&["a", "d"])).popcount()
    );
    assert_eq!(mv.parents, vec![c2, side]);

    // The H(t) view of the merge: D ∪ R = H(t-1), R ∪ N = S(t).
    assert_eq!(
        mv.departed.union(&mv.retained).popcount(),
        mv.parent_state.popcount()
    );
    assert_eq!(
        mv.retained.union(&mv.new).popcount(),
        mv.state.popcount()
    );
}

#[test]
fn gc_keeps_history_and_prunes_orphans() {
    let mut repo = Repository::new(MemoryStore::default());
    let a = repo.commit(&gs(&["keep"]), &[], "root").unwrap();
    let orphan = repo.commit(&gs(&["gone"]), &[], "orphan").unwrap();
    let b = repo.commit(&gs(&["keep", "more"]), &[a.clone()], "tip").unwrap();

    let pruned = repo.gc(&[]).unwrap().pruned;
    assert!(pruned >= 1, "the orphan's objects must be pruned");
    assert!(repo.store().contains(&a));
    assert!(repo.store().contains(&b));
    assert!(!repo.store().contains(&orphan));
    assert_eq!(repo.log().unwrap().len(), 2);
}

#[test]
fn loose_store_full_lifecycle() {
    let dir = std::env::temp_dir().join(format!("ewm-git-it-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    {
        let mut repo = Repository::new(LooseStore::new(&dir));
        let a = repo.commit(&gs(&["x", "y"]), &[], "a").unwrap();
        let b = repo.commit(&gs(&["y", "z"]), &[a.clone()], "b").unwrap();
        assert_eq!(repo.state(&b).unwrap().popcount(), hll(&["y", "z"]).popcount());
        let v = view(&repo, &b).unwrap();
        assert_eq!(v.departed.popcount(), hll(&["x"]).popcount());
    }
    // Reopen the store: objects and HEAD are persisted.
    {
        let mut repo = Repository::open(LooseStore::new(&dir));
        assert_eq!(repo.log().unwrap().len(), 2);
        let pruned = repo.gc(&[]).unwrap().pruned;
        assert_eq!(pruned, 0, "already minimal after reopen");
    }
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn gc_to_archives_pruned_branches() {
    let mut repo = Repository::new(MemoryStore::default());
    let a = repo.commit(&gs(&["keep"]), &[], "root").unwrap();
    let orphan = repo.commit(&gs(&["pruned", "branch"]), &[], "orphan").unwrap();
    let b = repo.commit(&gs(&["keep", "more"]), &[a.clone()], "tip").unwrap();

    // Archive-before-prune: the orphan's objects move to the archive.
    let mut archive = MemoryStore::default();
    let report = repo.gc_to(&mut archive, &[]).unwrap();
    assert_eq!(report.pruned, report.archived);
    assert!(report.archived >= 1, "orphan objects must be archived");

    // Working store: pruned. Archive: the branch is still addressable.
    assert!(!repo.store().contains(&orphan));
    assert!(repo.store().contains(&b));
    assert!(archive.contains(&orphan), "pruned branch must live on in the archive");

    // And its G1 state is still resolvable from the archive.
    let commit = match archive.get(&orphan).unwrap() {
        ewm_git::Object::Commit(c) => c,
        _ => panic!("expected commit"),
    };
    let state_blob = archive.get(&commit.trees[0]).unwrap();
    match state_blob {
        ewm_git::Object::Blob(bytes) => {
            let restored = HLLSet::from_bytes(&bytes).unwrap();
            assert_eq!(restored.popcount(), hll(&["pruned", "branch"]).popcount());
        }
        _ => panic!("expected state blob"),
    }
}

#[test]
fn time_travel_project_any_hllset_onto_any_commit() {
    let mut repo = Repository::new(MemoryStore::default());
    let c0 = repo.commit(&gs(&["a", "b"]), &[], "t0").unwrap();
    let c1 = repo.commit(&gs(&["a", "b", "c"]), &[c0.clone()], "t1").unwrap();

    let query = hll(&["a", "c", "z"]);
    // At t0 only "a" existed; at t1 "a" and "c" existed.
    assert_eq!(repo.project(&query, &c0, Gx::G1).unwrap().popcount(), hll(&["a"]).popcount());
    assert_eq!(
        repo.project(&query, &c1, Gx::G1).unwrap().popcount(),
        hll(&["a", "c"]).popcount()
    );
    // All three channels project consistently (single-seed in this test).
    let all = repo.project_all(&query, &c0).unwrap();
    assert_eq!(all[0].popcount(), all[1].popcount());
    assert_eq!(all[1].popcount(), all[2].popcount());
}

#[test]
fn content_addressing_is_iica() {
    let mut r1 = Repository::new(MemoryStore::default());
    let mut r2 = Repository::new(MemoryStore::default());
    let a1 = r1.commit(&gs(&["same"]), &[], "m").unwrap();
    let a2 = r2.commit(&gs(&["same"]), &[], "m").unwrap();
    assert_eq!(a1, a2);
    let b1 = r1.commit(&gs(&["same", "next"]), &[a1.clone()], "m2").unwrap();
    let b2 = r2.commit(&gs(&["same", "next"]), &[a2.clone()], "m2").unwrap();
    assert_eq!(b1, b2, "identical DAGs must be bit-for-bit identical");
}
