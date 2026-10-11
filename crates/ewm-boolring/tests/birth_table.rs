//! Integration tests for the birth table — the flag basis stratified by
//! birth commit (`docs/EWM_BASIS_STRATA.md` §1–§4), ported from the laws of
//! `notebooks/27_basis_strata.ipynb`.

use ewm_boolring::{push_recorded, BirthTable, BoolWindow};
use hllset_core::HLLSet;

fn set(bits: &[u32]) -> HLLSet {
    let mut h = HLLSet::new();
    for b in bits {
        h.add_bit(*b);
    }
    h
}

fn births_up_to(table: &BirthTable, push: u64) -> usize {
    table.iter().filter(|r| r.push_index <= push).count()
}

#[test]
fn flag_basis_laws_on_small_sets() {
    // Three originals on toy bits: {1,3}, {3,5}, {1,5}.
    //   push 0: R0 = {1,3}, pivot 1
    //   push 1: R1 = {3,5}, pivot 3, rotation of slot 0 -> {1,5}
    //   push 2: in-span (no birth)
    let originals = [set(&[1, 3]), set(&[3, 5]), set(&[1, 5])];
    let mut window = BoolWindow::new(8);
    let mut births = BirthTable::new();
    for (i, o) in originals.iter().enumerate() {
        push_recorded(&mut window, &mut births, o, Some(&format!("t:{i}")));
    }

    let t = births.len();
    assert_eq!(t, 2, "two of three pushes add a direction");
    assert_eq!(window.dimension(), 2);

    // Distinct pivots in birth order.
    let pivots: Vec<u32> = births.iter().map(|r| r.pivot).collect();
    let mut sorted = pivots.clone();
    sorted.sort_unstable();
    sorted.dedup();
    assert_eq!(pivots.len(), sorted.len(), "flag pivots are distinct");

    // Each birth residual is a unit vector in the flag basis (R_i -> e_i).
    for i in 0..t {
        let coords = births.flag_coordinates(&births.stratum(i).unwrap().residual).unwrap();
        let expected: Vec<bool> = (0..t).map(|j| j == i).collect();
        assert_eq!(coords, expected, "R_{i} has coordinates e_{i}");
    }

    // Each birth residual lies outside the earlier span (R_i ∉ span_{i-1}).
    for i in 0..t {
        assert!(
            !births.membership(&births.stratum(i).unwrap().residual, i),
            "R_{i} is outside the prefix span"
        );
    }

    // Span chain: every original is spanned by the strata born up to its push.
    for (i, o) in originals.iter().enumerate() {
        assert!(
            births.membership(o, births_up_to(&births, i as u64)),
            "original {i} is in the span of the strata born up to its push"
        );
    }

    // cover(R) = union(originals).
    let union_originals = originals.iter().fold(HLLSet::new(), |acc, x| acc.union(x));
    assert_eq!(
        births.cover().content_key(),
        union_originals.content_key(),
        "cover of the flag basis equals the union of the originals"
    );

    // Rotations happened: the RREF slot differs from the birth residual.
    assert_ne!(
        window.basis().basis[0].content_key(),
        births.stratum(0).unwrap().residual.content_key(),
        "slot 0 was re-pivoted by the later birth"
    );

    // Determinism: replay produces the same pivots and residuals.
    let mut w2 = BoolWindow::new(8);
    let mut b2 = BirthTable::new();
    for (i, o) in originals.iter().enumerate() {
        push_recorded(&mut w2, &mut b2, o, Some(&format!("t:{i}")));
    }
    let p2: Vec<u32> = b2.iter().map(|r| r.pivot).collect();
    assert_eq!(p2, pivots, "replay gives the same pivots");
    for (a, b) in births.iter().zip(b2.iter()) {
        assert_eq!(a.residual.content_key(), b.residual.content_key());
    }

    // Provenance: each birth carries the commit of the push that caused it.
    for r in births.iter() {
        assert_eq!(
            r.commit,
            Some(format!("t:{}", r.push_index)),
            "the birth records its materializing commit"
        );
        assert!(!r.residual.is_empty());
    }
}

#[test]
fn birth_table_query_api_on_token_sets() {
    let originals = [
        HLLSet::from_tokens(["a", "b", "c"]),
        HLLSet::from_tokens(["b", "c", "d"]),
        HLLSet::from_tokens(["d", "e", "f"]),
    ];
    let mut window = BoolWindow::new(8);
    let mut births = BirthTable::new();
    for (i, o) in originals.iter().enumerate() {
        push_recorded(&mut window, &mut births, o, Some(&format!("t:{i}")));
    }

    // stratum -> commit and commit -> stratum (the two-way index).
    assert_eq!(births.stratum(0).unwrap().generation, 0);
    assert_eq!(births.stratum(0).unwrap().commit.as_deref(), Some("t:0"));
    assert_eq!(births.generation_of("t:0"), Some(0));
    for r in births.iter() {
        assert_eq!(r.commit, Some(format!("t:{}", r.push_index)));
    }

    // The hard signature reconstructs an original exactly.
    for o in &originals {
        let sig = births.signature(o).expect("originals are in the span");
        let recon = sig.iter().fold(HLLSet::new(), |acc, &j| {
            acc.symmetric_difference(&births.stratum(j).unwrap().residual)
        });
        assert_eq!(recon.content_key(), o.content_key(), "signature reconstructs the original");
    }

    // Membership against the prefix of strata born up to each push.
    for (i, o) in originals.iter().enumerate() {
        assert!(births.membership(o, births_up_to(&births, i as u64)));
    }

    // Soft signature is always defined and in [0, 1].
    for o in &originals {
        let soft = births.soft_signature(o);
        assert!(soft.iter().all(|w| (0.0..=1.0).contains(w)));
    }

    // A foreign set (a bit outside the cover) is outside the span and has an
    // all-zero soft profile.
    let cover = births.cover();
    let plane_bits = (hllset_core::M * hllset_core::BITS_PER_REG as usize) as u32;
    let foreign_bit = (0..plane_bits)
        .find(|b| !cover.has_bit(b / 32, b % 32))
        .expect("the plane has room for a foreign bit");
    let foreign = set(&[foreign_bit]);
    assert!(births.signature(&foreign).is_none(), "foreign set is outside the span");
    let soft = births.soft_signature(&foreign);
    assert!(soft.iter().all(|w| *w == 0.0), "foreign set has an all-zero soft profile");
}

#[test]
fn windowed_mode_records_no_births() {
    // The birth flag is a monotone-mode structure; the legacy windowed mode
    // rebuilds the basis on eviction, so `push_with_birth` returns None there.
    let mut window = BoolWindow::windowed(2);
    let mut births = BirthTable::new();
    for s in [set(&[1, 2]), set(&[2, 3]), set(&[3, 4])] {
        push_recorded(&mut window, &mut births, &s, None);
    }
    assert!(births.is_empty(), "windowed mode has no birth flag");
}
