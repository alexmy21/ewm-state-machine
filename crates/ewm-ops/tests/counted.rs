//! Tests for the counted fragment: `exists_c` / `forall_c` words and the
//! `universe` / `@universe:<name>` registry.
//!
//! The semantics are pinned to `docs/FOL_HLLSET.md` §5: `∃_c^U X` is the
//! union of the full-plane fibres of the coordinate that meet `X`, restricted
//! to the universe `U`; `∀_c^U X` is the union of the fibres wholly contained
//! in `X`, restricted to `U`.

use ewm_ops::dispatch::value_edge;
use ewm_ops::{compile_boot, Dispatcher, EvalError, OpGraph};
use hllset_core::HLLSet;

fn set(bits: &[u32]) -> HLLSet {
    let mut h = HLLSet::new();
    for b in bits {
        h.add_bit(*b);
    }
    h
}

fn bits(set: &HLLSet) -> Vec<u32> {
    set.bitmap().iter().collect()
}

/// Register `source` (if needed), then run it on `[x, u]` and return the
/// single output. Input order: the expression seeds the stack with
/// `[inputs[0], inputs[1]]`, so for the `( x universe -- result )` effect the
/// inputs are `[x, u]`.
fn run2(graph: &mut OpGraph, source: &str, x: &str, u: &str) -> HLLSet {
    graph.add_op(source).unwrap();
    let spec = graph.ops.get(&OpGraph::op_cid_for(source)).unwrap().clone();
    let out = spec
        .expr
        .run(&[x.to_string(), u.to_string()], &graph.values, &graph.ops)
        .unwrap();
    out.into_iter().next().unwrap()
}

#[test]
fn quantifiers_match_hand_computed_fibres() {
    let mut g = OpGraph::new();
    // Universe U = {0,1,2} ∪ {32,33}: register 0 has tz 0,1,2 and register 1
    // has tz 0,1. A fibre of `tz` is one column {0,32}, {1,33}, {2}; a fibre
    // of `reg` is one register {0,1,2} or {32,33}.
    let u = g.add_value(set(&[0, 1, 2, 32, 33]));
    let x = g.add_value(set(&[0, 33]));

    // ∃_tz X: columns of the tz values present in X∩U (tz 0 and 1).
    let ex = run2(&mut g, "2 1 exists_c:tz", &x, &u);
    assert_eq!(bits(&ex), vec![0, 1, 32, 33]);
    // ∀_tz X: no column is wholly inside X.
    let fa = run2(&mut g, "2 1 forall_c:tz", &x, &u);
    assert!(fa.is_empty());
    // ∃_reg X: both registers are touched, so the whole U.
    let exr = run2(&mut g, "2 1 exists_c:reg", &x, &u);
    assert_eq!(bits(&exr), bits(&g.values.get(&u).unwrap().clone()));
    // ∀_reg X: neither register is wholly inside X.
    let far = run2(&mut g, "2 1 forall_c:reg", &x, &u);
    assert!(far.is_empty());

    // X = {0,1,2} is exactly register 0 of U.
    let x2 = g.add_value(set(&[0, 1, 2]));
    let ex2 = run2(&mut g, "2 1 exists_c:reg", &x2, &u);
    assert_eq!(bits(&ex2), vec![0, 1, 2]);
    let fa2 = run2(&mut g, "2 1 forall_c:reg", &x2, &u);
    assert_eq!(bits(&fa2), vec![0, 1, 2]);
    // ∀_tz X2: only the singleton column {2} is contained.
    let fat2 = run2(&mut g, "2 1 forall_c:tz", &x2, &u);
    assert_eq!(bits(&fat2), vec![2]);
    // ∃_tz X2: all three columns are touched.
    let ext2 = run2(&mut g, "2 1 exists_c:tz", &x2, &u);
    assert_eq!(bits(&ext2), vec![0, 1, 2, 32, 33]);

    // Quantifying the whole universe gives the whole universe back.
    let exu = run2(&mut g, "2 1 exists_c:tz", &u, &u);
    assert_eq!(bits(&exu), bits(&g.values.get(&u).unwrap().clone()));
    let fau = run2(&mut g, "2 1 forall_c:tz", &u, &u);
    assert_eq!(bits(&fau), bits(&g.values.get(&u).unwrap().clone()));
}

#[test]
fn quantifier_laws_hold_relative_to_a_universe() {
    let mut g = OpGraph::new();
    let u_set = HLLSet::from_tokens(["u0", "u1", "u2", "u3", "u4", "u5", "u6", "u7"]);
    let x_set = HLLSet::from_tokens(["u0", "u2", "u4", "x9"]);
    let y_set = HLLSet::from_tokens(["u1", "u2", "u3", "y9"]);
    let u = g.add_value(u_set);
    let x = g.add_value(x_set.intersection(g.values.get(&u).unwrap()));
    let y = g.add_value(y_set.intersection(g.values.get(&u).unwrap()));

    for coord in ["tz", "reg"] {
        let exists = format!("2 1 exists_c:{coord}");
        let forall = format!("2 1 forall_c:{coord}");

        let ex = run2(&mut g, &exists, &x, &u);
        let fa = run2(&mut g, &forall, &x, &u);

        // Extensive / contractive.
        let xv = g.values.get(&x).unwrap().clone();
        assert!(xv.difference(&ex).is_empty(), "∃X ⊇ X ({coord})");
        assert!(fa.difference(&xv).is_empty(), "∀X ⊆ X ({coord})");

        // Idempotent.
        let ex_cid = g.add_value(ex.clone());
        let fa_cid = g.add_value(fa.clone());
        let ex2 = run2(&mut g, &exists, &ex_cid, &u);
        let fa2 = run2(&mut g, &forall, &fa_cid, &u);
        assert_eq!(ex2.content_key(), ex.content_key(), "∃∃X = ∃X ({coord})");
        assert_eq!(fa2.content_key(), fa.content_key(), "∀∀X = ∀X ({coord})");

        // De Morgan: U \ ∃X = ∀(U \ X) and U \ ∀X = ∃(U \ X).
        let uv = g.values.get(&u).unwrap().clone();
        let not_x = uv.difference(&xv);
        let not_x_cid = g.add_value(not_x.clone());
        let fa_not = run2(&mut g, &forall, &not_x_cid, &u);
        assert_eq!(
            uv.difference(&ex).content_key(),
            fa_not.content_key(),
            "U \\ ∃X = ∀(U \\ X) ({coord})"
        );
        let ex_not = run2(&mut g, &exists, &not_x_cid, &u);
        assert_eq!(
            uv.difference(&fa).content_key(),
            ex_not.content_key(),
            "U \\ ∀X = ∃(U \\ X) ({coord})"
        );

        // Distributivity: ∃(X ∪ Y) = ∃X ∪ ∃Y and ∀(X ∩ Y) = ∀X ∩ ∀Y.
        let yv = g.values.get(&y).unwrap().clone();
        let xy_union = g.add_value(xv.union(&yv));
        let xy_inter = g.add_value(xv.intersection(&yv));
        let exy = run2(&mut g, &exists, &xy_union, &u);
        let ey = run2(&mut g, &exists, &y, &u);
        assert_eq!(
            exy.content_key(),
            ex.union(&ey).content_key(),
            "∃(X ∪ Y) = ∃X ∪ ∃Y ({coord})"
        );
        let fxy = run2(&mut g, &forall, &xy_inter, &u);
        let fy = run2(&mut g, &forall, &y, &u);
        assert_eq!(
            fxy.content_key(),
            fa.intersection(&fy).content_key(),
            "∀(X ∩ Y) = ∀X ∩ ∀Y ({coord})"
        );

        // Monotone: X ⊆ Y ⟹ ∃X ⊆ ∃Y. Build Y' = X ∪ Y.
        let yp = g.add_value(xv.union(&yv));
        let exp = run2(&mut g, &exists, &yp, &u);
        assert!(ex.difference(&exp).is_empty(), "∃ monotone ({coord})");
    }
}

#[test]
fn universe_directive_and_references_compile() {
    let script = "\
value a apple banana
value b cherry date
universe L @a @b
def ex ( 1 -- 1 ) @universe:L exists_c:tz
def ex_bare ( 1 -- 1 ) @universe exists_c:tz
def neg ( 1 -- 1 ) @universe:L swap diff
";
    let mut program = compile_boot(script).unwrap();

    // The universe is stored as a first-class value: the lattice join of its
    // references, and appears in the vocabulary value rows (so the v: CID
    // pins the universe registry).
    let a = program.vocab.values["a"].clone();
    let b = program.vocab.values["b"].clone();
    let l = program.graph.join(&a, &b).unwrap();
    assert_eq!(program.vocab.values["L"], l);
    assert_eq!(program.vocab.ops.len(), 3);

    // `@universe` and `@universe:L` resolve to the same universe value, so
    // the two programs are the same canonical source — same program CID.
    assert_eq!(program.vocab.ops["ex"], program.vocab.ops["ex_bare"]);

    // The resolved source pins the universe CID, not the name.
    let spec = program.graph.ops.get(&program.vocab.ops["ex"]).unwrap();
    assert!(spec
        .expr
        .source
        .starts_with(&format!("1 1 @{l} exists_c:tz")));
    let neg = program.graph.ops.get(&program.vocab.ops["neg"]).unwrap();
    assert_eq!(neg.expr.source, format!("1 1 @{l} swap diff"));
}

#[test]
fn universe_reference_errors_are_compile_time() {
    // Bare @universe with no declared universe.
    let err = compile_boot("value a apple\ndef f ( 1 -- 1 ) @universe\n").unwrap_err();
    assert!(matches!(err, EvalError::Parse(_)));

    // Bare @universe with two declared universes is ambiguous.
    let err = compile_boot(
        "value a apple\nvalue b banana\nuniverse L1 @a\nuniverse L2 @b\ndef f ( 1 -- 1 ) @universe\n",
    )
    .unwrap_err();
    assert!(matches!(err, EvalError::Parse(_)));

    // Unknown universe name.
    let err = compile_boot("value a apple\ndef f ( 1 -- 1 ) @universe:nope\n").unwrap_err();
    assert_eq!(err, EvalError::UnknownUniverse("nope".into()));

    // Unknown coordinate is a compile error, not a silent fallback.
    let err =
        compile_boot("value a apple\ndef f ( 1 -- 1 ) exists_c:tz exists_c:foo\n").unwrap_err();
    assert!(matches!(err, EvalError::Parse(_)));
}

#[test]
fn dispatcher_fires_quantifier_ops_and_commits_the_result() {
    let mut g = OpGraph::new();
    let u = g.add_value(set(&[0, 1, 2, 32, 33]));
    let x = g.add_value(set(&[0, 33]));
    let op = g.add_op("2 1 exists_c:tz").unwrap();

    // Port 0 = x (deeper), port 1 = universe (top).
    let (from, to) = value_edge(&x, &op, 0);
    g.connect(from, to);
    let (from, to) = value_edge(&u, &op, 1);
    g.connect(from, to);

    let expected_cid = set(&[0, 1, 32, 33]).content_key();

    let mut d = Dispatcher::new(&mut g);
    d.seed_values(&[x.clone(), u.clone()]);
    let log = d.run().unwrap().clone();

    assert_eq!(log.records.len(), 1, "the quantifier op fires once");
    assert_eq!(log.records[0].inputs, vec![x, u]);
    assert_eq!(log.records[0].outputs, vec![expected_cid]);
}
