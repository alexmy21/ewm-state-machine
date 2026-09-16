//! Integration tests for the operational graph: expression identity,
//! fan-out by reference, the dispatcher's determinism, and the agreement
//! between the lattice side and the processor side.

use ewm_ops::{Dispatcher, Edge, OpGraph, Port, Target};
use hllset_core::HLLSet;

fn set(bits: &[u32]) -> HLLSet {
    let mut h = HLLSet::new();
    for b in bits {
        h.add_bit(*b);
    }
    h
}

#[test]
fn expression_identity_is_the_source_sha1() {
    let a = OpGraph::new().add_op("2 1 union").unwrap();
    let b = OpGraph::new().add_op("2 1 union").unwrap();
    let c = OpGraph::new().add_op("2 1 inter").unwrap();
    assert_eq!(a, b, "same source → same program CID");
    assert_ne!(a, c, "different source → different program CID");
    assert!(a.starts_with("p:"), "program CIDs are p:<sha1>");
}

#[test]
fn expression_runs_on_the_lattice_ops() {
    let mut g = OpGraph::new();
    let x = g.add_value(set(&[1, 2, 3]));
    let y = g.add_value(set(&[2, 3, 4]));
    let union = g.add_op("2 1 union").unwrap();
    let spec = g.ops.get(&union).unwrap().clone();

    let out = spec.expr.run(&[x, y], &g.values, &g.ops).unwrap();
    assert_eq!(
        out[0].content_key(),
        set(&[1, 2, 3]).union(&set(&[2, 3, 4])).content_key(),
        "the op result is the lattice union, addressed identically"
    );
}

#[test]
fn call_inlines_another_program() {
    let mut g = OpGraph::new();
    let x = g.add_value(set(&[1, 2]));
    let y = g.add_value(set(&[2, 3]));

    let callee = g.add_op("2 1 union").unwrap();
    let caller_src = format!("2 1 call:{callee}");
    let caller = g.add_op(&caller_src).unwrap();

    let out = g.ops.get(&caller).unwrap().expr
        .run(&[x, y], &g.values, &g.ops)
        .unwrap();
    assert_eq!(out[0].content_key(), set(&[1, 2]).union(&set(&[2, 3])).content_key());
}

#[test]
fn value_store_is_idempotent() {
    let mut g = OpGraph::new();
    let a = g.add_value(set(&[1, 2]));
    let b = g.add_value(set(&[1, 2]));
    assert_eq!(a, b, "same content → same CID, added once");
    assert_eq!(g.values.len(), 1);
}

#[test]
fn dispatcher_fans_out_by_reference_and_agrees_with_the_lattice() {
    let mut g = OpGraph::new();
    let x = g.add_value(set(&[1, 2]));
    let c = g.add_value(set(&[9]));

    // Two consumers of the same value: A = x ∪ c, B = x ∩ c.
    let a = g.add_op(&format!("1 1 @{c} union")).unwrap();
    let b = g.add_op(&format!("1 1 @{c} inter")).unwrap();
    let (from, to) = ewm_ops::dispatch::value_edge(&x, &a, 0);
    g.connect(from, to);
    let (from, to) = ewm_ops::dispatch::value_edge(&x, &b, 0);
    g.connect(from, to);

    // The expected CIDs, computed on the lattice side *before* the
    // dispatcher borrows the graph.
    let join_cid = g.join(&x, &c).unwrap();
    let meet_cid = g.meet(&x, &c).unwrap();

    let mut d = Dispatcher::new(&mut g);
    d.seed_value(&x);
    let log = d.run().unwrap().clone();
    drop(d);

    assert_eq!(log.records.len(), 2, "both consumers fired");
    let fired: std::collections::BTreeSet<_> = log.records.iter().map(|r| r.op.clone()).collect();
    assert!(fired.contains(&a) && fired.contains(&b));

    for r in &log.records {
        assert_eq!(r.inputs, vec![x.clone()], "fan-out is the same reference");
        if r.op == a {
            assert_eq!(r.outputs[0], join_cid, "op union == lattice join");
        } else {
            assert_eq!(r.outputs[0], meet_cid, "op inter == lattice meet");
        }
    }

    // The lattice order is readable through the same API.
    assert!(g.subset(&x, &join_cid).unwrap());
}

#[test]
fn dispatcher_traverses_a_feedback_cycle_deterministically() {
    // A self-loop: dup consumes x and produces [x, x]; out-port 0 feeds back
    // into in-port 0. The dispatcher keeps firing; time = fire count.
    let build = || {
        let mut g = OpGraph::new();
        let x = g.add_value(set(&[1, 2]));
        let dup = g.add_op("1 2 dup").unwrap();
        let (from, to) = ewm_ops::dispatch::value_edge(&x, &dup, 0);
        g.connect(from, to);
        let (from, to) = ewm_ops::dispatch::op_edge(&dup, 0, &dup, 0);
        g.connect(from, to);
        (g, x, dup)
    };

    let (mut g1, x1, dup1) = build();
    let mut d1 = Dispatcher::new(&mut g1);
    d1.seed_value(&x1);
    let log1 = d1.run_limited(5).unwrap().clone();

    let (mut g2, x2, _dup2) = build();
    let mut d2 = Dispatcher::new(&mut g2);
    d2.seed_value(&x2);
    let log2 = d2.run_limited(5).unwrap().clone();

    assert_eq!(log1, log2, "the fire sequence is deterministic");
    assert_eq!(log1.records.len(), 5);
    for r in &log1.records {
        assert_eq!(r.op, dup1);
        assert_eq!(r.inputs, vec![x1.clone()]);
        assert_eq!(r.outputs, vec![x1.clone(), x1.clone()]);
    }
}

#[test]
fn graph_ports_are_typed_and_sorted() {
    let mut g = OpGraph::new();
    let x = g.add_value(set(&[1]));
    let a = g.add_op("1 1 dup").unwrap();
    let b = g.add_op("1 2 dup").unwrap();
    let e1 = Edge {
        from: Port::Value(x.clone()),
        to: Target::OpIn { op: b.clone(), index: 0 },
    };
    let e2 = Edge {
        from: Port::Value(x.clone()),
        to: Target::OpIn { op: a.clone(), index: 0 },
    };
    g.edges.push(e1);
    g.edges.push(e2);

    // Consumers come back sorted and deduped — the deterministic routing
    // order the dispatcher relies on.
    assert_eq!(
        g.consumers_of(&Port::Value(x)),
        vec![(a, 0), (b, 0)]
    );
}
