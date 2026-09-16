//! The two sides of the operational graph: values (the lattice) and
//! programs (the processors), plus the directed edges between them.
//!
//! The sides are physically separated collections; edges reference both by
//! CID. Nothing is copied — identity is the content address.

use std::collections::BTreeMap;

use hllset_core::HLLSet;
use serde::{Deserialize, Serialize};

use crate::expr::{op_cid, EvalError, Expression};

/// A value node id: `h:<sha1>` (the HLLSet content key).
pub type ValueCid = String;

/// A program node id: `p:<sha1 of the expression source>`.
pub type OpCid = String;

/// The lattice side: HLLSets addressed by content key.
///
/// Insertion is idempotent — inserting the same set twice returns the same
/// CID and adds nothing (the "discovered, not created" property).
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct ValueStore {
    values: BTreeMap<ValueCid, HLLSet>,
}

impl ValueStore {
    pub fn new() -> Self {
        Self::default()
    }

    /// Store a value and return its content key. Idempotent.
    pub fn insert(&mut self, set: HLLSet) -> ValueCid {
        let cid = set.content_key();
        self.values.entry(cid.clone()).or_insert(set);
        cid
    }

    pub fn get(&self, cid: &str) -> Option<&HLLSet> {
        self.values.get(cid)
    }

    pub fn contains(&self, cid: &str) -> bool {
        self.values.contains_key(cid)
    }

    pub fn len(&self) -> usize {
        self.values.len()
    }

    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    pub fn iter(&self) -> impl Iterator<Item = (&ValueCid, &HLLSet)> {
        self.values.iter()
    }

    pub fn cids(&self) -> impl Iterator<Item = &ValueCid> {
        self.values.keys()
    }
}

/// One program node: a compiled expression with its stack effect.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct OpSpec {
    pub cid: OpCid,
    pub arity: usize,
    pub outputs: usize,
    pub expr: Expression,
}

/// The processor side: programs addressed by the SHA1 of their source.
///
/// Adding the same source twice is idempotent — same bytes, same program.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct OpTable {
    ops: BTreeMap<OpCid, OpSpec>,
}

impl OpTable {
    pub fn new() -> Self {
        Self::default()
    }

    /// Compile and register a program. Returns its content-addressed id.
    pub fn add(&mut self, source: &str) -> Result<OpCid, EvalError> {
        let expr = Expression::compile(source)?;
        let cid = expr.op_cid();
        self.ops.entry(cid.clone()).or_insert_with(|| OpSpec {
            cid: cid.clone(),
            arity: expr.arity,
            outputs: expr.outputs,
            expr,
        });
        Ok(cid)
    }

    pub fn get(&self, cid: &str) -> Option<&OpSpec> {
        self.ops.get(cid)
    }

    pub fn contains(&self, cid: &str) -> bool {
        self.ops.contains_key(cid)
    }

    pub fn len(&self) -> usize {
        self.ops.len()
    }

    pub fn is_empty(&self) -> bool {
        self.ops.is_empty()
    }

    pub fn iter(&self) -> impl Iterator<Item = (&OpCid, &OpSpec)> {
        self.ops.iter()
    }
}

/// The source of an edge — a value node or an op output port.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum Port {
    /// A value node (the lattice side).
    Value(ValueCid),
    /// The `index`-th output of an op (the processor side).
    OpOut { op: OpCid, index: usize },
}

/// The destination of an edge — an op input port.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum Target {
    /// The `index`-th input of an op.
    OpIn { op: OpCid, index: usize },
}

/// A directed edge, by reference.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct Edge {
    pub from: Port,
    pub to: Target,
}

/// The operational graph: both sides plus the edges.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct OpGraph {
    pub values: ValueStore,
    pub ops: OpTable,
    pub edges: Vec<Edge>,
}

impl OpGraph {
    pub fn new() -> Self {
        Self::default()
    }

    // ── lattice side ────────────────────────────────────────────────────

    /// Store a value (idempotent) and return its CID.
    pub fn add_value(&mut self, set: HLLSet) -> ValueCid {
        self.values.insert(set)
    }

    /// The lattice join `a ∪ b`, stored and content-addressed.
    pub fn join(&mut self, a: &ValueCid, b: &ValueCid) -> Result<ValueCid, EvalError> {
        let (a, b) = self.lookup2(a, b)?;
        Ok(self.values.insert(a.union(&b)))
    }

    /// The lattice meet `a ∩ b`, stored and content-addressed.
    pub fn meet(&mut self, a: &ValueCid, b: &ValueCid) -> Result<ValueCid, EvalError> {
        let (a, b) = self.lookup2(a, b)?;
        Ok(self.values.insert(a.intersection(&b)))
    }

    /// The lattice order: `a ⊆ b`.
    pub fn subset(&self, a: &ValueCid, b: &ValueCid) -> Result<bool, EvalError> {
        let (a, b) = self.lookup2(a, b)?;
        Ok(a.difference(&b).is_empty())
    }

    fn lookup2(&self, a: &ValueCid, b: &ValueCid) -> Result<(&HLLSet, &HLLSet), EvalError> {
        let a = self
            .values
            .get(a)
            .ok_or_else(|| EvalError::MissingValue(a.clone()))?;
        let b = self
            .values
            .get(b)
            .ok_or_else(|| EvalError::MissingValue(b.clone()))?;
        Ok((a, b))
    }

    // ── processor side ──────────────────────────────────────────────────

    /// Compile and register a program (idempotent). Returns its CID.
    pub fn add_op(&mut self, source: &str) -> Result<OpCid, EvalError> {
        self.ops.add(source)
    }

    /// Connect `from` to `to` by reference.
    pub fn connect(&mut self, from: Port, to: Target) {
        self.edges.push(Edge { from, to });
    }

    /// The consumers of a port, in deterministic (sorted, deduped) order.
    pub fn consumers_of(&self, from: &Port) -> Vec<(OpCid, usize)> {
        let mut out: Vec<(OpCid, usize)> = self
            .edges
            .iter()
            .filter_map(|e| {
                if &e.from != from {
                    return None;
                }
                match &e.to {
                    Target::OpIn { op, index } => Some((op.clone(), *index)),
                }
            })
            .collect();
        out.sort();
        out.dedup();
        out
    }

    /// Convenience: the program CID for a source, without registering it.
    pub fn op_cid_for(source: &str) -> OpCid {
        op_cid(source)
    }
}
