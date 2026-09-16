//! The dispatcher — the stack pop machine that traverses the operational
//! graph.
//!
//! ```text
//! loop:
//!     pop stack
//!     Value(cid) → route to every consumer edge; when a consumer's in-ports
//!                  are all filled, push Fire(consumer)
//!     Fire(op)   → run op's expression on its filled inputs, store outputs
//!                  as values (idempotent), route them, record the firing
//! ```
//!
//! Fan-out is by reference: one value token routes the same CID to every
//! consumer — nothing is copied. The LIFO stack makes the fire sequence
//! deterministic for a given graph. The dispatcher is disposable: it holds
//! only the token stack and per-port fill state, both recomputable from the
//! graph; the committed state lives elsewhere.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::expr::EvalError;
use crate::graph::{OpCid, OpGraph, Port, Target, ValueCid};

/// A stack token: a produced value (routed by reference) or a ready [UM].
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Token {
    Value(ValueCid),
    Fire(OpCid),
}

/// One [UM] firing: the program, its input CIDs, and the CIDs it produced.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct FireRecord {
    pub op: OpCid,
    pub inputs: Vec<ValueCid>,
    pub outputs: Vec<ValueCid>,
}

/// The observed fire sequence — the state-machine trajectory.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct FireLog {
    pub records: Vec<FireRecord>,
}

impl FireLog {
    pub fn len(&self) -> usize {
        self.records.len()
    }

    pub fn is_empty(&self) -> bool {
        self.records.is_empty()
    }

    /// The produced values in fire order (the persisted value stack; the
    /// last element is the top of stack — the current state).
    pub fn produced_values(&self) -> Vec<ValueCid> {
        self.records
            .iter()
            .flat_map(|r| r.outputs.iter().cloned())
            .collect()
    }
}

/// Why the dispatcher stopped.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum CommitReason {
    /// The token stack drained (acyclic graph reached its fixpoint).
    Quiescence,
    /// The caller's commit predicate returned `true`.
    Predicate,
    /// The fire budget was exhausted.
    FireBudget,
}

/// A commit point: where the dispatcher stopped and how many [UM]s fired.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CommitPoint {
    pub fired: usize,
    pub reason: CommitReason,
}

/// The stack pop machine.
pub struct Dispatcher<'g> {
    graph: &'g mut OpGraph,
    stack: Vec<Token>,
    /// `(op, in-port) → value CID` — the fill state. Cleared after each
    /// firing so feedback loops can refill ports with new values.
    filled: HashMap<(OpCid, usize), ValueCid>,
    log: FireLog,
}

impl<'g> Dispatcher<'g> {
    pub fn new(graph: &'g mut OpGraph) -> Self {
        Self {
            graph,
            stack: Vec::new(),
            filled: HashMap::new(),
            log: FireLog::default(),
        }
    }

    /// Seed one value into the traversal.
    pub fn seed_value(&mut self, cid: &ValueCid) {
        self.stack.push(Token::Value(cid.clone()));
    }

    /// Seed several values into the traversal.
    pub fn seed_values(&mut self, cids: &[ValueCid]) {
        for cid in cids {
            self.seed_value(cid);
        }
    }

    /// The fire sequence observed so far.
    pub fn log(&self) -> &FireLog {
        &self.log
    }

    /// Run until the stack is empty. For acyclic graphs this terminates;
    /// for graphs with feedback use [`run_limited`](Self::run_limited) or
    /// [`run_until`](Self::run_until).
    pub fn run(&mut self) -> Result<&FireLog, EvalError> {
        self.run_until(usize::MAX, |_| false)?;
        Ok(&self.log)
    }

    /// Run at most `max_fires` [UM] firings, then stop (the unpopped tokens
    /// stay on the stack, so the traversal can resume).
    pub fn run_limited(&mut self, max_fires: usize) -> Result<&FireLog, EvalError> {
        self.run_until(max_fires, |_| false)?;
        Ok(&self.log)
    }

    /// Run until one of three commit conditions holds: the token stack
    /// drains (quiescence), the predicate over the fire log returns `true`
    /// (a caller-defined commit point, checked before each firing), or
    /// `max_fires` firings have run. Returns the commit point.
    pub fn run_until<P>(&mut self, max_fires: usize, mut pred: P) -> Result<CommitPoint, EvalError>
    where
        P: FnMut(&FireLog) -> bool,
    {
        while let Some(token) = self.stack.pop() {
            match token {
                Token::Value(cid) => self.route_value(&cid)?,
                Token::Fire(op) => {
                    if self.log.records.len() >= max_fires {
                        self.stack.push(Token::Fire(op));
                        return Ok(CommitPoint {
                            fired: self.log.records.len(),
                            reason: CommitReason::FireBudget,
                        });
                    }
                    if pred(&self.log) {
                        self.stack.push(Token::Fire(op));
                        return Ok(CommitPoint {
                            fired: self.log.records.len(),
                            reason: CommitReason::Predicate,
                        });
                    }
                    self.fire(&op)?;
                }
            }
        }
        Ok(CommitPoint {
            fired: self.log.records.len(),
            reason: CommitReason::Quiescence,
        })
    }

    /// Route a value token to every consumer edge (fan-out by reference).
    fn route_value(&mut self, cid: &ValueCid) -> Result<(), EvalError> {
        let consumers = self.graph.consumers_of(&Port::Value(cid.clone()));
        for (op, port) in consumers {
            if self.fill_port(op.clone(), port, cid.clone()) && self.is_ready(&op)? {
                self.stack.push(Token::Fire(op));
            }
        }
        Ok(())
    }

    /// Fill one input port. Returns `true` only when the port was empty —
    /// a duplicate value token for an already-filled port is a no-op, which
    /// is what prevents double firings.
    fn fill_port(&mut self, op: OpCid, port: usize, value: ValueCid) -> bool {
        use std::collections::hash_map::Entry;
        match self.filled.entry((op, port)) {
            Entry::Vacant(e) => {
                e.insert(value);
                true
            }
            Entry::Occupied(_) => false,
        }
    }

    fn is_ready(&self, op: &OpCid) -> Result<bool, EvalError> {
        let spec = self
            .graph
            .ops
            .get(op)
            .ok_or_else(|| EvalError::MissingOp(op.clone()))?;
        Ok((0..spec.arity).all(|i| self.filled.contains_key(&(op.clone(), i))))
    }

    /// Run one [UM]: read its inputs, clear its ports (so feedback can
    /// refill them), evaluate, store the outputs as values, route them.
    fn fire(&mut self, op: &OpCid) -> Result<(), EvalError> {
        let spec = self
            .graph
            .ops
            .get(op)
            .ok_or_else(|| EvalError::MissingOp(op.clone()))?
            .clone();

        let inputs: Vec<ValueCid> = (0..spec.arity)
            .map(|i| {
                self.filled
                    .get(&(op.clone(), i))
                    .cloned()
                    .ok_or_else(|| EvalError::InputNotFilled {
                        op: op.clone(),
                        port: i,
                    })
            })
            .collect::<Result<_, _>>()?;

        // Clear the ports before evaluating: the firing consumed its
        // inputs, and outputs may route straight back into this op.
        for i in 0..spec.arity {
            self.filled.remove(&(op.clone(), i));
        }

        let outputs = spec.expr.run(&inputs, &self.graph.values, &self.graph.ops)?;

        // Store outputs (idempotent) — the lattice side grows only by
        // discovering new nodes.
        let mut output_cids = Vec::with_capacity(outputs.len());
        for set in outputs {
            output_cids.push(self.graph.values.insert(set));
        }

        // Route the outputs. Value tokens are pushed onto the stack (value
        // consumers resolve when they are popped); `OpOut` edges resolve
        // immediately because they name the producing port, not a value.
        for cid in &output_cids {
            self.stack.push(Token::Value(cid.clone()));
        }
        for (index, cid) in output_cids.iter().enumerate() {
            let from = Port::OpOut {
                op: op.clone(),
                index,
            };
            let consumers = self.graph.consumers_of(&from);
            for (consumer, port) in consumers {
                if self.fill_port(consumer.clone(), port, cid.clone())
                    && self.is_ready(&consumer)?
                {
                    self.stack.push(Token::Fire(consumer));
                }
            }
        }

        self.log.records.push(FireRecord {
            op: op.clone(),
            inputs,
            outputs: output_cids,
        });
        Ok(())
    }
}

/// A convenience edge constructor: `Value(cid) → OpIn{op, index}`.
pub fn value_edge(value: &ValueCid, op: &OpCid, index: usize) -> (Port, Target) {
    (
        Port::Value(value.clone()),
        Target::OpIn {
            op: op.clone(),
            index,
        },
    )
}

/// A convenience edge constructor: `OpOut{op, out} → OpIn{consumer, index}`.
pub fn op_edge(op: &OpCid, out: usize, consumer: &OpCid, index: usize) -> (Port, Target) {
    (
        Port::OpOut {
            op: op.clone(),
            index: out,
        },
        Target::OpIn {
            op: consumer.clone(),
            index,
        },
    )
}
