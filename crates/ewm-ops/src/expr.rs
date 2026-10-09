//! Programs and their evaluation.
//!
//! A **program** is a canonical postfix DSL expression. Its identity is the
//! SHA1 of its source bytes (`p:<sha1>`), so a [UM] is any expression — no
//! file persistence required. The expression runs on a local stack seeded
//! with its input values and must leave exactly `outputs` values.

use hllset_contracts::sha1_hex;
use hllset_core::{HLLSet, M};
use serde::{Deserialize, Serialize};

use crate::graph::{OpCid, OpTable, ValueCid, ValueStore};

/// Maximum inline `call` depth (guards expression-level recursion; the
/// graph-level feedback loop is the dispatcher's business, not the
/// expression's).
pub const MAX_CALL_DEPTH: usize = 128;

/// The content-addressed id of a program: `p:<sha1 of source bytes>`.
pub fn op_cid(source: &str) -> OpCid {
    format!("p:{}", sha1_hex(source.as_bytes()))
}

/// A plane coordinate for the counted quantifiers `exists_c` / `forall_c`
/// (`docs/FOL_HLLSET.md` §5): the soldered plane is `[0,M) × [0,b)`, with
/// `reg(p) = p div b` and `tz(p) = p mod b`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Coord {
    /// `tz(p) = p mod 32` — the trailing-zero count axis (32 fibres of 1024).
    Tz,
    /// `reg(p) = p div 32` — the register axis (1024 fibres of 32).
    Reg,
}

/// One word of the postfix DSL.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Word {
    /// Push a literal value (referenced by CID) onto the stack.
    Value(ValueCid),
    /// `( a -- a a )`
    Dup,
    /// `( a b -- b a )`
    Swap,
    /// `( a -- )`
    Drop,
    /// `( a b -- a ∪ b )`
    Union,
    /// `( a b -- a ∩ b )`
    Intersection,
    /// `( a b -- a \ b )`
    Difference,
    /// `( a b -- a Δ b )`
    SymmetricDifference,
    /// `( x universe -- ∃_c^universe x )` — monadic exists over the fibre of
    /// `c` (`exists_c:tz` / `exists_c:reg`), restricted to the universe.
    ExistsC(Coord),
    /// `( x universe -- ∀_c^universe x )` — monadic forall over the fibre of
    /// `c` (`forall_c:tz` / `forall_c:reg`), restricted to the universe.
    ForallC(Coord),
    /// Inline another program's words on the current stack (Forth-style
    /// composition, bounded by [`MAX_CALL_DEPTH`]).
    Call(OpCid),
}

/// A compiled program: canonical source, stack effect, and words.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Expression {
    /// The canonical source text — the exact bytes whose SHA1 is the
    /// program CID.
    pub source: String,
    /// Number of input values the expression consumes from the stack seed.
    pub arity: usize,
    /// Number of values the expression must leave on the stack.
    pub outputs: usize,
    /// The compiled words.
    pub words: Vec<Word>,
}

impl Expression {
    /// Compile a canonical source of the form
    /// `<arity> <outputs> <word>...` (whitespace separated).
    pub fn compile(source: &str) -> Result<Self, EvalError> {
        let mut tokens = source.split_whitespace();
        let arity: usize = tokens
            .next()
            .ok_or_else(|| EvalError::Parse("missing arity".into()))?
            .parse()
            .map_err(|_| EvalError::Parse("arity is not an integer".into()))?;
        let outputs: usize = tokens
            .next()
            .ok_or_else(|| EvalError::Parse("missing outputs".into()))?
            .parse()
            .map_err(|_| EvalError::Parse("outputs is not an integer".into()))?;
        let words = tokens.map(parse_word).collect::<Result<Vec<_>, _>>()?;
        Ok(Self {
            source: source.to_string(),
            arity,
            outputs,
            words,
        })
    }

    /// The program CID: `p:<sha1 of source bytes>`.
    pub fn op_cid(&self) -> OpCid {
        op_cid(&self.source)
    }

    /// Evaluate the program against `inputs` (must equal [`arity`](Self::arity)).
    pub fn run(
        &self,
        inputs: &[ValueCid],
        values: &ValueStore,
        ops: &OpTable,
    ) -> Result<Vec<HLLSet>, EvalError> {
        if inputs.len() != self.arity {
            return Err(EvalError::ArityMismatch {
                op: self.op_cid(),
                expected: self.arity,
                got: inputs.len(),
            });
        }
        let mut stack: Vec<HLLSet> = inputs
            .iter()
            .map(|cid| {
                values
                    .get(cid)
                    .cloned()
                    .ok_or_else(|| EvalError::MissingValue(cid.clone()))
            })
            .collect::<Result<_, _>>()?;
        exec_words(&self.words, &mut stack, values, ops, 0)?;
        if stack.len() != self.outputs {
            return Err(EvalError::OutputMismatch {
                op: self.op_cid(),
                expected: self.outputs,
                got: stack.len(),
            });
        }
        Ok(stack)
    }
}

fn parse_word(token: &str) -> Result<Word, EvalError> {
    match token {
        "dup" => Ok(Word::Dup),
        "swap" => Ok(Word::Swap),
        "drop" => Ok(Word::Drop),
        "union" => Ok(Word::Union),
        "inter" => Ok(Word::Intersection),
        "diff" => Ok(Word::Difference),
        "symdiff" => Ok(Word::SymmetricDifference),
        _ if token.starts_with("exists_c:") => {
            Ok(Word::ExistsC(parse_coord(&token["exists_c:".len()..])?))
        }
        _ if token.starts_with("forall_c:") => {
            Ok(Word::ForallC(parse_coord(&token["forall_c:".len()..])?))
        }
        _ if token.starts_with('@') => Ok(Word::Value(token[1..].to_string())),
        _ if token.starts_with("call:") => Ok(Word::Call(token[5..].to_string())),
        _ => Err(EvalError::UnknownWord(token.to_string())),
    }
}

fn parse_coord(name: &str) -> Result<Coord, EvalError> {
    match name {
        "tz" => Ok(Coord::Tz),
        "reg" => Ok(Coord::Reg),
        other => Err(EvalError::Parse(format!("unknown coordinate: {other}"))),
    }
}

fn exec_words(
    words: &[Word],
    stack: &mut Vec<HLLSet>,
    values: &ValueStore,
    ops: &OpTable,
    depth: usize,
) -> Result<(), EvalError> {
    if depth > MAX_CALL_DEPTH {
        return Err(EvalError::CallDepthExceeded(depth));
    }
    for word in words {
        match word {
            Word::Value(cid) => stack.push(
                values
                    .get(cid)
                    .cloned()
                    .ok_or_else(|| EvalError::MissingValue(cid.clone()))?,
            ),
            Word::Dup => {
                let a = stack.pop().ok_or(EvalError::StackUnderflow)?;
                stack.push(a.clone());
                stack.push(a);
            }
            Word::Swap => {
                let b = stack.pop().ok_or(EvalError::StackUnderflow)?;
                let a = stack.pop().ok_or(EvalError::StackUnderflow)?;
                stack.push(b);
                stack.push(a);
            }
            Word::Drop => {
                stack.pop().ok_or(EvalError::StackUnderflow)?;
            }
            Word::Union => binary(stack, |a, b| a.union(&b))?,
            Word::Intersection => binary(stack, |a, b| a.intersection(&b))?,
            Word::Difference => binary(stack, |a, b| a.difference(&b))?,
            Word::SymmetricDifference => {
                binary(stack, |a, b| a.difference(&b).union(&b.difference(&a)))?
            }
            Word::ExistsC(coord) => {
                let universe = stack.pop().ok_or(EvalError::StackUnderflow)?;
                let x = stack.pop().ok_or(EvalError::StackUnderflow)?;
                stack.push(exists_c(&x, *coord, &universe));
            }
            Word::ForallC(coord) => {
                let universe = stack.pop().ok_or(EvalError::StackUnderflow)?;
                let x = stack.pop().ok_or(EvalError::StackUnderflow)?;
                stack.push(forall_c(&x, *coord, &universe));
            }
            Word::Call(callee) => {
                let spec = ops
                    .get(callee)
                    .ok_or_else(|| EvalError::MissingOp(callee.clone()))?;
                let callee_words = spec.expr.words.clone();
                exec_words(&callee_words, stack, values, ops, depth + 1)?;
            }
        }
    }
    Ok(())
}

fn binary<F>(stack: &mut Vec<HLLSet>, f: F) -> Result<(), EvalError>
where
    F: FnOnce(HLLSet, HLLSet) -> HLLSet,
{
    let b = stack.pop().ok_or(EvalError::StackUnderflow)?;
    let a = stack.pop().ok_or(EvalError::StackUnderflow)?;
    stack.push(f(a, b));
    Ok(())
}

/// `∃_c^U X` — the union of the full-plane fibres of `c` that meet `X`,
/// restricted to the universe `U` (`docs/FOL_HLLSET.md` §5). Computed on the
/// dense `[u32; M]` registers in `O(M)`: for `tz`, include every universe
/// column whose `tz` value occurs in `X ∩ U`; for `reg`, include every
/// universe register that `X ∩ U` touches.
fn exists_c(x: &HLLSet, coord: Coord, universe: &HLLSet) -> HLLSet {
    let xd = x.intersection(universe).to_dense();
    let ud = universe.to_dense();
    let mut out = vec![0u32; M];
    match coord {
        Coord::Tz => {
            // tz values present anywhere in X ∩ U.
            let tz_present = xd.iter().fold(0u32, |acc, &v| acc | v);
            for (r, &ur) in ud.iter().enumerate() {
                out[r] = ur & tz_present;
            }
        }
        Coord::Reg => {
            for (r, &ur) in ud.iter().enumerate() {
                if xd[r] != 0 {
                    out[r] = ur;
                }
            }
        }
    }
    HLLSet::from_dense(&out)
}

/// `∀_c^U X` — the union of the full-plane fibres of `c` wholly contained in
/// `X`, restricted to the universe `U` (`docs/FOL_HLLSET.md` §5). Computed on
/// the dense `[u32; M]` registers in `O(M)`: a `tz` column qualifies when
/// every register that has it in `U` also has it in `X`; a `reg` register
/// qualifies when all its `U` bits are in `X`.
fn forall_c(x: &HLLSet, coord: Coord, universe: &HLLSet) -> HLLSet {
    let xd = x.to_dense();
    let ud = universe.to_dense();
    let mut out = vec![0u32; M];
    match coord {
        Coord::Tz => {
            // Column tz qualifies iff (ud[r] has tz) implies (xd[r] has tz)
            // for every register r: the AND over r of (xd[r] | !ud[r]).
            let ok = ud
                .iter()
                .zip(&xd)
                .fold(u32::MAX, |acc, (&ur, &xr)| acc & (xr | !ur));
            for (r, &ur) in ud.iter().enumerate() {
                out[r] = ur & ok;
            }
        }
        Coord::Reg => {
            for (r, &ur) in ud.iter().enumerate() {
                if ur != 0 && (ur & !xd[r]) == 0 {
                    out[r] = ur;
                }
            }
        }
    }
    HLLSet::from_dense(&out)
}

/// Evaluation / compilation errors.
#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum EvalError {
    #[error("parse error: {0}")]
    Parse(String),
    #[error("unknown word: {0}")]
    UnknownWord(String),
    #[error("unknown universe: {0}")]
    UnknownUniverse(String),
    #[error("stack underflow")]
    StackUnderflow,
    #[error("arity mismatch for {op}: expected {expected}, got {got}")]
    ArityMismatch {
        op: OpCid,
        expected: usize,
        got: usize,
    },
    #[error("output mismatch for {op}: expected {expected}, got {got}")]
    OutputMismatch {
        op: OpCid,
        expected: usize,
        got: usize,
    },
    #[error("missing value: {0}")]
    MissingValue(ValueCid),
    #[error("missing op: {0}")]
    MissingOp(OpCid),
    #[error("input port not filled: {op}[{port}]")]
    InputNotFilled { op: OpCid, port: usize },
    #[error("call depth exceeded at {0}")]
    CallDepthExceeded(usize),
}
