//! The boot-script DSL: a plain-text operational graph.
//!
//! ```text
//! # comment
//! value <name> <token>...            define a value HLLSet from tokens
//! def <name> ( <in> -- <out> ) <word>...   define a program (op)
//! link <from> -> <to>                add a directed edge
//! stack <ref>...                     boot stack (bottom → top; top = state)
//! fires <n>                          0 = run to quiescence
//! ```
//!
//! Refs resolve in this order: `@<value-name>`, `p:<op-cid>`, `h:<value-cid>`.
//! Words may be `dup swap drop union inter diff symdiff @<ref> call:<op-name>`.
//! `call:<name>` and port refs are resolved at compile time, so the program
//! identity is the SHA1 of the **resolved canonical source** — the DSL text
//! is a surface, the CID is the ground truth.

use std::collections::BTreeMap;

use hllset_core::HLLSet;

use crate::expr::EvalError;
use crate::graph::{OpCid, OpGraph, Port, Target, ValueCid};
use crate::vocab::Vocabulary;

/// A compiled boot script: the graph, the boot stack, the fire budget, and
/// the content-addressed vocabulary the names resolved into.
#[derive(Clone, Debug, Default)]
pub struct BootProgram {
    pub graph: OpGraph,
    /// Boot stack, bottom → top; the top is the current state.
    pub stack: Vec<ValueCid>,
    /// 0 = run to quiescence; otherwise the fire budget.
    pub fires: usize,
    /// The named dictionary (ops + values) with its `v:<sha1>` identity.
    pub vocab: Vocabulary,
}

/// Compile a boot script into a graph + boot stack.
pub fn compile_boot(script: &str) -> Result<BootProgram, EvalError> {
    let mut graph = OpGraph::new();
    let mut op_names: BTreeMap<String, OpCid> = BTreeMap::new();
    let mut value_names: BTreeMap<String, ValueCid> = BTreeMap::new();
    let mut stack = Vec::new();
    let mut fires = 0usize;

    for (lineno, raw) in script.lines().enumerate() {
        let line = raw.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let mut parts = line.split_whitespace();
        let directive = parts
            .next()
            .ok_or_else(|| EvalError::Parse(format!("line {}: empty directive", lineno + 1)))?;
        let rest: Vec<&str> = parts.collect();
        match directive {
            "value" => {
                let name = *rest
                    .first()
                    .ok_or_else(|| EvalError::Parse(format!("line {}: value needs a name", lineno + 1)))?;
                let tokens = &rest[1..];
                if tokens.is_empty() {
                    return Err(EvalError::Parse(format!(
                        "line {}: value {} needs tokens",
                        lineno + 1,
                        name
                    )));
                }
                let set = HLLSet::from_tokens(tokens.iter().map(|t| t.as_bytes()));
                let cid = graph.add_value(set);
                value_names.insert(name.to_string(), cid);
            }
            "def" => {
                // def name ( in -- out ) words...
                let name = *rest
                    .first()
                    .ok_or_else(|| EvalError::Parse(format!("line {}: def needs a name", lineno + 1)))?;
                let body = &rest[1..];
                let open = body.iter().position(|t| *t == "(").ok_or_else(|| {
                    EvalError::Parse(format!("line {}: def {} missing (", lineno + 1, name))
                })?;
                let close = body.iter().position(|t| *t == ")").ok_or_else(|| {
                    EvalError::Parse(format!("line {}: def {} missing )", lineno + 1, name))
                })?;
                let arity: usize = body
                    .get(open + 1)
                    .ok_or_else(|| EvalError::Parse(format!("line {}: def {} missing arity", lineno + 1, name)))?
                    .parse()
                    .map_err(|_| EvalError::Parse(format!("line {}: def {} arity is not an integer", lineno + 1, name)))?;
                let outputs: usize = body
                    .get(close - 1)
                    .ok_or_else(|| EvalError::Parse(format!("line {}: def {} missing outputs", lineno + 1, name)))?
                    .parse()
                    .map_err(|_| EvalError::Parse(format!("line {}: def {} outputs is not an integer", lineno + 1, name)))?;
                let words: Vec<String> = body[close + 1..]
                    .iter()
                    .map(|w| resolve_word(w, &value_names, &op_names))
                    .collect::<Result<_, _>>()?;
                let source = format!("{} {} {}", arity, outputs, words.join(" "));
                let cid = graph.add_op(&source)?;
                op_names.insert(name.to_string(), cid);
            }
            "link" => {
                // link <from> -> <to>
                let arrow = rest.iter().position(|t| *t == "->").ok_or_else(|| {
                    EvalError::Parse(format!("line {}: link missing ->", lineno + 1))
                })?;
                let from = resolve_port(&rest[..arrow], &value_names, &op_names)?;
                let to = resolve_target(&rest[arrow + 1..], &op_names)?;
                graph.connect(from, to);
            }
            "stack" => {
                stack = rest
                    .iter()
                    .map(|r| resolve_ref(r, &value_names))
                    .collect::<Result<_, _>>()?;
            }
            "fires" => {
                fires = rest
                    .first()
                    .ok_or_else(|| EvalError::Parse(format!("line {}: fires needs a number", lineno + 1)))?
                    .parse()
                    .map_err(|_| EvalError::Parse(format!("line {}: fires is not an integer", lineno + 1)))?;
            }
            other => {
                return Err(EvalError::Parse(format!(
                    "line {}: unknown directive {}",
                    lineno + 1,
                    other
                )));
            }
        }
    }

    Ok(BootProgram {
        graph,
        stack,
        fires,
        vocab: Vocabulary {
            ops: op_names,
            values: value_names,
        },
    })
}

/// Resolve `@name` → `@h:<cid>`; `call:name` → `call:p:<cid>`; otherwise
/// leave the token as-is (built-in word or already a CID reference).
fn resolve_word(
    word: &str,
    value_names: &BTreeMap<String, ValueCid>,
    op_names: &BTreeMap<String, OpCid>,
) -> Result<String, EvalError> {
    if let Some(name) = word.strip_prefix('@') {
        return value_names
            .get(name)
            .map(|cid| format!("@{cid}"))
            .ok_or_else(|| EvalError::UnknownWord(format!("@{name}")));
    }
    if let Some(name) = word.strip_prefix("call:") {
        return op_names
            .get(name)
            .map(|cid| format!("call:{cid}"))
            .ok_or_else(|| EvalError::MissingOp(name.to_string()));
    }
    Ok(word.to_string())
}

/// Resolve a stack ref: `@name` → value CID; otherwise the token itself
/// must be a CID.
fn resolve_ref(
    r: &str,
    value_names: &BTreeMap<String, ValueCid>,
) -> Result<ValueCid, EvalError> {
    if let Some(name) = r.strip_prefix('@') {
        return value_names
            .get(name)
            .cloned()
            .ok_or_else(|| EvalError::UnknownWord(format!("@{name}")));
    }
    Ok(r.to_string())
}

/// Resolve a link source: `value:<ref>` or `out:<opref>.<index>`.
fn resolve_port(
    parts: &[&str],
    value_names: &BTreeMap<String, ValueCid>,
    op_names: &BTreeMap<String, OpCid>,
) -> Result<Port, EvalError> {
    let token = parts
        .first()
        .ok_or_else(|| EvalError::Parse("link source missing".into()))?;
    if let Some(rest) = token.strip_prefix("value:") {
        let cid = resolve_ref(rest, value_names)?;
        return Ok(Port::Value(cid));
    }
    if let Some(rest) = token.strip_prefix("out:") {
        let (op, index) = split_op_port(rest, op_names)?;
        return Ok(Port::OpOut { op, index });
    }
    Err(EvalError::Parse(format!("bad link source {token}")))
}

/// Resolve a link target: `in:<opref>.<index>`.
fn resolve_target(
    parts: &[&str],
    op_names: &BTreeMap<String, OpCid>,
) -> Result<Target, EvalError> {
    let token = parts
        .first()
        .ok_or_else(|| EvalError::Parse("link target missing".into()))?;
    if let Some(rest) = token.strip_prefix("in:") {
        let (op, index) = split_op_port(rest, op_names)?;
        return Ok(Target::OpIn { op, index });
    }
    Err(EvalError::Parse(format!("bad link target {token}")))
}

/// Split `<opref>.<index>` and resolve `opref` (name or `p:<cid>`).
fn split_op_port(
    s: &str,
    op_names: &BTreeMap<String, OpCid>,
) -> Result<(OpCid, usize), EvalError> {
    let (opref, index) = s
        .rsplit_once('.')
        .ok_or_else(|| EvalError::Parse(format!("port {s} must be op.index")))?;
    let index: usize = index
        .parse()
        .map_err(|_| EvalError::Parse(format!("port index in {s} is not an integer")))?;
    let op = if let Some(name) = opref.strip_prefix("p:") {
        // already a program CID
        format!("p:{name}")
    } else {
        op_names
            .get(opref)
            .cloned()
            .ok_or_else(|| EvalError::MissingOp(opref.to_string()))?
    };
    Ok((op, index))
}
