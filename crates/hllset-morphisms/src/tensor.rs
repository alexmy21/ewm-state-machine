//! Tensor morphisms — `conv(n, dim=N)` for arbitrary N.
//!
//! A [`Tensor`] is an N-dimensional token collection in lexicographic
//! (row-major, last-axis-fastest) order. Ingest slides `n×…×n` windows
//! (`n = 1..=4`) with a **full PAD border** of `TENSOR_BORDER` cells on
//! every axis, so the count-constrained ordered walk can always treat the
//! candidate cell as the window's lexicographically-last cell — the N-d
//! analog of the 1D n-gram chain and the 2D grid walk.
//!
//! Channels: `1×…×1` (shared G1, seed 0) and, per dimension `dim`, the
//! `2^dim` (seed `(dim−1)·3 + 1`), `3^dim` (+2), `4^dim` (+3) channels.
//! The set path is LUT-first over the first three channels; the order path
//! is the count-constrained lexicographic walk with joint
//! `2^dim ∧ 3^dim ∧ 4^dim` checks.
//!
//! Examples: `dim=1` sequences, `dim=2` grids (see [`crate::grid`]),
//! `dim=3` volumes — RGB frames, `W×H×T` clips, `W×H×D` depth.

use std::collections::{BTreeMap, BTreeSet};

use ::hllset_lut::LutIndex;
use hllset_contracts::BitAddress;
use hllset_core::HLLSet;

use crate::api::{join, PAD};
use crate::conv::{channel_name, seed, ConvSpec};
use crate::hllset_lut::HllsetLut;
use crate::materialize::materialize as materialize_lut_first;
use crate::tf::TfTable;

/// Largest window order in the tensor regime (the `4^dim` order channel).
pub const TENSOR_MAX_N: usize = 4;
/// PAD border on every axis (full border, so the candidate is always the
/// lexicographically-last cell of its window).
pub const TENSOR_BORDER: usize = TENSOR_MAX_N - 1;

/// The channel seed of an `n^dim` tensor window.
fn tseed(n: usize, dim: usize) -> u64 {
    seed(ConvSpec::new(n as u8, dim as u8))
}

/// An N-dimensional token collection in lexicographic order.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Tensor {
    pub shape: Vec<usize>,
    pub cells: Vec<Vec<u8>>,
}

impl Tensor {
    pub fn new(shape: Vec<usize>, cells: Vec<Vec<u8>>) -> Self {
        let total: usize = shape.iter().product();
        assert_eq!(cells.len(), total, "tensor cells must match the shape");
        Self { shape, cells }
    }

    pub fn dims(&self) -> usize {
        self.shape.len()
    }

    pub fn len(&self) -> usize {
        self.cells.len()
    }

    pub fn is_empty(&self) -> bool {
        self.cells.is_empty()
    }
}

/// The result of N-d convolution ingestion: four channels (`1^dim` …
/// `4^dim`), four LUTs, TF, and the preservation side effects.
#[derive(Clone, Debug)]
pub struct TensorIngested {
    /// Channels indexed by `n - 1` (`1^dim`, `2^dim`, `3^dim`, `4^dim`).
    pub channels: [HLLSet; TENSOR_MAX_N],
    /// LUTs indexed by `n - 1`; the fiber of a bit holds the window's
    /// lexicographically-first cell token (the pointed-to token).
    pub luts: [LutIndex; TENSOR_MAX_N],
    /// Monotonic term-frequency table over the original cells.
    pub tf: TfTable,
    pub shape: Vec<usize>,
    pub pad: Vec<u8>,
    /// `1^dim ∪ 2^dim ∪ 3^dim` — the token presentation (the `4^dim` is
    /// order-only).
    pub projection: HLLSet,
    /// SHA1 of the projection (`h:<sha1>`).
    pub key: String,
    /// SHA1 keys of the four channels.
    pub keys: [String; TENSOR_MAX_N],
    /// Named channel HLLSets registered and touch-counted (preservation).
    pub hllset_lut: HllsetLut,
}

impl Default for TensorIngested {
    fn default() -> Self {
        Self::new()
    }
}

impl TensorIngested {
    pub fn new() -> Self {
        let empty = HLLSet::new();
        let key = empty.content_key();
        Self {
            channels: std::array::from_fn(|_| HLLSet::new()),
            luts: std::array::from_fn(|_| LutIndex::default()),
            tf: TfTable::new(),
            shape: Vec::new(),
            pad: PAD.to_vec(),
            projection: empty,
            key,
            keys: std::array::from_fn(|_| String::new()),
            hllset_lut: HllsetLut::new(),
        }
    }
}

/// Row-major strides of a shape (last axis fastest).
fn strides(shape: &[usize]) -> Vec<usize> {
    let mut out = vec![1usize; shape.len()];
    for i in (0..shape.len().saturating_sub(1)).rev() {
        out[i] = out[i + 1] * shape[i + 1];
    }
    out
}

/// Convert a lexicographic cell index into a multi-index.
fn to_multi(mut idx: usize, shape: &[usize], strides: &[usize]) -> Vec<usize> {
    let mut out = vec![0usize; shape.len()];
    for i in 0..shape.len() {
        out[i] = idx / strides[i];
        idx %= strides[i];
    }
    out
}

/// Ingest a tensor with the default pad (`<PAD>`).
pub fn ingest_tensor(tensor: &Tensor) -> TensorIngested {
    ingest_tensor_with_pad(tensor, PAD)
}

/// Ingest a tensor with a custom pad.
pub fn ingest_tensor_with_pad(tensor: &Tensor, pad: &[u8]) -> TensorIngested {
    let mut out = TensorIngested::new();
    out.shape = tensor.shape.clone();
    out.pad = pad.to_vec();
    if tensor.is_empty() {
        return out;
    }

    let dim = tensor.shape.len();
    let strides = strides(&tensor.shape);
    let padded_shape: Vec<usize> = tensor
        .shape
        .iter()
        .map(|s| s + 2 * TENSOR_BORDER)
        .collect();

    for n in 1..=TENSOR_MAX_N {
        let ch = n - 1;
        let seed_n = tseed(n, dim);
        // Odometer over anchor positions.
        let mut anchor = vec![0usize; dim];
        loop {
            // Is the anchor cell (the window's first cell) a real cell?
            let anchor_real = anchor
                .iter()
                .zip(&tensor.shape)
                .all(|(a, s)| *a >= TENSOR_BORDER && *a < TENSOR_BORDER + s);

            // Build the n^dim window in lexicographic order.
            let mut parts: Vec<&[u8]> = Vec::with_capacity(n.pow(dim as u32));
            for w in 0..n.pow(dim as u32) {
                let mut off = vec![0usize; dim];
                let mut t = w;
                for i in (0..dim).rev() {
                    off[i] = t % n;
                    t /= n;
                }
                let mut idx = 0usize;
                let mut all_real = true;
                for axis in 0..dim {
                    let p = anchor[axis] + off[axis];
                    if p >= TENSOR_BORDER && p < TENSOR_BORDER + tensor.shape[axis] {
                        idx += (p - TENSOR_BORDER) * strides[axis];
                    } else {
                        all_real = false;
                    }
                }
                if all_real {
                    parts.push(&tensor.cells[idx]);
                } else {
                    parts.push(pad);
                }
            }
            let ngram = join(&parts);
            let addr = BitAddress::of_token_seeded(&ngram, seed_n);
            out.channels[ch].add_bit(addr.bit());
            // Real anchors register their token (the window's first cell).
            if anchor_real {
                let r_idx = anchor
                    .iter()
                    .zip(&strides)
                    .map(|(a, s)| (a - TENSOR_BORDER) * s)
                    .sum::<usize>();
                out.luts[ch]
                    .insert_token_at(tensor.cells[r_idx].clone(), addr.bit());
            }
            // Advance the anchor odometer.
            let mut carry = true;
            for axis in (0..dim).rev() {
                if carry {
                    anchor[axis] += 1;
                    if anchor[axis] > padded_shape[axis] - n {
                        anchor[axis] = 0;
                    } else {
                        carry = false;
                    }
                }
            }
            if carry {
                break;
            }
        }
    }

    for cell in &tensor.cells {
        out.tf.increment(cell);
    }

    out.projection = out.channels[0..3]
        .iter()
        .fold(HLLSet::new(), |acc, s| acc.union(s));
    out.key = out.projection.content_key();
    for (n, ch) in out.channels.iter().enumerate() {
        out.keys[n] = ch.content_key();
        let name = channel_name(n as u8 + 1, dim as u8);
        out.hllset_lut.register_named(&name, &out.keys[n]);
        out.hllset_lut.touch_named(&name, &out.keys[n]);
    }
    out
}

/// The plain-set restoration: LUT-first over `1^dim`, `2^dim`, `3^dim` (the
/// `4^dim` is order-only). Collided bits keep every candidate.
pub fn materialize_tensor_no_order(ing: &TensorIngested) -> BTreeSet<Vec<u8>> {
    let pairs: Vec<(&HLLSet, &LutIndex)> = (0..3)
        .map(|i| (&ing.channels[i], &ing.luts[i]))
        .collect();
    materialize_lut_first(&pairs)
}

/// Restore the tensor in its original lexicographic order (greedy walk).
pub fn materialize_tensor(ing: &TensorIngested) -> Tensor {
    materialize_tensor_with(ing, 1)
}

/// Restore the tensor with beam search of the given width.
pub fn materialize_tensor_beam(ing: &TensorIngested, width: usize) -> Tensor {
    materialize_tensor_with(ing, width.max(1))
}

fn materialize_tensor_with(ing: &TensorIngested, beam: usize) -> Tensor {
    let tokens = materialize_tensor_no_order(ing);
    let cells = tensor_order_tokens(&tokens, ing, beam);
    Tensor {
        shape: ing.shape.clone(),
        cells,
    }
}

/// The ordered restoration: count-constrained, TF-scored, joint-window
/// decoding over the lexicographic tensor.
fn tensor_order_tokens(
    tokens: &BTreeSet<Vec<u8>>,
    ing: &TensorIngested,
    beam: usize,
) -> Vec<Vec<u8>> {
    let total: usize = ing.shape.iter().product();
    if tokens.is_empty() || total == 0 {
        return Vec::new();
    }

    let remaining: BTreeMap<Vec<u8>, usize> = tokens
        .iter()
        .map(|t| (t.clone(), ing.tf.count(t) as usize))
        .filter(|(_, c)| *c > 0)
        .collect();

    if beam > 1 {
        if let Some(path) = tensor_beam_search(tokens, ing, beam, &remaining) {
            return path;
        }
    }

    let mut path = Vec::new();
    let mut rem = remaining;
    let mut budget = 20_000usize;
    if tensor_walk(tokens, ing, &mut path, &mut rem, &mut budget) {
        return path;
    }

    // Fallback: cannot reconstruct — return the unordered set.
    tokens.iter().cloned().collect()
}

/// `true` when the `n^dim` channel contains the atom of `bytes`.
fn tensor_channel_contains(ing: &TensorIngested, n: usize, bytes: &[u8]) -> bool {
    let addr = BitAddress::of_token_seeded(bytes, tseed(n, ing.shape.len()));
    ing.channels[n - 1].has_bit(addr.reg(), addr.tz())
}

/// Build the bytes of the `n^dim` window anchored at padded `anchor`.
///
/// All cells resolve from `path`/PAD; `candidate = Some((idx, x))` overrides
/// the lexicographically-last cell (the candidate during the walk).
fn tensor_window(
    ing: &TensorIngested,
    n: usize,
    anchor: &[usize],
    path: &[Vec<u8>],
    candidate: Option<(usize, &[u8])>,
) -> Vec<u8> {
    let dim = ing.shape.len();
    let strides = strides(&ing.shape);
    let pad = ing.pad.as_slice();
    let cells_in_window = n.pow(dim as u32);

    let mut parts: Vec<&[u8]> = Vec::with_capacity(cells_in_window);
    for w in 0..cells_in_window {
        let mut off = vec![0usize; dim];
        let mut t = w;
        for i in (0..dim).rev() {
            off[i] = t % n;
            t /= n;
        }
        let mut idx = 0usize;
        let mut all_real = true;
        for axis in 0..dim {
            let p = anchor[axis] + off[axis];
            if p >= TENSOR_BORDER && p < TENSOR_BORDER + ing.shape[axis] {
                idx += (p - TENSOR_BORDER) * strides[axis];
            } else {
                all_real = false;
            }
        }
        if all_real {
            match candidate {
                Some((cidx, x)) if cidx == idx => parts.push(x),
                _ => parts.push(&path[idx]),
            }
        } else {
            parts.push(pad);
        }
    }
    join(&parts)
}

/// `true` when candidate `x` satisfies every joint window check at cell
/// `idx`.
fn tensor_successor_ok(
    ing: &TensorIngested,
    path: &[Vec<u8>],
    idx: usize,
    x: &[u8],
) -> bool {
    let strides = strides(&ing.shape);
    let multi = to_multi(idx, &ing.shape, &strides);
    (2..=TENSOR_MAX_N).all(|n| {
        // The window whose lexicographically-last cell is the candidate:
        // padded anchor = real + TENSOR_MAX_N - n on every axis.
        let anchor: Vec<usize> = multi
            .iter()
            .map(|m| m + TENSOR_MAX_N - n)
            .collect();
        let bytes = tensor_window(ing, n, &anchor, path, Some((idx, x)));
        tensor_channel_contains(ing, n, &bytes)
    })
}

/// `true` when the completed tensor ends at the trailing PAD border.
fn tensor_termination_ok(ing: &TensorIngested, path: &[Vec<u8>]) -> bool {
    let dim = ing.shape.len();
    (2..=TENSOR_MAX_N).all(|n| {
        // The window whose lexicographically-last cell is the first PAD cell
        // after the last real cell: padded coords (s_i + 2) on all axes
        // except the last, which is (s_last + 3). Anchor = last - (n - 1).
        let anchor: Vec<usize> = (0..dim)
            .map(|axis| {
                let last_padded = if axis == dim - 1 {
                    ing.shape[axis] + 3
                } else {
                    ing.shape[axis] + 2
                };
                last_padded - (n - 1)
            })
            .collect();
        let bytes = tensor_window(ing, n, &anchor, path, None);
        tensor_channel_contains(ing, n, &bytes)
    })
}

/// Depth-first count-constrained walk over the lexicographic tensor.
fn tensor_walk(
    tokens: &BTreeSet<Vec<u8>>,
    ing: &TensorIngested,
    path: &mut Vec<Vec<u8>>,
    remaining: &mut BTreeMap<Vec<u8>, usize>,
    budget: &mut usize,
) -> bool {
    if *budget == 0 {
        return false;
    }
    *budget -= 1;

    let total: usize = ing.shape.iter().product();
    if path.len() == total {
        return tensor_termination_ok(ing, path);
    }

    let idx = path.len();
    let mut nexts: Vec<Vec<u8>> = tokens
        .iter()
        .filter(|t| remaining.get(*t).copied().unwrap_or(0) > 0)
        .filter(|t| tensor_successor_ok(ing, path, idx, t))
        .cloned()
        .collect();
    nexts.sort_by(|a, b| ing.tf.count(b).cmp(&ing.tf.count(a)).then(a.cmp(b)));

    for next in nexts {
        if let Some(n) = remaining.get_mut(&next) {
            *n -= 1;
        }
        path.push(next.clone());
        if tensor_walk(tokens, ing, path, remaining, budget) {
            return true;
        }
        path.pop();
        if let Some(n) = remaining.get_mut(&next) {
            *n += 1;
        }
    }
    false
}

/// One tensor beam-search hypothesis.
struct TensorBeamState {
    path: Vec<Vec<u8>>,
    score: u64,
    remaining: BTreeMap<Vec<u8>, usize>,
}

fn sort_tensor_beam(beam: &mut [TensorBeamState]) {
    beam.sort_by(|a, b| b.score.cmp(&a.score).then(a.path.cmp(&b.path)));
}

/// Beam search over the lexicographic tensor: keep the top-`width` partial
/// tensors by cumulative TF score, expand one cell at a time under the count
/// constraint, and return the highest-scoring completed tensor.
fn tensor_beam_search(
    tokens: &BTreeSet<Vec<u8>>,
    ing: &TensorIngested,
    width: usize,
    initial_remaining: &BTreeMap<Vec<u8>, usize>,
) -> Option<Vec<Vec<u8>>> {
    let total: usize = ing.shape.iter().product();
    let width = width.max(1);

    let mut beam: Vec<TensorBeamState> = vec![TensorBeamState {
        path: Vec::new(),
        score: 0,
        remaining: initial_remaining.clone(),
    }];
    let mut seen: BTreeSet<Vec<Vec<u8>>> = BTreeSet::new();
    let mut completed: Vec<(Vec<Vec<u8>>, u64)> = Vec::new();

    for _ in 0..=total {
        if beam.is_empty() {
            break;
        }

        let mut next: Vec<TensorBeamState> = Vec::new();
        for st in &beam {
            if st.path.len() == total {
                if tensor_termination_ok(ing, &st.path) {
                    completed.push((st.path.clone(), st.score));
                }
                continue;
            }
            let idx = st.path.len();
            for cand in tokens {
                if st.remaining.get(cand).copied().unwrap_or(0) == 0 {
                    continue;
                }
                if !tensor_successor_ok(ing, &st.path, idx, cand) {
                    continue;
                }
                let mut new_path = st.path.clone();
                new_path.push(cand.clone());
                if !seen.insert(new_path.clone()) {
                    continue;
                }
                let mut new_rem = st.remaining.clone();
                if let Some(n) = new_rem.get_mut(cand) {
                    *n -= 1;
                }
                next.push(TensorBeamState {
                    path: new_path,
                    score: st.score + ing.tf.count(cand),
                    remaining: new_rem,
                });
            }
        }

        if next.is_empty() {
            break;
        }
        sort_tensor_beam(&mut next);
        next.truncate(width);
        beam = next;
    }

    completed.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
    completed.into_iter().next().map(|(path, _)| path)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tok(s: &str) -> Vec<u8> {
        s.as_bytes().to_vec()
    }

    #[test]
    fn tensor_roundtrip_2x2x2_volume() {
        let cells: Vec<Vec<u8>> = ["a", "b", "c", "d", "e", "f", "g", "h"]
            .iter()
            .map(|s| tok(s))
            .collect();
        let t = Tensor::new(vec![2, 2, 2], cells);
        let ing = ingest_tensor(&t);
        assert_eq!(materialize_tensor(&ing), t, "2×2×2 restores exactly");
        assert_eq!(materialize_tensor_beam(&ing, 2), t, "beam-2 agrees");
    }

    #[test]
    fn tensor_roundtrip_rgb_like_4x4x3() {
        // A small RGB-like volume with repeats: cell (r, c, ch) -> tid.
        let cells: Vec<Vec<u8>> = (0..4 * 4 * 3)
            .map(|i| {
                let ch = i % 3;
                let c = (i / 3) % 4;
                let r = i / 12;
                format!("tid{}", (r * 7 + c * 5 + ch * 3) % 20).into_bytes()
            })
            .collect();
        let t = Tensor::new(vec![4, 4, 3], cells);
        let ing = ingest_tensor(&t);
        let restored = materialize_tensor(&ing);
        assert_eq!(restored, t, "4×4×3 RGB-like volume restores exactly");
        let set = materialize_tensor_no_order(&ing);
        assert_eq!(set.len(), t.cells.iter().collect::<BTreeSet<_>>().len());
    }

    #[test]
    fn tensor_roundtrip_4d_hypercube() {
        let cells: Vec<Vec<u8>> = (0..16)
            .map(|i| format!("t{}", i).into_bytes())
            .collect();
        let t = Tensor::new(vec![2, 2, 2, 2], cells);
        let ing = ingest_tensor(&t);
        assert_eq!(materialize_tensor(&ing), t, "2×2×2×2 restores exactly");
    }

    #[test]
    fn transposed_presentations_are_distinct_content() {
        // The same values in two orientations: (W,H) = (2,3) vs (H,W) =
        // (3,2). The hash is deterministic and content-addressed, so the
        // orientation is part of the content: the two layouts are two
        // different presentations of the same measured object.
        let cells: Vec<Vec<u8>> = (0..6).map(|i| format!("v{i}").into_bytes()).collect();
        let a = Tensor::new(vec![2, 3], cells.clone());
        let mut transposed = Vec::with_capacity(6);
        for r in 0..3 {
            for c in 0..2 {
                transposed.push(cells[c * 3 + r].clone());
            }
        }
        let b = Tensor::new(vec![3, 2], transposed);

        let ia = ingest_tensor(&a);
        let ib = ingest_tensor(&b);

        // The 1x1 channel is the shared token set — orientation-free.
        assert_eq!(
            ia.channels[0].content_key(),
            ib.channels[0].content_key(),
            "G1 is the same token set in both orientations"
        );
        // The 2x2 channel and the projection are orientation-dependent.
        assert_ne!(
            ia.channels[1].content_key(),
            ib.channels[1].content_key(),
            "2x2 neighborhoods differ under transposition"
        );
        assert_ne!(ia.key, ib.key, "the projections are different content");

        // The G1 gate extracts the token set from either presentation.
        let g1 = ia.channels[0].union(&ib.channels[0]);
        assert_eq!(
            crate::api::gate(&g1, &ia.projection).content_key(),
            ia.channels[0].content_key(),
            "gate(G1, H_a) == G1"
        );
        assert_eq!(
            crate::api::gate(&g1, &ib.projection).content_key(),
            ib.channels[0].content_key(),
            "gate(G1, H_b) == G1"
        );
    }

    #[test]
    fn tensor_channels_and_key_are_populated() {
        let t = Tensor::new(
            vec![2, 2],
            vec![tok("x"), tok("y"), tok("y"), tok("z")],
        );
        let ing = ingest_tensor(&t);
        assert!(ing.key.starts_with("h:"));
        for ch in &ing.channels {
            assert!(ch.popcount() > 0, "every channel is populated");
        }
        assert_eq!(
            materialize_tensor_no_order(&ing),
            BTreeSet::from([tok("x"), tok("y"), tok("z")])
        );
    }
}
