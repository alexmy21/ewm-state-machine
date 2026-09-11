//! Grid morphisms — `conv(n, dim=2)`.
//!
//! A [`Grid`] is a 2D token collection (row-major). Ingest slides `n × n`
//! windows (`n = 1..=4`) with a **full PAD border** of `GRID_BORDER` cells
//! on every side, so the count-constrained ordered walk can always treat the
//! candidate cell as the window's bottom-right corner (the 2D analog of the
//! 1D n-gram chain).
//!
//! Channels: `1×1` (shared G1, seed 0), `2×2` (seed 4), `3×3` (seed 5),
//! `4×4` (seed 6, the order-only side channel). The set path is LUT-first
//! over the first three channels; the order path is the count-constrained
//! row-major walk with joint `2×2 ∧ 3×3 ∧ 4×4` checks.

use std::collections::{BTreeMap, BTreeSet};

use ::hllset_lut::LutIndex;
use hllset_contracts::BitAddress;
use hllset_core::HLLSet;

use crate::api::{join, PAD};
use crate::conv::{channel_name, seed, ConvSpec};
use crate::hllset_lut::HllsetLut;
use crate::materialize::materialize as materialize_lut_first;
use crate::tf::TfTable;

/// Largest window order in the grid regime (the 4×4 order side channel).
pub const GRID_MAX_N: usize = 4;
/// PAD border on every side of the grid (full border, so the candidate is
/// always the bottom-right cell of its window).
pub const GRID_BORDER: usize = GRID_MAX_N - 1;

/// The channel seed of an `n × n` grid window.
fn gseed(n: usize) -> u64 {
    seed(ConvSpec::new(n as u8, 2))
}

/// A 2D token collection in row-major order.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Grid {
    pub width: usize,
    pub height: usize,
    pub cells: Vec<Vec<u8>>,
}

impl Grid {
    pub fn new(width: usize, height: usize, cells: Vec<Vec<u8>>) -> Self {
        assert_eq!(
            cells.len(),
            width * height,
            "grid cells must be width * height"
        );
        Self {
            width,
            height,
            cells,
        }
    }

    pub fn cell(&self, r: usize, c: usize) -> &[u8] {
        &self.cells[r * self.width + c]
    }

    pub fn len(&self) -> usize {
        self.cells.len()
    }

    pub fn is_empty(&self) -> bool {
        self.cells.is_empty()
    }
}

/// The result of 2D convolution ingestion: four channels (`1×1` … `4×4`),
/// four LUTs, TF, and the preservation side effects.
#[derive(Clone, Debug)]
pub struct GridIngested {
    /// Channels indexed by `n - 1` (1×1, 2×2, 3×3, 4×4).
    pub channels: [HLLSet; GRID_MAX_N],
    /// LUTs indexed by `n - 1`; the fiber of a bit holds the window's
    /// top-left cell token (the pointed-to token).
    pub luts: [LutIndex; GRID_MAX_N],
    /// Monotonic term-frequency table over the original cells.
    pub tf: TfTable,
    pub width: usize,
    pub height: usize,
    pub pad: Vec<u8>,
    /// `1×1 ∪ 2×2 ∪ 3×3` — the token presentation (the 4×4 is order-only).
    pub projection: HLLSet,
    /// SHA1 of the projection (`h:<sha1>`).
    pub key: String,
    /// SHA1 keys of the four channels.
    pub keys: [String; GRID_MAX_N],
    /// Named channel HLLSets registered and touch-counted (preservation).
    pub hllset_lut: HllsetLut,
}

impl Default for GridIngested {
    fn default() -> Self {
        Self::new()
    }
}

impl GridIngested {
    pub fn new() -> Self {
        let empty = HLLSet::new();
        let key = empty.content_key();
        Self {
            channels: std::array::from_fn(|_| HLLSet::new()),
            luts: std::array::from_fn(|_| LutIndex::default()),
            tf: TfTable::new(),
            width: 0,
            height: 0,
            pad: PAD.to_vec(),
            projection: empty,
            key,
            keys: std::array::from_fn(|_| String::new()),
            hllset_lut: HllsetLut::new(),
        }
    }
}

/// Ingest a grid with the default pad (`<PAD>`).
pub fn ingest_grid(grid: &Grid) -> GridIngested {
    ingest_grid_with_pad(grid, PAD)
}

/// Ingest a grid with a custom pad.
pub fn ingest_grid_with_pad(grid: &Grid, pad: &[u8]) -> GridIngested {
    let mut out = GridIngested::new();
    out.width = grid.width;
    out.height = grid.height;
    out.pad = pad.to_vec();
    if grid.is_empty() {
        return out;
    }

    let (h, w) = (grid.height, grid.width);
    let ph = h + 2 * GRID_BORDER; // padded height
    let pw = w + 2 * GRID_BORDER; // padded width

    for n in 1..=GRID_MAX_N {
        let ch = n - 1;
        let seed_n = gseed(n);
        for i in 0..=(ph - n) {
            for j in 0..=(pw - n) {
                let mut parts: Vec<&[u8]> = Vec::with_capacity(n * n);
                for di in 0..n {
                    for dj in 0..n {
                        let (pi, pj) = (i + di, j + dj);
                        let val: &[u8] = if pi >= GRID_BORDER
                            && pi < GRID_BORDER + h
                            && pj >= GRID_BORDER
                            && pj < GRID_BORDER + w
                        {
                            &grid.cells[(pi - GRID_BORDER) * w + (pj - GRID_BORDER)]
                        } else {
                            pad
                        };
                        parts.push(val);
                    }
                }
                let ngram = join(&parts);
                let addr = BitAddress::of_token_seeded(&ngram, seed_n);
                out.channels[ch].add_bit(addr.bit());
                // Real top-left anchors register their token in the LUT.
                if i >= GRID_BORDER
                    && i < GRID_BORDER + h
                    && j >= GRID_BORDER
                    && j < GRID_BORDER + w
                {
                    let token =
                        grid.cells[(i - GRID_BORDER) * w + (j - GRID_BORDER)].clone();
                    out.luts[ch].insert_token_at(token, addr.bit());
                }
            }
        }
    }

    for cell in &grid.cells {
        out.tf.increment(cell);
    }

    out.projection = out.channels[0..3]
        .iter()
        .fold(HLLSet::new(), |acc, s| acc.union(s));
    out.key = out.projection.content_key();
    for (n, ch) in out.channels.iter().enumerate() {
        out.keys[n] = ch.content_key();
        let name = channel_name(n as u8 + 1, 2);
        out.hllset_lut.register_named(&name, &out.keys[n]);
        out.hllset_lut.touch_named(&name, &out.keys[n]);
    }
    out
}

/// The plain-set restoration: LUT-first over `1×1`, `2×2`, `3×3` (the 4×4 is
/// order-only). Collided bits keep every candidate.
pub fn materialize_grid_no_order(ing: &GridIngested) -> BTreeSet<Vec<u8>> {
    let pairs: Vec<(&HLLSet, &LutIndex)> = (0..3)
        .map(|i| (&ing.channels[i], &ing.luts[i]))
        .collect();
    materialize_lut_first(&pairs)
}

/// Restore the grid in its original row-major order (greedy walk).
pub fn materialize_grid(ing: &GridIngested) -> Grid {
    materialize_grid_with(ing, 1)
}

/// Restore the grid with beam search of the given width.
pub fn materialize_grid_beam(ing: &GridIngested, width: usize) -> Grid {
    materialize_grid_with(ing, width.max(1))
}

fn materialize_grid_with(ing: &GridIngested, beam: usize) -> Grid {
    let tokens = materialize_grid_no_order(ing);
    let cells = grid_order_tokens(&tokens, ing, beam);
    Grid {
        width: ing.width,
        height: ing.height,
        cells,
    }
}

/// The ordered restoration: count-constrained, TF-scored, joint-window
/// decoding over the row-major grid.
fn grid_order_tokens(
    tokens: &BTreeSet<Vec<u8>>,
    ing: &GridIngested,
    beam: usize,
) -> Vec<Vec<u8>> {
    let n = ing.width * ing.height;
    if tokens.is_empty() || n == 0 {
        return Vec::new();
    }

    // The count constraint: TF counts observations of every cell token.
    let remaining: BTreeMap<Vec<u8>, usize> = tokens
        .iter()
        .map(|t| (t.clone(), ing.tf.count(t) as usize))
        .filter(|(_, c)| *c > 0)
        .collect();

    if beam > 1 {
        if let Some(path) = grid_beam_search(tokens, ing, beam, &remaining) {
            return path;
        }
    }

    let mut path = Vec::new();
    let mut rem = remaining;
    let mut budget = 20_000usize;
    if grid_walk(tokens, ing, &mut path, &mut rem, &mut budget) {
        return path;
    }

    // Fallback: cannot reconstruct — return the unordered set.
    tokens.iter().cloned().collect()
}

/// `true` when the `n × n` channel contains the atom of `bytes`.
fn grid_channel_contains(ing: &GridIngested, n: usize, bytes: &[u8]) -> bool {
    let addr = BitAddress::of_token_seeded(bytes, gseed(n));
    ing.channels[n - 1].has_bit(addr.reg(), addr.tz())
}

/// Build the bytes of the `n × n` window anchored at padded `(ai, aj)`.
///
/// All cells resolve from `path`/PAD; `candidate = Some((r, c, x))`
/// overrides the bottom-right cell (the candidate during the walk).
fn grid_window(
    ing: &GridIngested,
    n: usize,
    ai: usize,
    aj: usize,
    path: &[Vec<u8>],
    candidate: Option<(usize, usize, &[u8])>,
) -> Vec<u8> {
    let (w, h) = (ing.width, ing.height);
    let pad = ing.pad.as_slice();
    let mut parts: Vec<&[u8]> = Vec::with_capacity(n * n);
    for di in 0..n {
        for dj in 0..n {
            let (pi, pj) = (ai + di, aj + dj);
            let val: &[u8] = if pi >= GRID_BORDER
                && pi < GRID_BORDER + h
                && pj >= GRID_BORDER
                && pj < GRID_BORDER + w
            {
                let (rr, cc) = (pi - GRID_BORDER, pj - GRID_BORDER);
                match candidate {
                    Some((cr, ccc, x)) if rr == cr && cc == ccc => x,
                    _ => &path[rr * w + cc],
                }
            } else {
                pad
            };
            parts.push(val);
        }
    }
    join(&parts)
}

/// `true` when candidate `x` satisfies every joint window check at cell
/// `(r, c)`.
fn grid_successor_ok(
    ing: &GridIngested,
    path: &[Vec<u8>],
    r: usize,
    c: usize,
    x: &[u8],
) -> bool {
    (2..=GRID_MAX_N).all(|n| {
        // The window whose bottom-right is the candidate: padded anchor
        // (r + GRID_MAX_N - n, c + GRID_MAX_N - n).
        let ai = r + GRID_MAX_N - n;
        let aj = c + GRID_MAX_N - n;
        let bytes = grid_window(ing, n, ai, aj, path, Some((r, c, x)));
        grid_channel_contains(ing, n, &bytes)
    })
}

/// `true` when the completed grid ends at the trailing PAD border.
fn grid_termination_ok(ing: &GridIngested, path: &[Vec<u8>]) -> bool {
    let (w, h) = (ing.width, ing.height);
    (2..=GRID_MAX_N).all(|n| {
        // The window whose bottom-right is the first PAD cell after the last
        // real cell: padded anchor (h + 3 - n, w + 4 - n).
        let ai = h + GRID_MAX_N - 1 - n;
        let aj = w + GRID_MAX_N - n;
        let bytes = grid_window(ing, n, ai, aj, path, None);
        grid_channel_contains(ing, n, &bytes)
    })
}

/// Depth-first count-constrained walk over the row-major grid.
fn grid_walk(
    tokens: &BTreeSet<Vec<u8>>,
    ing: &GridIngested,
    path: &mut Vec<Vec<u8>>,
    remaining: &mut BTreeMap<Vec<u8>, usize>,
    budget: &mut usize,
) -> bool {
    if *budget == 0 {
        return false;
    }
    *budget -= 1;

    let total = ing.width * ing.height;
    if path.len() == total {
        return grid_termination_ok(ing, path);
    }

    let idx = path.len();
    let r = idx / ing.width;
    let c = idx % ing.width;

    let mut nexts: Vec<Vec<u8>> = tokens
        .iter()
        .filter(|t| remaining.get(*t).copied().unwrap_or(0) > 0)
        .filter(|t| grid_successor_ok(ing, path, r, c, t))
        .cloned()
        .collect();
    nexts.sort_by(|a, b| ing.tf.count(b).cmp(&ing.tf.count(a)).then(a.cmp(b)));

    for next in nexts {
        if let Some(n) = remaining.get_mut(&next) {
            *n -= 1;
        }
        path.push(next.clone());
        if grid_walk(tokens, ing, path, remaining, budget) {
            return true;
        }
        path.pop();
        if let Some(n) = remaining.get_mut(&next) {
            *n += 1;
        }
    }
    false
}

/// One grid beam-search hypothesis.
struct GridBeamState {
    path: Vec<Vec<u8>>,
    score: u64,
    remaining: BTreeMap<Vec<u8>, usize>,
}

fn sort_grid_beam(beam: &mut [GridBeamState]) {
    beam.sort_by(|a, b| b.score.cmp(&a.score).then(a.path.cmp(&b.path)));
}

/// Beam search over the row-major grid: keep the top-`width` partial grids by
/// cumulative TF score, expand one cell at a time under the count
/// constraint, and return the highest-scoring completed grid.
fn grid_beam_search(
    tokens: &BTreeSet<Vec<u8>>,
    ing: &GridIngested,
    width: usize,
    initial_remaining: &BTreeMap<Vec<u8>, usize>,
) -> Option<Vec<Vec<u8>>> {
    let total = ing.width * ing.height;
    let width = width.max(1);

    let mut beam: Vec<GridBeamState> = vec![GridBeamState {
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

        let mut next: Vec<GridBeamState> = Vec::new();
        for st in &beam {
            if st.path.len() == total {
                if grid_termination_ok(ing, &st.path) {
                    completed.push((st.path.clone(), st.score));
                }
                continue;
            }
            let idx = st.path.len();
            let r = idx / ing.width;
            let c = idx % ing.width;
            for cand in tokens {
                if st.remaining.get(cand).copied().unwrap_or(0) == 0 {
                    continue;
                }
                if !grid_successor_ok(ing, &st.path, r, c, cand) {
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
                next.push(GridBeamState {
                    path: new_path,
                    score: st.score + ing.tf.count(cand),
                    remaining: new_rem,
                });
            }
        }

        if next.is_empty() {
            break;
        }
        sort_grid_beam(&mut next);
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
    fn grid_roundtrip_2x2_distinct() {
        let grid = Grid::new(
            2,
            2,
            vec![tok("a"), tok("b"), tok("c"), tok("d")],
        );
        let ing = ingest_grid(&grid);
        let restored = materialize_grid(&ing);
        assert_eq!(restored, grid, "2×2 grid restores exactly");
        let set = materialize_grid_no_order(&ing);
        assert_eq!(
            set,
            BTreeSet::from([tok("a"), tok("b"), tok("c"), tok("d")])
        );
    }

    #[test]
    fn grid_roundtrip_3x3_with_duplicates() {
        let cells: Vec<Vec<u8>> = ["a", "b", "a", "c", "a", "c", "b", "c", "b"]
            .iter()
            .map(|s| tok(s))
            .collect();
        let grid = Grid::new(3, 3, cells);
        let ing = ingest_grid(&grid);
        assert_eq!(materialize_grid(&ing), grid, "greedy restores duplicates");
        assert_eq!(materialize_grid_beam(&ing, 2), grid, "beam-2 agrees");
    }

    #[test]
    fn grid_roundtrip_16x16_pseudo_frame() {
        // The image-application shape: a 16×16 grid of tid tokens with
        // repeats, exactly like the scene frames.
        let cells: Vec<Vec<u8>> = (0..256)
            .map(|i| {
                let r = i / 16;
                let c = i % 16;
                format!("tid{}", (r * 37 + c * 7 + (r + c) / 3) % 60).into_bytes()
            })
            .collect();
        let grid = Grid::new(16, 16, cells);
        let ing = ingest_grid(&grid);
        let restored = materialize_grid(&ing);
        assert_eq!(restored, grid, "16×16 pseudo-frame restores exactly");
        let restored_beam = materialize_grid_beam(&ing, 3);
        assert_eq!(restored_beam, grid, "beam-3 agrees");
    }

    #[test]
    fn grid_channels_and_key_are_populated() {
        let grid = Grid::new(2, 2, vec![tok("x"), tok("y"), tok("y"), tok("z")]);
        let ing = ingest_grid(&grid);
        assert!(ing.key.starts_with("h:"));
        assert!(ing.channels[0].popcount() > 0, "1×1 channel");
        assert!(ing.channels[1].popcount() > 0, "2×2 channel");
        assert!(ing.channels[2].popcount() > 0, "3×3 channel");
        assert!(ing.channels[3].popcount() > 0, "4×4 channel");
        // The set restoration keeps every candidate of the duplicated token.
        assert_eq!(
            materialize_grid_no_order(&ing),
            BTreeSet::from([tok("x"), tok("y"), tok("z")])
        );
    }
}
