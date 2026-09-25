//! `ewm-scene` — the direct morphisms path for the LLM ↔ HLLSet side-car.
//!
//! The scene-analytics application (rebuild of the vLLM reference) calls this
//! helper instead of the Lua DSL: frames of `tid{n}` tokens go in as JSONL,
//! HLLSet statistics come out as JSON. Everything here is the default
//! application-level interface — [`ewm_app::ingest`], [`ewm_app::materialize`],
//! and the HLLSet lattice operations.

use ewm_app::{
    ingest, ingest_grid, ingest_tensor, materialize, materialize_beam, materialize_grid,
    materialize_grid_beam, materialize_grid_no_order, materialize_no_order, materialize_tensor,
    materialize_tensor_beam, materialize_tensor_no_order, Grid, Tensor,
};
use hllset_morphisms::materialize::materialize as materialize_lut_first;
use hllset_morphisms::Ingest;
use hllset_core::HLLSet;

/// One frame: `id` + the token collection (in patch order).
#[derive(Clone, Debug)]
pub struct Frame {
    pub id: u64,
    pub tokens: Vec<String>,
}

/// Ingest every frame; keep the projection HLLSet (`G1 ∪ G2 ∪ G3`), its key,
/// and the G1 sketch (the channel shared verbatim between n-gram and n-seed
/// schemes — used for G1-scoped comparisons).
#[derive(Clone, Debug)]
pub struct FrameSet {
    pub frames: Vec<Frame>,
    pub hllsets: Vec<HLLSet>,
    pub keys: Vec<String>,
    pub pops: Vec<u64>,
    pub g1s: Vec<HLLSet>,
}

impl FrameSet {
    pub fn from_frames(frames: Vec<Frame>) -> Self {
        let mut hllsets = Vec::with_capacity(frames.len());
        let mut keys = Vec::with_capacity(frames.len());
        let mut pops = Vec::with_capacity(frames.len());
        let mut g1s = Vec::with_capacity(frames.len());
        for frame in &frames {
            // The lattice line is n-seed: the frame HLLSet is the set of the
            // frame's token atoms (projection G1 ∪ G2 ∪ G3, no PAD). The
            // n-gram api is reserved for order restoration in `restore`.
            let mut ig = Ingest::new();
            ig.ingest_tokens(frame.tokens.iter().map(|t| t.as_bytes()));
            let projection = ig.projection();
            keys.push(ig.key());
            pops.push(projection.popcount());
            hllsets.push(projection);
            g1s.push(ig.hllset(0).clone());
        }
        Self {
            frames,
            hllsets,
            keys,
            pops,
            g1s,
        }
    }

    pub fn len(&self) -> usize {
        self.frames.len()
    }

    pub fn is_empty(&self) -> bool {
        self.frames.is_empty()
    }

    /// BSSτ inclusion: `|A ∩ B| / |B|` (how much of B is already in A).
    pub fn bss(&self, i: usize, j: usize) -> f64 {
        let inter = self.hllsets[i].intersection(&self.hllsets[j]).popcount() as f64;
        let denom = self.hllsets[j].popcount() as f64;
        if denom == 0.0 {
            1.0
        } else {
            inter / denom
        }
    }

    /// Jaccard similarity: `|A ∩ B| / |A ∪ B|`.
    pub fn jaccard(&self, i: usize, j: usize) -> f64 {
        let inter = self.hllsets[i].intersection(&self.hllsets[j]).popcount() as f64;
        let union = self.hllsets[i].union(&self.hllsets[j]).popcount() as f64;
        if union == 0.0 {
            1.0
        } else {
            inter / union
        }
    }

    /// Consecutive BSSτ and Jaccard over `(i-1, i)` pairs.
    pub fn bss_series(&self) -> (Vec<f64>, Vec<f64>) {
        let n = self.len();
        let mut tau = Vec::with_capacity(n.saturating_sub(1));
        let mut jac = Vec::with_capacity(n.saturating_sub(1));
        for i in 1..n {
            tau.push(self.bss(i - 1, i));
            jac.push(self.jaccard(i - 1, i));
        }
        (tau, jac)
    }

    /// Trailing-window unions in HLLSet space and BSSτ between consecutive
    /// windows. Returns `(t0, fast, slow)` aligned on `t = t0..=N` (1-based),
    /// where `t0 = long_n + 1`; `fast`/`slow` use the short/long windows.
    pub fn moving_averages(&self, short_n: usize, long_n: usize) -> (Vec<usize>, Vec<f64>, Vec<f64>) {
        let n = self.len();
        assert!(short_n >= 1 && long_n >= short_n && long_n <= n);

        let window = |t: usize, w: usize| -> HLLSet {
            let mut acc = HLLSet::new();
            for j in (t - w)..t {
                acc = acc.union(&self.hllsets[j]);
            }
            acc
        };

        let t0 = long_n + 1;
        let mut t_axis = Vec::new();
        let mut fast = Vec::new();
        let mut slow = Vec::new();
        for t in t0..=n {
            let s_prev = window(t - 1, short_n);
            let s_cur = window(t, short_n);
            let l_prev = window(t - 1, long_n);
            let l_cur = window(t, long_n);
            t_axis.push(t);
            fast.push(bss(&s_prev, &s_cur));
            slow.push(bss(&l_prev, &l_cur));
        }
        (t_axis, fast, slow)
    }

    /// Per-pair D/R/N decomposition and the three Noether indicators.
    pub fn noether(&self) -> NoetherOut {
        noether_series(&self.hllsets)
    }
}

/// Per-pair D/R/N decomposition over any HLLSet series (frames, unions,
/// perceptron states — the same lattice math at every level of the pyramid).
pub fn noether_series(hllsets: &[HLLSet]) -> NoetherOut {
    let n = hllsets.len();
    let mut dp = Vec::with_capacity(n.saturating_sub(1));
    let mut rp = Vec::with_capacity(n.saturating_sub(1));
    let mut np = Vec::with_capacity(n.saturating_sub(1));
    let mut ind1 = Vec::with_capacity(n.saturating_sub(1));
    let mut ind3 = Vec::with_capacity(n.saturating_sub(1));
    let mut ind2 = Vec::with_capacity(n.saturating_sub(3));

    let mut r_prev: Option<HLLSet> = None;
    for i in 0..n.saturating_sub(1) {
        let d = hllsets[i].difference(&hllsets[i + 1]);
        let r = hllsets[i].intersection(&hllsets[i + 1]);
        let nn = hllsets[i + 1].difference(&hllsets[i]);
        let (dpc, rpc, npc) = (d.popcount(), r.popcount(), nn.popcount());
        dp.push(dpc);
        rp.push(rpc);
        np.push(npc);

        let rn = (r.union(&nn)).popcount() as f64;
        let rd = (r.union(&d)).popcount() as f64;
        ind1.push(if rn == 0.0 { 0.0 } else { dpc as f64 / rn });
        ind3.push(if rd == 0.0 { 0.0 } else { npc as f64 / rd });

        if let Some(r_prev) = &r_prev {
            ind2.push(bss(r_prev, &r));
        }
        r_prev = Some(r);
    }

    NoetherOut {
        dp,
        rp,
        np,
        ind1,
        ind2,
        ind3,
    }
}

/// BSSτ between two HLLSets: `|A ∩ B| / |B|`.
pub fn bss(a: &HLLSet, b: &HLLSet) -> f64 {
    let denom = b.popcount() as f64;
    if denom == 0.0 {
        1.0
    } else {
        a.intersection(b).popcount() as f64 / denom
    }
}

/// The Noether decomposition of one frame pair: D/R/N popcounts + indicators.
#[derive(Clone, Debug, Default)]
pub struct NoetherOut {
    pub dp: Vec<u64>,
    pub rp: Vec<u64>,
    pub np: Vec<u64>,
    pub ind1: Vec<f64>,
    pub ind2: Vec<f64>,
    pub ind3: Vec<f64>,
}

/// One frame's measurement against the Boolean-ring basis **before** the frame
/// was inserted — the soft key (BSS weights over the basis), the hard key
/// (GF(2) coordinates, span members only), and the residual (linear novelty).
///
/// This is the Phase-1 trajectory record of the side-car: a per-frame point in
/// soft-key space plus its step length against the previous frame, both
/// measured in the *current* basis so consecutive steps share one coordinate
/// system (docs/BOOLRING.md — coordinates are comparable inside a window).
#[derive(Clone, Debug, Default)]
pub struct SidecarFrame {
    /// `w_i = |S(t) ∩ B_i| / |B_i|` for each basis element `B_i` of the ring
    /// basis *before* this frame arrived. Any HLLSet can be measured this way.
    pub soft: Vec<f64>,
    /// GF(2) coordinates `(1,0,1,…)` — `Some` only when the frame is exactly
    /// in the span of the current basis (the hard key; exact structural match).
    pub hard: Option<Vec<bool>>,
    /// Popcounts `|B_i|` of the basis elements the keys refer to.
    pub basis_pop: Vec<u64>,
    /// Popcount of the residual against the current span — linear novelty.
    pub residual: u64,
    /// `true` when the residual is empty (the frame is in the span).
    pub in_span: bool,
    /// Span dimension *after* inserting this frame.
    pub dim: usize,
    /// Number of existing basis elements re-pivoted by this frame's insertion
    /// (the rotation component of the basis change; zero when in-span).
    pub rotation_count: u64,
    /// Total Hamming change of the old basis: `rotation_count * residual`.
    pub rotation_mass: u64,
    /// L2 distance between this frame's soft key and the previous frame's soft
    /// key, both recomputed against the current basis. `0.0` for the first
    /// frame.
    pub step: f64,
    /// Time travel: the soft key of this frame projected into the **first**
    /// recorded basis (the initial interpretation; docs/BASIS_FRAMES.md).
    /// Same coordinate system for every frame — how the present lines up
    /// with the past's directions.
    pub soft_first: Vec<f64>,
    /// Time travel: `|frame \ cover(F_0)|` — how many bits of this frame the
    /// first basis cannot even see (the measurable error of the travel).
    pub spill_first: u64,
}

/// One interpretation (basis frame) of the ring over time: the basis
/// content at a generation change, stamped with the monotonic generation
/// (docs/BASIS_FRAMES.md). The first entry is the initial interpretation;
/// every later entry is a structural event.
#[derive(Clone, Debug, Default, serde::Serialize)]
pub struct BasisFrame {
    pub generation: u64,
    pub dimension: usize,
    pub cover_pop: u64,
}

/// The Phase-1 side-car trajectory: per-frame soft/hard keys against the
/// growing Boolean-ring basis, step lengths, and the jump detector over the
/// step series (threshold = in-scene mean + 3σ).
#[derive(Clone, Debug, Default)]
pub struct SidecarOut {
    pub frames: Vec<SidecarFrame>,
    /// 1-based frame indices flagged by the jump detector.
    pub jumps: Vec<u64>,
    /// The step-length threshold the detector used.
    pub threshold: f64,
    /// The basis history (one entry per generation change) — the
    /// interpretation timeline the time-travel fields refer to.
    pub basis_history: Vec<BasisFrame>,
}

impl FrameSet {
    /// The side-car trajectory over the Boolean ring, wired exactly like the
    /// [UM] cache: a [`ewm_boolring::BoolWindow`] over the original turn
    /// HLLSets with [`ewm_app::RING_CAPACITY`] (64).
    ///
    /// For each frame, before it enters the window, measure the soft key (BSS
    /// weights over the window basis), the hard key (coordinates when the
    /// frame is in the span) and the residual (linear novelty). Then push the
    /// frame — eviction slides the window and recomputes the basis. Step
    /// lengths compare consecutive soft keys in the same (current) basis.
    pub fn sidecar(&self) -> SidecarOut {
        self.sidecar_with_cap(ewm_app::RING_CAPACITY)
    }

    /// The side-car trajectory with a configurable window capacity. A capacity
    /// at least the frame count is a growing basis (context so far); a smaller
    /// capacity is a sliding scene-bounded window whose residual series is
    /// comparable across the clip (the basis dimension stays bounded).
    pub fn sidecar_with_cap(&self, cap: usize) -> SidecarOut {
        self.sidecar_with_cap_freeze(cap, None)
    }

    /// The side-car trajectory with a configurable window capacity and an
    /// optional **basis freeze**: after `freeze` frames the basis is no longer
    /// updated — every later frame is only *evaluated* against the frozen
    /// basis (soft key, hard key, residual, step). This is the stable-
    /// coordinate mode: all post-freeze soft keys live in the same
    /// `k`-dimensional space, so they form a fixed-dim time series ready for
    /// DFT over `t` (the ring's ingestion order *is* the time axis).
    pub fn sidecar_with_cap_freeze(&self, cap: usize, freeze: Option<usize>) -> SidecarOut {
        sidecar_series(&self.hllsets, cap, freeze)
    }
}

/// The side-car trajectory over any HLLSet series — frames at the bottom of
/// the pyramid, perceptron states one level up, unions at the top. Same
/// Boolean-ring algebra at every level (docs/ASSIGNMENT_QWENDRIVE.md Phase 2).
pub fn sidecar_series(
    hllsets: &[HLLSet],
    cap: usize,
    freeze: Option<usize>,
) -> SidecarOut {
    let mut window = ewm_boolring::BoolWindow::new(cap.max(1));
    let mut frames = Vec::with_capacity(hllsets.len());
    let mut prev_set: Option<HLLSet> = None;
    let freeze = freeze.map(|n| n.max(1));

    // Time travel: capture the first interpretation (basis + cover) and the
    // full basis history (one entry per generation change).
    let mut first_basis: Option<ewm_boolring::BoolBasis> = None;
    let mut first_cover: Option<HLLSet> = None;
    let mut basis_history: Vec<BasisFrame> = Vec::new();
    let mut last_gen: u64 = 0;

    for (idx, set) in hllsets.iter().enumerate() {
        let frozen = freeze.is_some_and(|n| idx >= n);
        let basis = window.basis();
        let soft: Vec<f64> = basis
            .basis
            .iter()
            .map(|b| bss(set, b))
            .collect();
        let basis_pop: Vec<u64> = basis.basis.iter().map(|b| b.popcount()).collect();
        let hard = basis.coordinates(set);
        let residual_set = basis.residual(set);
        let residual = residual_set.popcount();
        let in_span = residual == 0;

        // Step length: both frames measured against the *current* basis
        // (which, after the freeze, is the same basis for every step).
        let step = match &prev_set {
            Some(prev) => {
                let prev_soft: Vec<f64> = basis
                    .basis
                    .iter()
                    .map(|b| bss(prev, b))
                    .collect();
                l2_distance(&prev_soft, &soft)
            }
            None => 0.0,
        };

        // Push after measuring (unless frozen), so the record describes
        // the novelty of the incoming frame against the context so far.
        let (rotation_count, rotation_mass) = if frozen {
            (0, 0) // an evaluation never changes the basis
        } else {
            let ring_stats = window.push(set);
            if first_basis.is_none() {
                first_basis = Some(window.basis().clone());
                first_cover = Some(
                    window
                        .basis()
                        .basis
                        .iter()
                        .fold(HLLSet::new(), |acc, b| acc.union(b)),
                );
            }
            let gen = window.generation();
            if basis_history.is_empty() || gen != last_gen {
                basis_history.push(BasisFrame {
                    generation: gen,
                    dimension: window.dimension(),
                    cover_pop: window
                        .basis()
                        .basis
                        .iter()
                        .fold(0u64, |acc, b| acc + b.popcount()),
                });
                last_gen = gen;
            }
            (ring_stats.rotation_count, ring_stats.rotation_mass)
        };
        frames.push(SidecarFrame {
            soft,
            hard,
            basis_pop,
            residual,
            in_span,
            dim: window.dimension(),
            rotation_count,
            rotation_mass,
            step,
            soft_first: Vec::new(),
            spill_first: 0,
        });
        prev_set = Some(set.clone());
    }

    // Time travel: project every frame into the first basis — the same
    // coordinate system, so the whole trajectory is comparable to the
    // initial interpretation. Spill = |frame \ cover(F_0)|.
    if let (Some(fb), Some(cover)) = (&first_basis, &first_cover) {
        for (i, frame) in frames.iter_mut().enumerate() {
            frame.soft_first = fb
                .basis
                .iter()
                .map(|b| bss(&hllsets[i], b))
                .collect();
            frame.spill_first = hllsets[i].difference(cover).popcount();
        }
    }

    // Jump detector over the step series: mean + 3σ (σ = standard
    // deviation, population). The first frame's zero step is excluded from
    // the baseline so it cannot pull the threshold down.
    let steps: Vec<f64> = frames.iter().map(|f| f.step).collect();
    let baseline: Vec<f64> = steps.iter().skip(1).copied().collect();
    let mean = mean(&baseline);
    let std = std_dev(&baseline, mean);
    let threshold = mean + 3.0 * std;
    let jumps: Vec<u64> = frames
        .iter()
        .enumerate()
        .filter_map(|(i, f)| if f.step > threshold { Some((i + 1) as u64) } else { None })
        .collect();

    SidecarOut {
        frames,
        jumps,
        threshold,
        basis_history,
    }
}

/// One perceptron of a pyramid frame: a name and its token collection.
#[derive(Clone, Debug)]
pub struct PerceptronTokens {
    pub name: String,
    pub tokens: Vec<String>,
}

/// One pyramid frame: `m` perceptron token collections at time `t`.
#[derive(Clone, Debug)]
pub struct PyramidFrame {
    pub id: u64,
    pub perceptrons: Vec<PerceptronTokens>,
}

/// One pyramid frame, ingested: the per-perceptron HLLSets (decomposition b)
/// and the union top state `u-HLLSet(t)`.
#[derive(Clone, Debug)]
pub struct PyramidFrameState {
    pub id: u64,
    pub names: Vec<String>,
    /// One HLLSet per perceptron — the joined decomposition
    /// `u-HLLSet(t) = <p1-HLLSet(t), …, pm-HLLSet(t)>`.
    pub hllsets: Vec<HLLSet>,
    pub keys: Vec<String>,
    pub pops: Vec<u64>,
    /// `⋃_i p_i-HLLSet(t)` — the top perceptron state.
    pub union_hll: HLLSet,
    pub union_key: String,
    pub union_pop: u64,
}

/// The pyramid analysis: the top-perceptron union stream with its three
/// decompositions — (a) D/R/N over consecutive unions, (b) the joined
/// per-perceptron components, (c) the Boolean-ring basis of the union stream.
#[derive(Clone, Debug)]
pub struct PyramidOut {
    /// Perceptron names in the order they appear in every frame.
    pub names: Vec<String>,
    pub frames: Vec<PyramidFrameState>,
    /// (a) D/R/N decomposition of `u-HLLSet(t)` vs `u-HLLSet(t-1)`.
    pub drn: NoetherOut,
    /// (c) u-ring basis decomposition over the union stream.
    pub ring: SidecarOut,
}

impl PyramidFrame {
    /// Ingest one pyramid frame: per-perceptron projections + the union.
    pub fn ingest(&self) -> PyramidFrameState {
        let mut hllsets = Vec::with_capacity(self.perceptrons.len());
        let mut keys = Vec::with_capacity(self.perceptrons.len());
        let mut pops = Vec::with_capacity(self.perceptrons.len());
        let mut names = Vec::with_capacity(self.perceptrons.len());
        for p in &self.perceptrons {
            let mut ig = Ingest::new();
            ig.ingest_tokens(p.tokens.iter().map(|t| t.as_bytes()));
            keys.push(ig.key());
            pops.push(ig.projection().popcount());
            hllsets.push(ig.projection());
            names.push(p.name.clone());
        }
        let union_hll = hllsets
            .iter()
            .fold(HLLSet::new(), |acc, s| acc.union(s));
        let union_key = union_hll.content_key();
        let union_pop = union_hll.popcount();
        PyramidFrameState {
            id: self.id,
            names,
            hllsets,
            keys,
            pops,
            union_hll,
            union_key,
            union_pop,
        }
    }
}

/// Run the pyramid over a sequence of `m`-perceptron frames. The top
/// perceptron is the union of the component HLLSets; its stream is analysed
/// with the same lattice (D/R/N) and Boolean-ring algebra (soft/hard keys,
/// residual) as the single-perceptron side-car — one level up.
pub fn pyramid(frames: &[PyramidFrame], cap: usize, freeze: Option<usize>) -> PyramidOut {
    assert!(!frames.is_empty(), "pyramid needs at least one frame");
    let states: Vec<PyramidFrameState> = frames.iter().map(|f| f.ingest()).collect();
    let unions: Vec<HLLSet> = states.iter().map(|s| s.union_hll.clone()).collect();
    let drn = noether_series(&unions);
    let ring = sidecar_series(&unions, cap, freeze);
    PyramidOut {
        names: states[0].names.clone(),
        frames: states,
        drn,
        ring,
    }
}

/// One projection dimension: a named HLLSet — the `i`-th axis of a
/// decomposition frame (a perceptron state, a D/R/N set, a ring basis
/// element, any named HLLSet). The `g1` sketch is the dimension's G1
/// channel, used for G1-scoped (cross-scheme safe) BSS.
#[derive(Clone, Debug)]
pub struct Dimension {
    pub name: String,
    pub hll: HLLSet,
    pub g1: HLLSet,
}

impl Dimension {
    /// Build one dimension from a token collection (the same ingestion as the
    /// flat frame path: projection `G1 ∪ G2 ∪ G3`, no PAD).
    pub fn from_tokens(name: String, tokens: &[String]) -> Self {
        let mut ig = Ingest::new();
        ig.ingest_tokens(tokens.iter().map(|t| t.as_bytes()));
        Dimension {
            name,
            hll: ig.projection(),
            g1: ig.hllset(0).clone(),
        }
    }
}

/// One stream frame projected onto the frame dimensions.
#[derive(Clone, Debug)]
pub struct ProjectFrameOut {
    pub id: u64,
    /// Raw projections `|X ∩ D_i|`.
    pub intersections: Vec<u64>,
    /// The BSS similarity vector `(|X ∩ D_i| / |D_i|)` — the coordinates of
    /// `X` in the decomposition frame.
    pub bss: Vec<f64>,
    /// The G1-scoped BSS `(|X_G1 ∩ D_i_G1| / |D_i_G1|)` — the cross-scheme
    /// safe comparison (only 1-gram/seed-0 bits are matched).
    pub bss_g1: Vec<f64>,
}

/// The BSS trajectory of a stream over a decomposition frame: every
/// decomposition (D/R/N, joined perceptrons, ring basis) is an ordered
/// collection of HLLSet dimensions, and this is the coordinate map
/// `φ_D(X) = (|X∩D_1|/|D_1|, …, |X∩D_k|/|D_k|)` applied per time step.
#[derive(Clone, Debug)]
pub struct ProjectOut {
    pub names: Vec<String>,
    pub pops: Vec<u64>,
    pub frames: Vec<ProjectFrameOut>,
}

/// Project a stream of HLLSets onto a frame of named dimensions. `g1s` holds
/// each stream frame's G1 sketch (parallel to `hllsets`) for the G1-scoped
/// BSS — comparing an n-gram stream with n-seed dimensions on the full
/// projection would dilute identical-token similarity to ~1/3.
pub fn project(
    hllsets: &[HLLSet],
    g1s: &[HLLSet],
    ids: &[u64],
    dims: &[Dimension],
) -> ProjectOut {
    assert_eq!(hllsets.len(), ids.len(), "stream and ids must align");
    assert_eq!(hllsets.len(), g1s.len(), "stream and g1 sketches must align");
    let names: Vec<String> = dims.iter().map(|d| d.name.clone()).collect();
    let pops: Vec<u64> = dims.iter().map(|d| d.hll.popcount()).collect();
    let g1_pops: Vec<u64> = dims.iter().map(|d| d.g1.popcount()).collect();
    let frames = hllsets
        .iter()
        .zip(g1s)
        .zip(ids)
        .map(|((x, x_g1), id)| {
            let intersections: Vec<u64> = dims
                .iter()
                .map(|d| x.intersection(&d.hll).popcount())
                .collect();
            let bss: Vec<f64> = intersections
                .iter()
                .zip(&pops)
                .map(|(inter, pop)| {
                    if *pop == 0 {
                        1.0
                    } else {
                        *inter as f64 / *pop as f64
                    }
                })
                .collect();
            let bss_g1: Vec<f64> = dims
                .iter()
                .zip(&g1_pops)
                .map(|(d, pop)| {
                    let inter = x_g1.intersection(&d.g1).popcount() as f64;
                    if *pop == 0 {
                        1.0
                    } else {
                        inter / *pop as f64
                    }
                })
                .collect();
            ProjectFrameOut {
                id: *id,
                intersections,
                bss,
                bss_g1,
            }
        })
        .collect();
    ProjectOut {
        names,
        pops,
        frames,
    }
}

/// L2 distance between two soft-key vectors (padded mismatch cannot happen:
/// both are measured against the same basis here).
fn l2_distance(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len(), "soft keys must share one basis");
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y) * (x - y))
        .sum::<f64>()
        .sqrt()
}

fn mean(xs: &[f64]) -> f64 {
    if xs.is_empty() {
        0.0
    } else {
        xs.iter().sum::<f64>() / xs.len() as f64
    }
}

fn std_dev(xs: &[f64], mean: f64) -> f64 {
    if xs.len() < 2 {
        return 0.0;
    }
    let var = xs.iter().map(|x| (x - mean) * (x - mean)).sum::<f64>() / xs.len() as f64;
    var.sqrt()
}

/// Restore one frame's token collection: greedy order, plain set, beam.
pub fn restore(frame: &Frame) -> Restored {
    restore_with(frame, 2)
}

/// Restore with a configurable beam width (the beam path is bounded; the
/// greedy path carries its own node budget and falls back when exceeded).
pub fn restore_with(frame: &Frame, beam: usize) -> Restored {
    let ing = ingest(frame.tokens.iter());
    let to_str = |v: Vec<u8>| String::from_utf8_lossy(&v).into_owned();
    let beam_n = materialize_beam(&ing, beam.max(1))
        .into_iter()
        .map(to_str)
        .collect();
    let ordered = materialize(&ing).into_iter().map(to_str).collect();
    let set: Vec<String> = materialize_no_order(&ing).into_iter().map(to_str).collect();
    Restored {
        id: frame.id,
        ordered,
        set,
        beam2: beam_n,
    }
}

/// The restored presentations of one frame.
#[derive(Clone, Debug)]
pub struct Restored {
    pub id: u64,
    pub ordered: Vec<String>,
    pub set: Vec<String>,
    pub beam2: Vec<String>,
}

/// One grid frame (row-major cells) for the `conv(n, dim=2)` path.
#[derive(Clone, Debug)]
pub struct GridFrame {
    pub id: u64,
    pub width: usize,
    pub height: usize,
    pub tokens: Vec<String>,
}

/// The restored presentations of one grid frame.
#[derive(Clone, Debug)]
pub struct GridRestored {
    pub id: u64,
    pub width: usize,
    pub height: usize,
    pub ordered: Vec<String>,
    pub set: Vec<String>,
    pub beam2: Vec<String>,
}

/// Restore one grid frame through the 2D morphisms.
pub fn grid_restore(frame: &GridFrame) -> GridRestored {
    grid_restore_with(frame, 2)
}

/// The D/R/N subframes of one transition: the exact token sets restored
/// from the `1×1` channel differences, plus the per-channel bit counts.
#[derive(Clone, Debug, Default)]
pub struct Subframes {
    pub pair: [u64; 2],
    /// Tokens present in frame A but not frame B (departed subframe).
    pub departed: Vec<String>,
    /// Tokens present in both (retained subframe).
    pub retained: Vec<String>,
    /// Tokens present in frame B but not frame A (new subframe).
    pub new: Vec<String>,
    /// Per-channel D/R/N popcounts (channels: 1×1, 2×2, 3×3, 4×4).
    pub dp: [u64; 4],
    pub rp: [u64; 4],
    pub np: [u64; 4],
}

/// Decompose the transition A → B into its three subframes.
///
/// The `1×1` channel is the token set, so its set differences are exactly
/// the departed / retained / new tokens (modulo hash collisions); each is
/// materialized LUT-first over the frame that holds it.
pub fn subframes(a: &GridFrame, b: &GridFrame) -> Subframes {
    let cells = |f: &GridFrame| -> Vec<Vec<u8>> {
        f.tokens.iter().map(|t| t.as_bytes().to_vec()).collect()
    };
    let ia = ingest_grid(&Grid::new(a.width, a.height, cells(a)));
    let ib = ingest_grid(&Grid::new(b.width, b.height, cells(b)));
    let to_str = |v: Vec<u8>| String::from_utf8_lossy(&v).into_owned();

    let d1 = ia.channels[0].difference(&ib.channels[0]);
    let r1 = ia.channels[0].intersection(&ib.channels[0]);
    let n1 = ib.channels[0].difference(&ia.channels[0]);

    let departed: Vec<String> = materialize_lut_first(&[(&d1, &ia.luts[0])])
        .into_iter()
        .map(to_str)
        .collect();
    let retained: Vec<String> = materialize_lut_first(&[(&r1, &ia.luts[0])])
        .into_iter()
        .map(to_str)
        .collect();
    let new: Vec<String> = materialize_lut_first(&[(&n1, &ib.luts[0])])
        .into_iter()
        .map(to_str)
        .collect();

    let (mut dp, mut rp, mut np) = ([0u64; 4], [0u64; 4], [0u64; 4]);
    for ch in 0..4 {
        dp[ch] = ia.channels[ch].difference(&ib.channels[ch]).popcount();
        rp[ch] = ia.channels[ch].intersection(&ib.channels[ch]).popcount();
        np[ch] = ib.channels[ch].difference(&ia.channels[ch]).popcount();
    }

    Subframes {
        pair: [a.id, b.id],
        departed,
        retained,
        new,
        dp,
        rp,
        np,
    }
}

/// Restore one grid frame with a configurable beam width.
pub fn grid_restore_with(frame: &GridFrame, beam: usize) -> GridRestored {
    let cells: Vec<Vec<u8>> = frame.tokens.iter().map(|t| t.as_bytes().to_vec()).collect();
    let grid = Grid::new(frame.width, frame.height, cells);
    let ing = ingest_grid(&grid);
    let to_str = |v: Vec<u8>| String::from_utf8_lossy(&v).into_owned();

    let ordered = materialize_grid(&ing)
        .cells
        .into_iter()
        .map(to_str)
        .collect();
    let beam_n = materialize_grid_beam(&ing, beam.max(1))
        .cells
        .into_iter()
        .map(to_str)
        .collect();
    let set = materialize_grid_no_order(&ing)
        .into_iter()
        .map(to_str)
        .collect();
    GridRestored {
        id: frame.id,
        width: frame.width,
        height: frame.height,
        ordered,
        set,
        beam2: beam_n,
    }
}

/// One N-d tensor frame for the `conv(n, dim=N)` path: a shape plus the token
/// collection in lexicographic (row-major, last-axis-fastest) order.
#[derive(Clone, Debug)]
pub struct TensorFrame {
    pub id: u64,
    pub shape: Vec<usize>,
    pub tokens: Vec<String>,
}

/// The restored presentations of one tensor frame.
#[derive(Clone, Debug)]
pub struct TensorRestored {
    pub id: u64,
    pub shape: Vec<usize>,
    pub ordered: Vec<String>,
    pub set: Vec<String>,
    pub beam2: Vec<String>,
}

/// Restore one tensor frame through the N-d morphisms with a configurable
/// beam width. The order path is the count-constrained lexicographic walk
/// with joint `2^dim ∧ 3^dim ∧ 4^dim` window checks.
pub fn tensor_restore_with(frame: &TensorFrame, beam: usize) -> TensorRestored {
    let cells: Vec<Vec<u8>> = frame.tokens.iter().map(|t| t.as_bytes().to_vec()).collect();
    let tensor = Tensor::new(frame.shape.clone(), cells);
    let ing = ingest_tensor(&tensor);
    let to_str = |v: Vec<u8>| String::from_utf8_lossy(&v).into_owned();

    let ordered = materialize_tensor(&ing)
        .cells
        .into_iter()
        .map(to_str)
        .collect();
    let beam_n = materialize_tensor_beam(&ing, beam.max(1))
        .cells
        .into_iter()
        .map(to_str)
        .collect();
    let set = materialize_tensor_no_order(&ing)
        .into_iter()
        .map(to_str)
        .collect();
    TensorRestored {
        id: frame.id,
        shape: frame.shape.clone(),
        ordered,
        set,
        beam2: beam_n,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn frame(id: u64, tokens: &[&str]) -> Frame {
        Frame {
            id,
            tokens: tokens.iter().map(|t| t.to_string()).collect(),
        }
    }

    #[test]
    fn bss_and_jaccard_match_lattice_math() {
        let fs = FrameSet::from_frames(vec![
            frame(1, &["tid1", "tid2", "tid3"]),
            frame(2, &["tid2", "tid3", "tid4"]),
        ]);
        // A={1,2,3}, B={2,3,4}: |A∩B|=2, |B|=3, |A∪B|=4
        assert!((fs.bss(0, 1) - 2.0 / 3.0).abs() < 1e-12);
        assert!((fs.jaccard(0, 1) - 2.0 / 4.0).abs() < 1e-12);
    }

    #[test]
    fn sidecar_measures_soft_key_step_and_jumps() {
        let fs = FrameSet::from_frames(vec![
            frame(1, &["tid1", "tid2", "tid3"]),
            frame(2, &["tid2", "tid3", "tid4"]),
            frame(3, &["tid9", "tid10"]), // far from the growing context
        ]);
        let out = fs.sidecar();
        assert_eq!(out.frames.len(), 3);

        // Frame 1: empty basis — nothing to measure against, full novelty.
        assert!(out.frames[0].soft.is_empty());
        assert!(out.frames[0].hard.is_none());
        assert!(!out.frames[0].in_span);
        assert_eq!(out.frames[0].dim, 1);
        assert_eq!(out.frames[0].step, 0.0);

        // Frame 2: basis = {f1}; soft key w = |f2 ∩ f1| / |f1| = 2/3.
        assert_eq!(out.frames[1].soft.len(), 1);
        assert!((out.frames[1].soft[0] - 2.0 / 3.0).abs() < 1e-12);
        assert!(!out.frames[1].in_span, "f2 is not an XOR copy of f1");
        assert_eq!(out.frames[1].dim, 2);
        // Step: |soft(f1) - soft(f2)| in the common basis {f1} = |1 - 2/3|.
        assert!((out.frames[1].step - 1.0 / 3.0).abs() < 1e-12);

        // Frame 3 is far from the context: its step dwarfs the in-context step.
        assert!(out.frames[2].step > out.frames[1].step);
        assert!(out.frames[2].residual > 0);
    }

    #[test]
    fn sidecar_jump_detector_flags_far_frames() {
        // Eleven near-identical frames, then a far one: the jump detector's
        // 3σ threshold sits above the (tiny) in-scene baseline and below the
        // far frame's step.
        let mut frames: Vec<Frame> = Vec::new();
        for i in 1..=11 {
            frames.push(frame(i, &["tid1", "tid2"]));
        }
        frames.push(frame(12, &["tid90", "tid91"]));
        let fs = FrameSet::from_frames(frames);
        let out = fs.sidecar();
        assert_eq!(out.jumps, vec![12], "jumps = {:?}", out.jumps);
        assert!(out.threshold > 0.0);
    }

    #[test]
    fn sidecar_hard_key_is_span_membership() {
        // f2 is an exact copy of f1 -> hard key [true]; f3 duplicates f1's
        // structure again -> still in span, and the step to a repeated frame
        // is zero (both soft keys are all-ones in the same basis).
        let fs = FrameSet::from_frames(vec![
            frame(1, &["tid1", "tid2"]),
            frame(2, &["tid1", "tid2"]),
        ]);
        let out = fs.sidecar();
        assert!(out.frames[1].in_span);
        assert_eq!(out.frames[1].hard, Some(vec![true]));
        assert_eq!(out.frames[1].residual, 0);
        assert_eq!(out.frames[1].dim, 1);
        assert!(out.frames[1].step.abs() < 1e-12);
        assert!(out.jumps.is_empty());
    }

    #[test]
    fn sidecar_freeze_gives_stable_coordinates() {
        // Freeze after two frames: the basis stays {f1, f2} and later frames
        // are evaluated, never inserted — fixed-dim soft keys, zero rotation.
        let fs = FrameSet::from_frames(vec![
            frame(1, &["tid1"]),
            frame(2, &["tid2"]),
            frame(3, &["tid3"]),
            frame(4, &["tid2"]), // in the frozen span
        ]);
        let out = fs.sidecar_with_cap_freeze(64, Some(2));
        assert_eq!(out.frames.len(), 4);

        assert_eq!(out.frames[2].dim, 2, "basis frozen after two frames");
        assert_eq!(out.frames[3].dim, 2);
        assert_eq!(out.frames[2].soft.len(), 2, "fixed-dim soft keys");
        assert_eq!(out.frames[3].soft.len(), 2);
        assert_eq!(out.frames[2].rotation_count, 0, "evaluation never rotates");
        assert_eq!(out.frames[3].rotation_count, 0);
        assert!(!out.frames[2].in_span, "f3 is outside the frozen span");
        assert!(out.frames[3].in_span, "f4 == f2 is in the frozen span");
        assert_eq!(out.frames[3].hard, Some(vec![false, true]));
    }

    #[test]
    fn moving_averages_use_trailing_unions() {
        let fs = FrameSet::from_frames(vec![
            frame(1, &["tid1"]),
            frame(2, &["tid2"]),
            frame(3, &["tid3"]),
        ]);
        let (t0, fast, slow) = fs.moving_averages(1, 2);
        assert_eq!(t0, vec![3]);
        assert_eq!(fast.len(), 1);
        assert_eq!(slow.len(), 1);
        // fast line at t=3: BSS(frame2, frame3) = 0 (disjoint)
        assert!(fast[0].abs() < 1e-12);
        // slow line at t=3: BSS(f1∪f2, f2∪f3) = |{2}| / 2 = 0.5
        assert!((slow[0] - 0.5).abs() < 1e-12);
    }

    #[test]
    fn noether_decomposes_transitions() {
        let fs = FrameSet::from_frames(vec![
            frame(1, &["tid1", "tid2"]),
            frame(2, &["tid2", "tid3"]),
        ]);
        let out = fs.noether();
        let d = fs.hllsets[0].difference(&fs.hllsets[1]).popcount();
        let r = fs.hllsets[0].intersection(&fs.hllsets[1]).popcount();
        let n = fs.hllsets[1].difference(&fs.hllsets[0]).popcount();
        assert_eq!(out.dp, vec![d]); // tid1's atoms departed
        assert_eq!(out.rp, vec![r]); // tid2's atoms retained
        assert_eq!(out.np, vec![n]); // tid3's atoms new
        assert!(d > 0 && r > 0 && n > 0);
    }

    #[test]
    fn restore_recovers_the_ordered_collection() {
        let tokens = ["tid0", "tid1", "tid0", "tid2"];
        let f = frame(1, &tokens);
        let r = restore(&f);
        assert_eq!(r.ordered, tokens);
        assert_eq!(r.set.len(), 3, "set keeps distinct tokens");
        assert_eq!(r.beam2, tokens, "beam-2 agrees on the unambiguous chain");
    }

    #[test]
    fn subframes_split_a_transition_into_d_r_n() {
        let a = GridFrame {
            id: 1,
            width: 2,
            height: 2,
            tokens: ["a", "b", "c", "d"].iter().map(|s| s.to_string()).collect(),
        };
        let b = GridFrame {
            id: 2,
            width: 2,
            height: 2,
            tokens: ["b", "d", "e", "f"].iter().map(|s| s.to_string()).collect(),
        };
        let sf = subframes(&a, &b);
        assert_eq!(sf.pair, [1, 2]);
        assert_eq!(sf.departed, vec!["a".to_string(), "c".to_string()]);
        assert_eq!(sf.retained, vec!["b".to_string(), "d".to_string()]);
        assert_eq!(sf.new, vec!["e".to_string(), "f".to_string()]);
        // Every channel decomposes: D + R = A's channel bits, N + R = B's.
        for ch in 0..4 {
            assert!(sf.dp[ch] + sf.rp[ch] > 0, "channel {ch} has A bits");
            assert!(sf.np[ch] + sf.rp[ch] > 0, "channel {ch} has B bits");
        }
    }

    #[test]
    fn grid_restore_recovers_a_16x16_frame_exactly() {
        let tokens: Vec<String> = (0..256)
            .map(|i| {
                let r = i / 16;
                let c = i % 16;
                format!("tid{}", (r * 37 + c * 7 + (r + c) / 3) % 60)
            })
            .collect();
        let f = GridFrame {
            id: 1,
            width: 16,
            height: 16,
            tokens: tokens.clone(),
        };
        let r = grid_restore_with(&f, 2);
        assert_eq!(r.ordered, tokens, "2D morphism restores the exact grid order");
        assert_eq!(r.beam2, tokens, "beam-2 agrees");
        assert_eq!(r.width, 16);
        assert_eq!(r.height, 16);
    }

    #[test]
    fn tensor_restore_recovers_a_3x4x5_volume_exactly() {
        // A 3×4×5 volume (views × rows × cols) with repeated tids — the
        // multi-view perception-token shape. The N-d joint-window walk must
        // recover the exact lexicographic order.
        let shape = vec![3usize, 4, 5];
        let tokens: Vec<String> = (0..60)
            .map(|i| {
                let v = i / 20;
                let r = (i / 5) % 4;
                let c = i % 5;
                format!("tid{}", (v * 7 + r * 3 + c) % 11)
            })
            .collect();
        let f = TensorFrame {
            id: 1,
            shape: shape.clone(),
            tokens: tokens.clone(),
        };
        let r = tensor_restore_with(&f, 2);
        assert_eq!(r.ordered, tokens, "3D morphism restores the exact volume order");
        assert_eq!(r.beam2, tokens, "beam-2 agrees");
        assert_eq!(r.shape, shape);
    }

    #[test]
    fn pyramid_union_drn_and_ring_decompose() {
        let pf = |id: u64, p1: &[&str], p2: &[&str]| PyramidFrame {
            id,
            perceptrons: vec![
                PerceptronTokens {
                    name: "perception".into(),
                    tokens: p1.iter().map(|s| s.to_string()).collect(),
                },
                PerceptronTokens {
                    name: "ego".into(),
                    tokens: p2.iter().map(|s| s.to_string()).collect(),
                },
            ],
        };
        let frames = vec![
            pf(1, &["a", "b"], &["b", "c"]),
            pf(2, &["a", "b"], &["c", "d"]),
        ];
        let out = pyramid(&frames, 64, None);
        assert_eq!(out.names, vec!["perception", "ego"]);
        assert_eq!(out.frames.len(), 2);

        // (b) joined components: per-perceptron HLLSets + the union.
        let f0 = &out.frames[0];
        assert_eq!(f0.pops.len(), 2);
        assert!(f0.pops[0] > 0 && f0.pops[1] > 0);
        let union12 = f0.hllsets[0].union(&f0.hllsets[1]);
        assert!(union12.difference(&f0.union_hll).is_empty());
        assert!(f0.union_hll.difference(&union12).is_empty(),
                "union equals the OR of the components");

        // (a) D/R/N of the union stream: f1={a,b,c}, f2={a,b,c,d}.
        assert_eq!(out.drn.dp, vec![0], "nothing departed");
        assert!(out.drn.rp[0] > 0, "common atoms retained");
        assert!(out.drn.np[0] > 0, "d's atoms are new");

        // (c) u-ring: first union is added, second is outside the span.
        assert_eq!(out.ring.frames.len(), 2);
        assert!(!out.ring.frames[0].in_span);
        assert!(!out.ring.frames[1].in_span, "f2 != f1 in GF(2)");
        assert_eq!(out.ring.frames[1].dim, 2);
    }

    #[test]
    fn project_gives_the_bss_coordinates_in_any_frame() {
        // Frame = two named dimensions; stream = two frames.
        let dims = vec![
            Dimension::from_tokens("D".into(), &["a".into(), "b".into()]),
            Dimension::from_tokens("N".into(), &["c".into(), "d".into()]),
        ];
        let fs = FrameSet::from_frames(vec![
            frame(1, &["a", "b", "c"]),
            frame(2, &["c", "d"]),
        ]);
        let ids: Vec<u64> = fs.frames.iter().map(|f| f.id).collect();
        let out = project(&fs.hllsets, &fs.g1s, &ids, &dims);

        assert_eq!(out.names, vec!["D", "N"]);
        assert_eq!(out.frames.len(), 2);
        // frame 1: {a,b,c} ∩ D = {a,b} (full), ∩ N = {c} (part of N).
        assert_eq!(out.frames[0].intersections[0], out.pops[0]);
        assert!((out.frames[0].bss[0] - 1.0).abs() < 1e-12);
        assert!(out.frames[0].bss[1] > 0.0 && out.frames[0].bss[1] < 1.0);
        // frame 2: {c,d} ∩ D = 0, ∩ N = full.
        assert!(out.frames[1].bss[0].abs() < 1e-12);
        assert!((out.frames[1].bss[1] - 1.0).abs() < 1e-12);
    }
}
