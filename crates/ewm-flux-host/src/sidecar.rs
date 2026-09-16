//! The in-process side-car loop: per-step S(t), H(t-1), D/R/N, the
//! Boolean-ring window, warnings, and the two-score evaluation.
//!
//! The host-agnostic core is [`sidecar_series`] — the same Boolean-ring
//! algebra as the ewm-scene side-car (soft key = BSS weights over the ring
//! basis, hard key = GF(2) coordinates, residual = linear novelty), measured
//! *before* each step enters the window so the record describes the novelty
//! of the incoming state against the context so far (docs/BOOLRING.md).

use std::collections::BTreeSet;

use ewm_app::{ingest, materialize, materialize_no_order};
use ewm_boolring::{BoolBasis, BoolWindow};
use hllset_contracts::token::{parse_token_id, token_in_bytes};
use hllset_core::HLLSet;
use serde::Serialize;

use crate::probe::{CodebookProbe, DecodeRecord, LatentState, SidecarProbe};
use crate::shim::{Disturbance, SyntheticFluxHost};
use crate::RING_CAPACITY;

/// The D/R/N decomposition of one transition — the three disjoint pieces of
/// the same two nodes: `D = S(t-1) \ S(t)`, `R = S(t-1) ∩ S(t)`,
/// `N = S(t) \ S(t-1)` (step 0: `N = S(0)`, `H(-1) = ∅`).
#[derive(Clone, Copy, Debug, Default, Serialize)]
pub struct DrnRecord {
    pub d: u64,
    pub r: u64,
    pub n: u64,
}

/// One step's Boolean-ring measurement.
#[derive(Clone, Copy, Debug, Default, Serialize)]
pub struct RingRecord {
    /// Popcount of the residual against the window span *before* insertion —
    /// the step's linear novelty.
    pub residual: u64,
    /// `true` when the step is already expressible from the window (an XOR
    /// of originals).
    pub in_span: bool,
    /// Span dimension after insertion.
    pub dimension: usize,
    /// Number of existing basis elements re-pivoted by this insertion.
    pub rotation_count: u64,
    /// Total Hamming change of the old basis: `rotation_count * residual`.
    pub rotation_mass: u64,
}

/// The host-agnostic per-step record of the side-car.
#[derive(Clone, Debug, Serialize)]
pub struct SidecarFrame {
    pub step: usize,
    /// The [`BoolWindow`] basis-content generation this frame's `soft` /
    /// `hard` keys were measured against — the **interpretation** of this
    /// step at its own moment.
    pub basis_generation: u64,
    /// `true` when this step's push changed the ring basis (extension /
    /// rotation, or an eviction rebuild). A basis change is a **structural
    /// event** — the context re-indexed itself — and is the second commit
    /// condition next to `new bits > 0`.
    pub basis_change: bool,
    /// Soft key: BSS weights `w_i = |S(t) ∩ B_i| / |B_i|` over the ring
    /// basis *before* this step arrived.
    pub soft: Vec<f64>,
    /// Hard key: GF(2) coordinates — `Some` only when the step is exactly in
    /// the span of the current basis.
    pub hard: Option<Vec<bool>>,
    /// Popcounts `|B_i|` of the basis elements the keys refer to.
    pub basis_pop: Vec<u64>,
    pub ring: RingRecord,
    /// L2 distance between this step's soft key and the previous step's soft
    /// key, both measured in the current basis. `0.0` for the first step.
    pub step_len: f64,
}

/// Warning flags over the trajectory (0-based step indices).
#[derive(Clone, Debug, Default, Serialize)]
pub struct Warnings {
    /// Steps whose soft-key step length exceeds the in-run baseline mean + 3σ.
    pub jumps: Vec<usize>,
    pub jump_threshold: f64,
    /// Steps whose ring residual exceeds the in-run residual baseline mean + 3σ.
    pub residual_spikes: Vec<usize>,
    pub residual_threshold: f64,
    /// Steps whose `N` popcount exceeds the in-run N baseline mean + 3σ.
    pub drn_shifts: Vec<usize>,
    pub drn_threshold: f64,
}

/// The consistent final coordinate system: every step re-projected onto the
/// final ring basis, plus the jump detector over the resulting (now
/// comparable) step series.
#[derive(Clone, Debug, Default, Serialize)]
pub struct FinalProjection {
    /// Dimension of the final basis (all `soft_final` rows share it).
    pub dimension: usize,
    /// The final basis-content generation of the ring window.
    pub generation: u64,
    /// Popcounts of the final basis elements.
    pub basis_pop: Vec<u64>,
    /// Popcount of the final basis cover `∪ B_i` — the part of the bit plane
    /// the frame explains.
    pub cover_pop: u64,
    /// Mean spill `|S(t) \ cover|` over steps — the mean part of a step the
    /// final frame cannot see.
    pub spill_mean: f64,
    /// Spill of the **last** step projected into the **first** recorded
    /// basis frame — the error of reading the newest state in the oldest
    /// basis (nonzero by construction once the basis grew).
    pub spill_oldest: u64,
    /// Steps whose final-basis step length exceeds mean + 3σ.
    pub jumps: Vec<usize>,
    pub jump_threshold: f64,
}

/// One projection of an HLLSet into a basis frame: the soft key (BSS weights
/// over the basis elements — defined for **any** HLLSet), the spill (the
/// part of the set outside the frame's cover — its set-theoretic novelty
/// against this frame), and the hard key (GF(2) coordinates — span members
/// only).
#[derive(Clone, Debug, Default, Serialize)]
pub struct Projection {
    pub soft: Vec<f64>,
    /// `|X \ (∪ B_i)|` — bits of `X` this basis frame cannot see.
    pub spill: u64,
    pub hard: Option<Vec<bool>>,
}

/// One basis frame: the RREF basis at a generation, plus its **cover**
/// (`∪ B_i`, precomputed so the spill is one difference + popcount).
///
/// The basis itself is the coordinate system, so it is what we keep — not
/// per-HLLSet BSS vectors. Any HLLSet can be projected into any stored frame
/// on demand; projecting a *new* HLLSet into an *old* frame is a legitimate
/// similarity reading, and its `spill` is exactly the error of that reading.
#[derive(Clone, Debug)]
pub struct BasisSnapshot {
    pub generation: u64,
    pub basis: BoolBasis,
    pub cover: HLLSet,
}

impl BasisSnapshot {
    pub fn new(generation: u64, basis: &BoolBasis) -> Self {
        let cover = basis
            .basis
            .iter()
            .fold(HLLSet::new(), |acc, b| acc.union(b));
        Self {
            generation,
            basis: basis.clone(),
            cover,
        }
    }

    pub fn dimension(&self) -> usize {
        self.basis.dimension()
    }

    /// Popcounts of the basis elements (the denominators of the soft key).
    pub fn pops(&self) -> Vec<u64> {
        self.basis.basis.iter().map(|b| b.popcount()).collect()
    }

    /// Project `set` into this frame.
    pub fn project(&self, set: &HLLSet) -> Projection {
        let soft: Vec<f64> = self.basis.basis.iter().map(|b| bss(set, b)).collect();
        let spill = set.difference(&self.cover).popcount();
        let hard = self.basis.coordinates(set);
        Projection { soft, spill, hard }
    }
}

/// The history of basis frames, one per basis-content generation change.
///
/// A basis change is a structural event — the context re-indexed itself —
/// which makes it a natural signal for an enforced commit (the side-car can
/// commit on generation bump even when a turn brought no new bits). Keeping
/// every frame lets the caller project any HLLSet into any historical basis.
#[derive(Clone, Debug, Default)]
pub struct BasisHistory {
    pub snapshots: Vec<BasisSnapshot>,
}

impl BasisHistory {
    pub fn new() -> Self {
        Self::default()
    }

    /// Record a snapshot when the window's basis generation changed since the
    /// last recorded one.
    pub fn push_if_changed(&mut self, window: &BoolWindow) {
        let gen = window.generation();
        if self.snapshots.last().map(|s| s.generation) != Some(gen) {
            self.snapshots.push(BasisSnapshot::new(gen, window.basis()));
        }
    }

    pub fn last(&self) -> Option<&BasisSnapshot> {
        self.snapshots.last()
    }

    /// The number of recorded basis changes (generation bumps).
    pub fn change_count(&self) -> usize {
        self.snapshots.len()
    }

    /// Project `set` into the `i`-th recorded frame.
    pub fn project_into(&self, i: usize, set: &HLLSet) -> Projection {
        self.snapshots[i].project(set)
    }
}

/// The result of the host-agnostic side-car series: the per-step frames, the
/// final ring window, and the recorded basis history.
#[derive(Clone, Debug)]
pub struct SidecarSeries {
    pub frames: Vec<SidecarFrame>,
    pub window: BoolWindow,
    pub history: BasisHistory,
}

/// The two-score evaluation summary.
#[derive(Clone, Debug, Default, Serialize)]
pub struct Scores {
    /// Mean per-step **ordered positional** restoration accuracy (expected
    /// 1.0 under the Qwen bench conditions: codebook ≫ token count).
    pub loop_accuracy_mean: f64,
    /// `true` when every step's ordered materialization returned exactly the
    /// quantized ids in order.
    pub loop_order_exact: bool,
    /// Mean hash-level set recall over steps — IICA: every builder is
    /// returned, so this is expected 1.0 always.
    pub loop_set_recall_mean: f64,
    /// Mean hash-level set precision over steps — collisions can add
    /// neighbors, never remove participants (expected 1.0 for an
    /// unambiguous run).
    pub loop_set_precision_mean: f64,
    /// `true` when every step's set restoration is exact (recall = precision
    /// = 1) — the IICA contract.
    pub loop_exact: bool,
    /// Mean decode cosine similarity over steps (latent reconstruction).
    pub decode_cosine_mean: f64,
    /// Mean decode MSE per channel over steps.
    pub decode_mse_mean: f64,
}

/// The run configuration, echoed in the report for reproducibility.
#[derive(Clone, Debug, Serialize)]
pub struct FluxConfig {
    pub seq_len: usize,
    pub dim: usize,
    pub steps: usize,
    pub codebook: usize,
    pub seed: u64,
    pub amp: f32,
    pub disturbance: Option<Disturbance>,
    pub ring_capacity: usize,
}

/// One full step of the Flux side-car loop.
#[derive(Clone, Debug, Serialize)]
pub struct StepFrame {
    pub step: usize,
    /// Denoising time `t = step / steps` (0 = noise, 1 = target).
    pub t: f64,
    /// Popcount of S(t).
    pub pop: u64,
    /// Content key of S(t).
    pub key: String,
    pub drn: DrnRecord,
    /// The basis generation `soft` / `hard` were measured against (online).
    pub basis_generation: u64,
    /// `true` when this step changed the ring basis — a structural event,
    /// the second commit condition next to `new bits > 0`.
    pub basis_change: bool,
    pub soft: Vec<f64>,
    pub hard: Option<Vec<bool>>,
    pub basis_pop: Vec<u64>,
    pub ring: RingRecord,
    pub step_len: f64,
    /// Soft key re-projected onto the **final** basis — every row of
    /// `soft_final` lives in the same coordinate system.
    pub soft_final: Vec<f64>,
    /// L2 distance between consecutive `soft_final` vectors. `0.0` for the
    /// first step.
    pub step_final: f64,
    /// `|S(t) \ cover(final basis)|` — the part of this step the final basis
    /// frame cannot see (its set-theoretic novelty against that frame).
    pub spill_final: u64,
    /// The quantized ids (host encodings) of this step.
    pub ids: Vec<u32>,
    /// The ordered materialization parsed back to ids — the hand-back to the
    /// host.
    pub restored_ids: Vec<u32>,
    /// Position-wise restoration accuracy of this step (1.0 = exact).
    pub loop_accuracy: f64,
    /// Hash-level set recall of this step (`|orig ∩ restored| / |orig|`).
    pub loop_recall: f64,
    /// Hash-level set precision of this step (`|orig ∩ restored| / |restored|`).
    pub loop_precision: f64,
    pub decode: DecodeRecord,
}

/// The whole Flux side-car run: per-step frames, warnings, the consistent
/// final projection, the basis history, and the two-score summary.
#[derive(Clone, Debug, Serialize)]
pub struct FluxReport {
    pub config: FluxConfig,
    pub frames: Vec<StepFrame>,
    pub warnings: Warnings,
    /// The consistent final-basis view of the trajectory.
    pub final_projection: FinalProjection,
    /// Number of basis-content changes (generation bumps) during the run —
    /// each is a structural event of the context re-indexing itself, a
    /// natural signal for an enforced commit.
    pub basis_changes: usize,
    /// The recorded basis generations, in order of change.
    pub basis_generations: Vec<u64>,
    pub scores: Scores,
}

/// The D/R/N series over an HLLSet stream (the same lattice math at every
/// level of the pyramid — here, one level: denoising steps).
pub fn drn_series(hllsets: &[HLLSet]) -> Vec<DrnRecord> {
    let mut out = Vec::with_capacity(hllsets.len());
    for (i, s) in hllsets.iter().enumerate() {
        if i == 0 {
            out.push(DrnRecord {
                d: 0,
                r: 0,
                n: s.popcount(),
            });
        } else {
            let prev = &hllsets[i - 1];
            out.push(DrnRecord {
                d: prev.difference(s).popcount(),
                r: prev.intersection(s).popcount(),
                n: s.difference(prev).popcount(),
            });
        }
    }
    out
}

/// The side-car trajectory over an HLLSet series — the host-agnostic core.
///
/// Wired exactly like the [UM] cache ring: a [`BoolWindow`] over the original
/// per-step HLLSets with [`RING_CAPACITY`]. For each step, before it enters
/// the window, the soft key (BSS weights over the window basis), the hard key
/// (coordinates when the step is in the span) and the step length are
/// measured; then the step is pushed (eviction slides the window and
/// recomputes the basis).
///
/// Soft keys are computed on demand against the live basis — a projection is
/// sub-millisecond (a handful of roaring intersections per basis element), so
/// no per-HLLSet BSS vectors are stored. Instead every **basis change** is
/// recorded as a [`BasisSnapshot`] in the [`BasisHistory`], so any HLLSet can
/// later be projected into any historical basis.
pub fn sidecar_series(hllsets: &[HLLSet]) -> Vec<SidecarFrame> {
    sidecar_series_with_window(hllsets).frames
}

/// The side-car trajectory plus the final ring window and the recorded basis
/// history.
pub fn sidecar_series_with_window(hllsets: &[HLLSet]) -> SidecarSeries {
    let mut window = BoolWindow::new(RING_CAPACITY);
    let mut history = BasisHistory::new();
    let mut frames = Vec::with_capacity(hllsets.len());

    for set in hllsets {
        let idx = frames.len();

        let basis = window.basis();
        let basis_pop: Vec<u64> = basis.basis.iter().map(|b| b.popcount()).collect();
        let hard = basis.coordinates(set);
        let basis_generation = window.generation();
        let soft: Vec<f64> = basis.basis.iter().map(|b| bss(set, b)).collect();

        // Step length: both frames measured against the *current* basis, so
        // consecutive steps share one coordinate system.
        let step_len = if idx == 0 {
            0.0
        } else {
            let prev_soft: Vec<f64> = basis
                .basis
                .iter()
                .map(|b| bss(&hllsets[idx - 1], b))
                .collect();
            l2_distance(&prev_soft, &soft)
        };

        // Push after measuring, so the record describes the novelty of the
        // incoming state against the context so far.
        let stats = window.push(set);
        let basis_change = window.generation() != basis_generation;
        history.push_if_changed(&window);
        frames.push(SidecarFrame {
            step: idx,
            basis_generation,
            basis_change,
            soft,
            hard,
            basis_pop,
            ring: RingRecord {
                residual: stats.residual,
                in_span: stats.in_span,
                dimension: stats.dimension,
                rotation_count: stats.rotation_count,
                rotation_mass: stats.rotation_mass,
            },
            step_len,
        });
    }
    SidecarSeries {
        frames,
        window,
        history,
    }
}

/// Flag the warning series: any value above the in-run baseline mean + 3σ.
/// The first step is excluded from every baseline (it is the full-novelty
/// seed, not a transition).
pub fn warnings(frames: &[SidecarFrame], drn: &[DrnRecord]) -> Warnings {
    if frames.is_empty() {
        return Warnings::default();
    }
    let steps: Vec<f64> = frames.iter().map(|f| f.step_len).collect();
    let residuals: Vec<f64> = frames.iter().map(|f| f.ring.residual as f64).collect();
    let ns: Vec<f64> = drn.iter().map(|d| d.n as f64).collect();

    let (mut jumps, jump_threshold) = threshold_flags(&steps[1..]);
    for j in &mut jumps {
        *j += 1;
    }
    let (mut residual_spikes, residual_threshold) = threshold_flags(&residuals[1..]);
    for j in &mut residual_spikes {
        *j += 1;
    }
    let (mut drn_shifts, drn_threshold) = threshold_flags(&ns[1..]);
    for j in &mut drn_shifts {
        *j += 1;
    }

    Warnings {
        jumps,
        jump_threshold,
        residual_spikes,
        residual_threshold,
        drn_shifts,
        drn_threshold,
    }
}

/// Run the full side-car loop over the host's synthetic latent trajectory.
pub fn run(host: &SyntheticFluxHost, probe: &mut CodebookProbe) -> FluxReport {
    let states = host.sample_states();
    run_with_config(
        &states,
        probe,
        FluxConfig {
            seq_len: host.seq_len,
            dim: host.dim,
            steps: host.steps,
            codebook: probe.codebook.count(),
            seed: host.seed,
            amp: host.amp,
            disturbance: host.disturbance,
            ring_capacity: RING_CAPACITY,
        },
    )
}

/// Run the full side-car loop over **any** latent trajectory — the real-host
/// path. `states[t]` is the per-token latent state at denoising step `t`
/// (`states.len() - 1` denoising steps, the first state being noise). The
/// config is derived from the trajectory itself (seed 0, no disturbance).
pub fn run_states(states: &[LatentState], probe: &mut CodebookProbe) -> FluxReport {
    let seq_len = states.first().map_or(0, |s| s.len());
    let dim = states.first().and_then(|s| s.first()).map_or(0, |v| v.len());
    let steps = states.len().saturating_sub(1);
    run_with_config(
        states,
        probe,
        FluxConfig {
            seq_len,
            dim,
            steps,
            codebook: probe.codebook.count(),
            seed: 0,
            amp: 0.0,
            disturbance: None,
            ring_capacity: RING_CAPACITY,
        },
    )
}

fn run_with_config(
    states: &[LatentState],
    probe: &mut CodebookProbe,
    config: FluxConfig,
) -> FluxReport {
    let mut hllsets = Vec::with_capacity(states.len());
    let mut keys = Vec::with_capacity(states.len());
    let mut pops = Vec::with_capacity(states.len());
    let mut ids_per_step = Vec::with_capacity(states.len());
    let mut restored_per_step = Vec::with_capacity(states.len());
    let mut loop_acc_per_step = Vec::with_capacity(states.len());
    let mut recall_per_step = Vec::with_capacity(states.len());
    let mut precision_per_step = Vec::with_capacity(states.len());
    let mut decode_per_step = Vec::with_capacity(states.len());

    for (step, state) in states.iter().enumerate() {
        let s = probe.probe(step, state);
        keys.push(s.content_key());
        pops.push(s.popcount());
        hllsets.push(s);

        // Score 1 — loop accuracy: the ordered materialization must hand the
        // host back its own quantized encodings (positional), and the
        // hash-level set restoration must return every builder (IICA).
        let ids = probe.encode(state);
        let ing = ingest(ids.iter().map(|&n| token_in_bytes(n)));
        let restored: Vec<u32> = materialize(&ing)
            .iter()
            .filter_map(|b| parse_token_id(b))
            .collect();
        let loop_accuracy = position_accuracy(&ids, &restored);
        let orig_set: BTreeSet<u32> = ids.iter().copied().collect();
        let restored_set: BTreeSet<u32> = materialize_no_order(&ing)
            .iter()
            .filter_map(|b| parse_token_id(b))
            .collect();
        let inter = orig_set.intersection(&restored_set).count();
        let recall = inter as f64 / orig_set.len().max(1) as f64;
        let precision = inter as f64 / restored_set.len().max(1) as f64;
        // Score 2 — decode quality: latent reconstruction from the codebook.
        let decode = probe.decode_metrics(state, &ids);

        ids_per_step.push(ids);
        restored_per_step.push(restored);
        loop_acc_per_step.push(loop_accuracy);
        recall_per_step.push(recall);
        precision_per_step.push(precision);
        decode_per_step.push(decode);
    }

    let sidecar = sidecar_series_with_window(&hllsets);
    let drn = drn_series(&hllsets);
    let warn = warnings(&sidecar.frames, &drn);

    // Re-project every step onto the **final** basis. BSS projections are
    // computed on demand (sub-millisecond each — see BasisSnapshot); what is
    // stored are the basis frames themselves, not per-HLLSet BSS vectors.
    let SidecarSeries {
        frames: core,
        history,
        ..
    } = sidecar;
    let basis_changes = history.change_count();
    let basis_generations: Vec<u64> = history.snapshots.iter().map(|s| s.generation).collect();

    let n = hllsets.len();
    let mut final_soft = Vec::with_capacity(n);
    let mut final_spills = Vec::with_capacity(n);
    match history.last() {
        Some(snap) => {
            for set in &hllsets {
                let p = snap.project(set);
                final_soft.push(p.soft);
                final_spills.push(p.spill);
            }
        }
        None => {
            for _ in 0..n {
                final_soft.push(Vec::new());
                final_spills.push(0);
            }
        }
    }

    let (final_pops, final_generation, final_dimension, final_cover_pop) = match history.last() {
        Some(snap) => (
            snap.pops(),
            snap.generation,
            snap.dimension(),
            snap.cover.popcount(),
        ),
        None => (Vec::new(), 0, 0, 0),
    };
    // The error of reading the newest state in the oldest basis frame — the
    // "new HLLSet into an old basis" case: the soft key still reads, and the
    // spill measures how much of the set the old frame cannot see.
    let spill_oldest = match (history.snapshots.first(), hllsets.last()) {
        (Some(snap), Some(last)) => snap.project(last).spill,
        _ => 0,
    };
    let spill_mean = mean(&final_spills.iter().map(|&s| s as f64).collect::<Vec<_>>());

    let step_final: Vec<f64> = final_soft
        .iter()
        .enumerate()
        .map(|(i, s)| {
            if i == 0 {
                0.0
            } else {
                l2_distance(&final_soft[i - 1], s)
            }
        })
        .collect();
    let (mut final_jumps, final_threshold) = threshold_flags(&step_final[1..]);
    for j in &mut final_jumps {
        *j += 1;
    }

    let steps_f = config.steps.max(1) as f64;
    let frames: Vec<StepFrame> = core
        .into_iter()
        .enumerate()
        .map(|(i, c)| StepFrame {
            step: i,
            t: i as f64 / steps_f,
            pop: pops[i],
            key: keys[i].clone(),
            drn: drn[i],
            basis_generation: c.basis_generation,
            basis_change: c.basis_change,
            soft: c.soft,
            hard: c.hard,
            basis_pop: c.basis_pop,
            ring: c.ring,
            step_len: c.step_len,
            soft_final: final_soft[i].clone(),
            step_final: step_final[i],
            spill_final: final_spills[i],
            ids: ids_per_step[i].clone(),
            restored_ids: restored_per_step[i].clone(),
            loop_accuracy: loop_acc_per_step[i],
            loop_recall: recall_per_step[i],
            loop_precision: precision_per_step[i],
            decode: decode_per_step[i],
        })
        .collect();

    let loop_order_exact = frames
        .iter()
        .all(|f| f.ids == f.restored_ids && (f.loop_accuracy - 1.0).abs() < 1e-12);
    let loop_exact = frames.iter().all(|f| {
        (f.loop_recall - 1.0).abs() < 1e-12 && (f.loop_precision - 1.0).abs() < 1e-12
    });
    let scores = Scores {
        loop_accuracy_mean: mean(&loop_acc_per_step),
        loop_order_exact,
        loop_set_recall_mean: mean(&recall_per_step),
        loop_set_precision_mean: mean(&precision_per_step),
        loop_exact,
        decode_cosine_mean: mean(&frames.iter().map(|f| f.decode.cosine).collect::<Vec<_>>()),
        decode_mse_mean: mean(&frames.iter().map(|f| f.decode.mse).collect::<Vec<_>>()),
    };

    FluxReport {
        config,
        frames,
        warnings: warn,
        final_projection: FinalProjection {
            dimension: final_dimension,
            generation: final_generation,
            basis_pop: final_pops,
            cover_pop: final_cover_pop,
            spill_mean,
            spill_oldest,
            jumps: final_jumps,
            jump_threshold: final_threshold,
        },
        basis_changes,
        basis_generations,
        scores,
    }
}

/// BSSτ between two HLLSets: `|A ∩ B| / |B|` (1.0 for an empty denominator).
fn bss(a: &HLLSet, b: &HLLSet) -> f64 {
    let denom = b.popcount() as f64;
    if denom == 0.0 {
        1.0
    } else {
        a.intersection(b).popcount() as f64 / denom
    }
}

/// L2 distance between two soft-key vectors (measured in the same basis).
fn l2_distance(a: &[f64], b: &[f64]) -> f64 {
    debug_assert_eq!(a.len(), b.len(), "soft keys must share one basis");
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

fn std_dev(xs: &[f64], m: f64) -> f64 {
    if xs.len() < 2 {
        return 0.0;
    }
    let var = xs.iter().map(|x| (x - m) * (x - m)).sum::<f64>() / xs.len() as f64;
    var.sqrt()
}

/// Indices of values above `mean + 3σ` within the baseline slice.
fn threshold_flags(baseline: &[f64]) -> (Vec<usize>, f64) {
    let m = mean(baseline);
    let std = std_dev(baseline, m);
    let threshold = m + 3.0 * std;
    let flags: Vec<usize> = baseline
        .iter()
        .enumerate()
        .filter_map(|(i, &v)| if v > threshold { Some(i) } else { None })
        .collect();
    (flags, threshold)
}

/// Position-wise agreement between the quantized ids and the restored ids
/// (1.0 when both are empty).
fn position_accuracy(a: &[u32], b: &[u32]) -> f64 {
    if a.is_empty() && b.is_empty() {
        return 1.0;
    }
    let n = a.len().max(b.len()).max(1) as f64;
    let agree = a.iter().zip(b).filter(|(x, y)| x == y).count() as f64;
    agree / n
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::probe::CodebookProbe;
    use ewm_app::CodebookEncoder;

    fn set(bits: &[u32]) -> HLLSet {
        let mut h = HLLSet::new();
        for b in bits {
            h.add_bit(*b);
        }
        h
    }

    #[test]
    fn drn_series_decomposes_transitions() {
        let hllsets = vec![set(&[1, 2]), set(&[2, 3])];
        let drn = drn_series(&hllsets);
        assert_eq!(drn[0].d, 0);
        assert_eq!(drn[0].r, 0);
        assert_eq!(drn[0].n, 2, "step 0: N = S(0)");
        assert_eq!(drn[1].d, 1, "bit 1 departed");
        assert_eq!(drn[1].r, 1, "bit 2 retained");
        assert_eq!(drn[1].n, 1, "bit 3 new");
    }

    #[test]
    fn sidecar_series_matches_ewm_scene_semantics() {
        // Same expectations as ewm-scene's sidecar tests: an empty basis for
        // step 0, soft key 2/3 for step 1 against {f1}, and the far step 2
        // carries both a bigger step and a nonempty residual.
        let hllsets = vec![set(&[1, 2, 3]), set(&[2, 3, 4]), set(&[9, 10])];
        let frames = sidecar_series(&hllsets);
        assert_eq!(frames.len(), 3);

        assert!(frames[0].soft.is_empty());
        assert!(frames[0].hard.is_none());
        assert!(!frames[0].ring.in_span);
        assert_eq!(frames[0].ring.dimension, 1);
        assert_eq!(frames[0].step_len, 0.0);

        assert_eq!(frames[1].soft.len(), 1);
        assert!((frames[1].soft[0] - 2.0 / 3.0).abs() < 1e-12);
        assert!(!frames[1].ring.in_span);
        assert_eq!(frames[1].ring.dimension, 2);
        assert!((frames[1].step_len - 1.0 / 3.0).abs() < 1e-12);

        assert!(frames[2].step_len > frames[1].step_len);
        assert!(frames[2].ring.residual > 0);
    }

    #[test]
    fn warnings_flag_a_far_step() {
        // Eleven near-identical frames, then a far one — the jump detector's
        // 3σ threshold sits above the tiny baseline and below the far step.
        let mut hllsets: Vec<HLLSet> = Vec::new();
        for _ in 0..11 {
            hllsets.push(set(&[1, 2]));
        }
        hllsets.push(set(&[90, 91]));
        let frames = sidecar_series(&hllsets);
        let drn = drn_series(&hllsets);
        let w = warnings(&frames, &drn);
        assert_eq!(w.jumps, vec![11], "jumps = {:?}", w.jumps);
        assert!(w.jump_threshold > 0.0);
    }

    #[test]
    fn run_closes_the_loop_with_exact_restoration() {
        // A codebook much larger than the token count (the Qwen bench
        // condition) keeps the n-gram order path unambiguous, so both the
        // ordered and the hash-level restoration are exact.
        let host = SyntheticFluxHost::new(64, 32, 10, 9);
        let mut probe = CodebookProbe::new(CodebookEncoder::new(32, 4096, 9));
        let report = run(&host, &mut probe);

        assert_eq!(report.frames.len(), 11);
        assert!(report.scores.loop_order_exact, "ordered loop must be exact");
        assert!(report.scores.loop_exact, "hash-level loop must be exact");
        assert!((report.scores.loop_accuracy_mean - 1.0).abs() < 1e-12);
        assert!((report.scores.loop_set_recall_mean - 1.0).abs() < 1e-12);
        assert!((report.scores.loop_set_precision_mean - 1.0).abs() < 1e-12);
        for f in &report.frames {
            assert_eq!(f.ids, f.restored_ids, "step {} round-trips", f.step);
            assert!((f.loop_recall - 1.0).abs() < 1e-12);
            assert!((f.loop_precision - 1.0).abs() < 1e-12);
        }
        // Decode quality is defined for every step.
        for f in &report.frames {
            assert!(f.decode.cosine.is_finite());
            assert!(f.decode.mse.is_finite());
        }
    }

    #[test]
    fn hash_level_restoration_holds_even_when_order_is_ambiguous() {
        // A tiny codebook makes the ordered walk ambiguous (the restored ids
        // come back as a permutation), but IICA still returns every builder:
        // set recall and precision stay 1.0.
        let host = SyntheticFluxHost::new(64, 32, 10, 9);
        let mut probe = CodebookProbe::new(CodebookEncoder::new(32, 128, 9));
        let report = run(&host, &mut probe);
        assert!(report.scores.loop_exact, "hash-level restoration must be exact");
        assert!((report.scores.loop_set_recall_mean - 1.0).abs() < 1e-12);
        assert!((report.scores.loop_set_precision_mean - 1.0).abs() < 1e-12);
    }

    #[test]
    fn run_states_accepts_an_external_latent_trajectory() {
        // The real-host path: any latent trajectory, not just the synthetic
        // shim. Config is derived from the trajectory itself.
        let host = SyntheticFluxHost::new(32, 16, 6, 5);
        let states = host.sample_states();
        let mut probe = CodebookProbe::new(CodebookEncoder::new(16, 1024, 5));
        let report = run_states(&states, &mut probe);
        assert_eq!(report.frames.len(), states.len());
        assert_eq!(report.config.steps, 6);
        assert_eq!(report.config.seq_len, 32);
        assert_eq!(report.config.dim, 16);
        assert!(report.scores.loop_order_exact);
        assert!(report.scores.loop_exact);
    }

    #[test]
    fn disturbance_fires_warnings() {
        let host = SyntheticFluxHost::with_opts(
            128,
            32,
            24,
            21,
            2.0,
            Some(Disturbance {
                step: 12,
                scale: 8.0,
                patch: 32,
            }),
        );
        let mut probe = CodebookProbe::new(CodebookEncoder::new(32, 512, 21));
        let report = run(&host, &mut probe);
        assert!(
            report.warnings.jumps.contains(&12)
                || report.warnings.residual_spikes.contains(&12)
                || report.warnings.drn_shifts.contains(&12),
            "the injected jump must fire at least one warning: {:?}",
            report.warnings
        );
    }

    #[test]
    fn sidecar_series_stamps_each_frame_with_its_basis_generation() {
        let hllsets = vec![set(&[1, 2]), set(&[2, 3])];
        let series = sidecar_series_with_window(&hllsets);
        assert_eq!(series.frames[0].basis_generation, 0, "empty basis for step 0");
        assert_eq!(series.frames[1].basis_generation, 1, "basis {{f1}} for step 1");
        assert_eq!(series.window.generation(), 2, "two additions bumped twice");
        assert!(series.frames[0].basis_change, "first push changes the basis");
        assert!(series.frames[1].basis_change, "second push changes the basis");

        // A repeated step is in-span: no basis change, no structural event.
        let series = sidecar_series_with_window(&vec![set(&[1, 2]), set(&[1, 2])]);
        assert!(series.frames[0].basis_change);
        assert!(!series.frames[1].basis_change);
        assert_eq!(series.window.generation(), 1);
    }

    #[test]
    fn basis_history_records_every_basis_change_and_projects_into_any_frame() {
        let a = set(&[1, 2]);
        let b = set(&[2, 3]);

        let mut window = BoolWindow::new(64);
        let mut history = BasisHistory::new();
        assert_eq!(history.change_count(), 0);

        // The empty starting basis is a frame too.
        history.push_if_changed(&window);
        assert_eq!(history.change_count(), 1);
        assert_eq!(history.last().unwrap().generation, 0);
        assert_eq!(history.last().unwrap().dimension(), 0);

        window.push(&a); // basis changes -> recorded
        history.push_if_changed(&window);
        assert_eq!(history.change_count(), 2);
        assert_eq!(history.last().unwrap().generation, 1);

        window.push(&a); // in-span -> not a basis change, not recorded
        history.push_if_changed(&window);
        assert_eq!(history.change_count(), 2);

        window.push(&b); // basis changes again -> recorded
        history.push_if_changed(&window);
        assert_eq!(history.change_count(), 3);
        assert_eq!(history.last().unwrap().dimension(), 2);

        // Project `a` into frame 1 (basis {a}): inside, no spill, exact coords.
        let p = history.project_into(1, &a);
        assert_eq!(p.soft.len(), 1);
        assert!((p.soft[0] - 1.0).abs() < 1e-12);
        assert_eq!(p.spill, 0);
        assert_eq!(p.hard, Some(vec![true]));

        // Project `b` (a *newer* set) into the *older* frame {a}: the soft key
        // is still a valid similarity reading, and the spill is exactly the
        // part of `b` outside cover(a) = {1, 2}.
        let p2 = history.project_into(1, &b);
        assert!((p2.soft[0] - 0.5).abs() < 1e-12, "|b ∩ a| / |a| = 1/2");
        assert_eq!(p2.spill, 1, "b \\ a = {{3}}");
        assert_eq!(p2.hard, None, "b is outside the span of frame {{a}}");
    }

    #[test]
    fn final_projection_is_a_single_coordinate_system() {
        let host = SyntheticFluxHost::new(32, 16, 6, 5);
        let mut probe = CodebookProbe::new(CodebookEncoder::new(16, 1024, 5));
        let report = run(&host, &mut probe);

        let dim = report.final_projection.dimension;
        assert_eq!(dim, report.final_projection.basis_pop.len());
        assert!(report.basis_changes > 0, "the basis grows during the run");
        assert_eq!(report.basis_generations.len(), report.basis_changes);
        assert!(report.final_projection.cover_pop > 0);
        assert!(report.final_projection.spill_mean >= 0.0);

        for f in &report.frames {
            assert_eq!(
                f.soft_final.len(),
                dim,
                "every soft_final row shares the final basis"
            );
            assert!(
                f.spill_final <= report.final_projection.cover_pop,
                "spill cannot exceed the cover"
            );
        }
        assert_eq!(report.frames[0].step_final, 0.0);
        // step_final is the L2 distance between consecutive consistent rows.
        for i in 1..report.frames.len() {
            let expect = l2_distance(
                &report.frames[i - 1].soft_final,
                &report.frames[i].soft_final,
            );
            assert!((report.frames[i].step_final - expect).abs() < 1e-12);
        }
    }
}
