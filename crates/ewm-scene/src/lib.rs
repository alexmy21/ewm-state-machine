//! `ewm-scene` — the direct morphisms path for the LLM ↔ HLLSet side-car.
//!
//! The scene-analytics application (rebuild of the vLLM reference) calls this
//! helper instead of the Lua DSL: frames of `tid{n}` tokens go in as JSONL,
//! HLLSet statistics come out as JSON. Everything here is the default
//! application-level interface — [`ewm_app::ingest`], [`ewm_app::materialize`],
//! and the HLLSet lattice operations.

use ewm_app::{ingest, materialize, materialize_beam, materialize_no_order};
use hllset_morphisms::Ingest;
use hllset_core::HLLSet;

/// One frame: `id` + the token collection (in patch order).
#[derive(Clone, Debug)]
pub struct Frame {
    pub id: u64,
    pub tokens: Vec<String>,
}

/// Ingest every frame; keep the projection HLLSet (`G1 ∪ G2 ∪ G3`) and its key.
#[derive(Clone, Debug)]
pub struct FrameSet {
    pub frames: Vec<Frame>,
    pub hllsets: Vec<HLLSet>,
    pub keys: Vec<String>,
    pub pops: Vec<u64>,
}

impl FrameSet {
    pub fn from_frames(frames: Vec<Frame>) -> Self {
        let mut hllsets = Vec::with_capacity(frames.len());
        let mut keys = Vec::with_capacity(frames.len());
        let mut pops = Vec::with_capacity(frames.len());
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
        }
        Self {
            frames,
            hllsets,
            keys,
            pops,
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
        let n = self.len();
        let mut dp = Vec::with_capacity(n.saturating_sub(1));
        let mut rp = Vec::with_capacity(n.saturating_sub(1));
        let mut np = Vec::with_capacity(n.saturating_sub(1));
        let mut ind1 = Vec::with_capacity(n.saturating_sub(1));
        let mut ind3 = Vec::with_capacity(n.saturating_sub(1));
        let mut ind2 = Vec::with_capacity(n.saturating_sub(3));

        let mut r_prev: Option<HLLSet> = None;
        for i in 0..n.saturating_sub(1) {
            let d = self.hllsets[i].difference(&self.hllsets[i + 1]);
            let r = self.hllsets[i].intersection(&self.hllsets[i + 1]);
            let nn = self.hllsets[i + 1].difference(&self.hllsets[i]);
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

/// Restore one frame's token collection: greedy order, plain set, beam-2.
pub fn restore(frame: &Frame) -> Restored {
    let ing = ingest(frame.tokens.iter());
    let to_str = |v: Vec<u8>| String::from_utf8_lossy(&v).into_owned();
    let ordered = materialize(&ing).into_iter().map(to_str).collect();
    let set: Vec<String> = materialize_no_order(&ing).into_iter().map(to_str).collect();
    let beam2 = materialize_beam(&ing, 2).into_iter().map(to_str).collect();
    Restored {
        id: frame.id,
        ordered,
        set,
        beam2,
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
}
