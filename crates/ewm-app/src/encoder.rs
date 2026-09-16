//! The side-car encoder — quantizes host encodings into `tid{n}` ids.
//!
//! The host line (LLM / vision tower) produces **encodings** — dense float
//! vectors. The [UM] consumes `tid{n}` ids, so an [`Encoder`] maps one to
//! the other. This is the *only* vocabulary-aware step in the loop; the
//! morphisms below it stay vocabulary-agnostic (docs/SEPARATION.md).
//!
//! ```text
//! host encodings (f32 vectors)
//!        │  Encoder::encode  (codebook quantization — the DS-OCR step)
//!        ▼
//! tid{n} ids → ingest → HLLSets (S(t), H(t-1), D/R/N, ring)
//!        │
//!        ▼
//! ordered materialize → tid{n} ids → back to the host
//! ```

use hllset_contracts::token::TokenId;

/// Maps host encodings to the LLM encoding ids the [UM] ingests.
pub trait Encoder {
    fn encode(&self, encodings: &[Vec<f32>]) -> Vec<TokenId>;
}

/// The DS-OCR-style codebook quantizer: `count` unit-norm anchors in
/// `dim`-space. Each encoding is centered against the batch mean, then
/// mapped to its nearest anchor (cosine similarity), exactly like the host
/// quantization step.
///
/// Construct it from the host's anchors ([`CodebookEncoder::from_anchors`])
/// so both sides share one codebook, or with the deterministic demo
/// constructor ([`CodebookEncoder::new`]) for tests and notebooks.
#[derive(Clone, Debug)]
pub struct CodebookEncoder {
    anchors: Vec<Vec<f32>>,
}

impl CodebookEncoder {
    /// A deterministic codebook (xorshift + Box–Muller, unit-norm anchors).
    /// The host path should prefer [`from_anchors`](Self::from_anchors) so
    /// the two sides share the exact anchors.
    pub fn new(dim: usize, count: usize, seed: u64) -> Self {
        assert!(dim > 0, "codebook dimension must be positive");
        assert!(count > 0, "codebook must have anchors");
        let mut state = seed | 1;
        let anchors = (0..count)
            .map(|_| {
                let mut v: Vec<f32> = (0..dim).map(|_| next_std_normal(&mut state)).collect();
                normalize(&mut v);
                v
            })
            .collect();
        Self { anchors }
    }

    /// The host's codebook — use this in production so quantization on both
    /// sides is bit-identical.
    pub fn from_anchors(mut anchors: Vec<Vec<f32>>) -> Self {
        assert!(!anchors.is_empty(), "codebook must have anchors");
        for a in anchors.iter_mut() {
            normalize(a);
        }
        Self { anchors }
    }

    pub fn dim(&self) -> usize {
        self.anchors[0].len()
    }

    pub fn count(&self) -> usize {
        self.anchors.len()
    }

    /// The codebook anchors (unit-norm). The host decoder side needs them to
    /// reconstruct latent vectors from quantized ids — the decode-quality
    /// score of the side-car loop.
    pub fn anchors(&self) -> &[Vec<f32>] {
        &self.anchors
    }
}

impl Encoder for CodebookEncoder {
    fn encode(&self, encodings: &[Vec<f32>]) -> Vec<TokenId> {
        if encodings.is_empty() {
            return Vec::new();
        }
        let dim = self.dim();
        // Center the batch (subtract the mean encoding) — the host's step.
        let mut mean = vec![0f32; dim];
        for f in encodings {
            assert_eq!(f.len(), dim, "encoding dimension mismatch");
            for (m, x) in mean.iter_mut().zip(f) {
                *m += x;
            }
        }
        let n = encodings.len() as f32;
        for m in mean.iter_mut() {
            *m /= n;
        }

        encodings
            .iter()
            .map(|f| {
                let centered: Vec<f32> = f.iter().zip(&mean).map(|(x, m)| x - m).collect();
                nearest_anchor(&centered, &self.anchors) as TokenId
            })
            .collect()
    }
}

fn nearest_anchor(feature: &[f32], anchors: &[Vec<f32>]) -> usize {
    let mut best = 0usize;
    let mut best_sim = f32::NEG_INFINITY;
    for (i, a) in anchors.iter().enumerate() {
        let sim: f32 = feature.iter().zip(a).map(|(x, y)| x * y).sum();
        if sim > best_sim {
            best_sim = sim;
            best = i;
        }
    }
    best
}

fn normalize(v: &mut [f32]) {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for x in v.iter_mut() {
            *x /= norm;
        }
    }
}

/// xorshift64 — deterministic stream for the demo constructor.
fn next_u64(state: &mut u64) -> u64 {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    x
}

/// Standard normal via Box–Muller over `next_u64`.
fn next_std_normal(state: &mut u64) -> f32 {
    let u1 = (next_u64(state) as f64 / u64::MAX as f64).max(1e-12);
    let u2 = next_u64(state) as f64 / u64::MAX as f64;
    ((-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()) as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn codebook_is_deterministic_and_unit_norm() {
        let a = CodebookEncoder::new(8, 16, 42);
        let b = CodebookEncoder::new(8, 16, 42);
        let feats: Vec<Vec<f32>> = (0..4).map(|i| vec![i as f32; 8]).collect();
        assert_eq!(a.encode(&feats), b.encode(&feats), "same seed, same ids");
        assert!(a
            .anchors
            .iter()
            .all(|v| (v.iter().map(|x| x * x).sum::<f32>() - 1.0).abs() < 1e-4));
    }

    #[test]
    fn nearest_anchor_picks_the_closest() {
        let anchors = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        assert_eq!(nearest_anchor(&[1.0, 0.0], &anchors), 0);
        assert_eq!(nearest_anchor(&[0.0, 1.0], &anchors), 1);
    }

    #[test]
    fn encoding_roundtrips_through_the_um_loop() {
        use crate::{StateCache, StateMachine};
        use ewm_git::MemoryStore;

        let encoder = CodebookEncoder::new(4, 8, 7);
        let encodings: Vec<Vec<f32>> = vec![
            vec![1.0, 0.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0, 0.0],
            vec![1.0, 0.0, 0.0, 0.0],
        ];

        let mut um = StateMachine::new(MemoryStore::default());
        let mut cache = StateCache::empty();
        let out = um
            .run_turn_encoded(&mut cache, &encoder, &encodings)
            .expect("encoded turn");
        assert_eq!(out.restored_ids, encoder.encode(&encodings),
            "ordered materialize hands the host back its own encodings");
    }
}
