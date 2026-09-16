//! The side-car probe — the only host-specific step of the loop.
//!
//! The host line emits per-step **latent states** (dense f32 tensors). The
//! side-car consumes **HLLSets**. [`SidecarProbe`] is the seam between the
//! two: read a tensor, reduce it to bits. Everything downstream of the
//! returned HLLSet is vocabulary- and host-agnostic (docs/SEPARATION.md).

use ewm_app::{CodebookEncoder, Encoder};
use hllset_contracts::token::token_in_bytes;
use hllset_core::HLLSet;
use serde::Serialize;

/// One latent state of the host at a denoising step: `seq_len` MMDiT latent
/// token vectors of `dim` channels each (batch = 1).
pub type LatentState = Vec<Vec<f32>>;

/// The host-agnostic seam: read the host latent state at denoising step
/// `step` and reduce it to the side-car's HLLSet. This is the *only*
/// host-specific step of the loop; when aarambh-vision-inference stabilizes,
/// the port is mostly replacing this implementation with a tensor read.
pub trait SidecarProbe {
    fn probe(&mut self, step: usize, state: &[Vec<f32>]) -> HLLSet;
}

/// The decode-quality record of one step: how well the quantized latent
/// reconstructs the original latent (anchor + batch mean, the inverse of the
/// encoder's centering).
#[derive(Clone, Copy, Debug, Default, Serialize)]
pub struct DecodeRecord {
    /// Mean cosine similarity between the original and reconstructed token
    /// vectors (1.0 = perfect direction reconstruction).
    pub cosine: f64,
    /// Mean squared error per channel (0.0 = perfect reconstruction).
    pub mse: f64,
}

/// The Flux host's probe: a shared codebook quantizer (the same
/// [`CodebookEncoder`] as the Qwen-Drive bench) mapping per-token latent
/// vectors to `tid{n}` ids, then the default morphism to an HLLSet.
pub struct CodebookProbe {
    pub codebook: CodebookEncoder,
}

impl CodebookProbe {
    pub fn new(codebook: CodebookEncoder) -> Self {
        Self { codebook }
    }

    /// Quantize one latent state to `tid{n}` ids (the host's encodings).
    pub fn encode(&self, state: &[Vec<f32>]) -> Vec<u32> {
        self.codebook.encode(state)
    }

    /// The full default ingest of the quantized ids — used by the
    /// materialize round-trip of the loop-accuracy score.
    pub fn ingest(&self, state: &[Vec<f32>]) -> ewm_app::Ingested {
        let ids = self.encode(state);
        ewm_app::ingest(ids.iter().map(|&n| token_in_bytes(n)))
    }

    /// The decode-quality score: reconstruct the latent from the quantized
    /// ids and compare against the original state. Reconstruction is the
    /// encoder's inverse — `anchor[id] + batch_mean`.
    pub fn decode_metrics(&self, state: &[Vec<f32>], ids: &[u32]) -> DecodeRecord {
        if state.is_empty() {
            return DecodeRecord {
                cosine: 1.0,
                mse: 0.0,
            };
        }
        let dim = self.codebook.dim();
        let mean = batch_mean(state, dim);
        let anchors = self.codebook.anchors();

        let mut cos_sum = 0.0f64;
        let mut mse_sum = 0.0f64;
        for (i, &id) in ids.iter().enumerate() {
            let anchor = &anchors[id as usize];
            let mut dot = 0.0f32;
            let mut orig_norm = 0.0f32;
            let mut recon_norm = 0.0f32;
            let mut sq_err = 0.0f32;
            for d in 0..dim {
                let recon = anchor[d] + mean[d];
                dot += state[i][d] * recon;
                orig_norm += state[i][d] * state[i][d];
                recon_norm += recon * recon;
                let diff = state[i][d] - recon;
                sq_err += diff * diff;
            }
            let denom = orig_norm.sqrt() * recon_norm.sqrt();
            cos_sum += if denom > 0.0 {
                (dot / denom) as f64
            } else {
                0.0
            };
            mse_sum += (sq_err / dim as f32) as f64;
        }
        let n = ids.len().max(1) as f64;
        DecodeRecord {
            cosine: cos_sum / n,
            mse: mse_sum / n,
        }
    }
}

impl SidecarProbe for CodebookProbe {
    fn probe(&mut self, _step: usize, state: &[Vec<f32>]) -> HLLSet {
        // The default morphism projection (G1 ∪ G2 ∪ G3) — the same HLLSet
        // the ewm-scene side-car uses for a frame.
        self.ingest(state).projection
    }
}

/// Batch mean over token vectors — the exact centering the encoder applies
/// before nearest-anchor lookup.
fn batch_mean(state: &[Vec<f32>], dim: usize) -> Vec<f32> {
    let mut mean = vec![0.0f32; dim];
    for f in state {
        for (m, x) in mean.iter_mut().zip(f) {
            *m += x;
        }
    }
    let n = state.len() as f32;
    for m in mean.iter_mut() {
        *m /= n;
    }
    mean
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn probe_returns_the_projection_hllset() {
        let codebook = CodebookEncoder::new(8, 32, 11);
        let mut probe = CodebookProbe::new(codebook);
        let state: Vec<Vec<f32>> = (0..16).map(|i| vec![i as f32; 8]).collect();
        let hll = probe.probe(0, &state);
        let ing = probe.ingest(&state);
        assert_eq!(hll.popcount(), ing.projection.popcount());
        assert!(hll.difference(&ing.projection).is_empty());
        assert!(ing.projection.difference(&hll).is_empty());
        assert!(!hll.is_empty());
    }

    #[test]
    fn decode_of_a_codebook_anchor_is_high_cosine() {
        // When every token IS an anchor shifted by a constant, reconstruction
        // is near-exact: the encoder maps each token to a near-identical
        // anchor, and the anchor + batch mean inverse recovers the direction
        // almost perfectly.
        let codebook = CodebookEncoder::new(8, 16, 3);
        let probe = CodebookProbe::new(codebook.clone());
        let anchors = codebook.anchors();
        let state: Vec<Vec<f32>> = anchors
            .iter()
            .map(|a| {
                let mean = vec![0.5f32; 8];
                a.iter().zip(&mean).map(|(x, m)| x + m).collect()
            })
            .collect();
        let ids = probe.encode(&state);
        let metrics = probe.decode_metrics(&state, &ids);
        assert!(metrics.cosine > 0.9, "cosine = {}", metrics.cosine);
        assert!(metrics.mse < 1.0, "mse = {}", metrics.mse);
    }
}
