//! Synthetic random-weight MMDiT shim — the host line.
//!
//! The real host is aarambh-vision-inference's sampler loop; until it
//! stabilizes, this deterministic shim stands in. It integrates a
//! rectified-flow ODE with a random-weight "MMDiT block" per step:
//!
//! ```text
//! dx/dt = (x0 − z) + amp · tanh(W x + t·b) · t(1−t)
//! ```
//!
//! `x0` is the synthetic target latent (the image being generated), `z` the
//! initial noise, `W` a random `dim × dim` matrix shared across latent tokens,
//! `b` a random time-conditioning bias. Random weights are fine: the side-car
//! monitors the *trajectory*, not the image quality
//! (docs/notes/flux-notes.md).

use serde::Serialize;

/// A disturbance injected into the latent trajectory at one denoising step —
/// the synthetic stand-in for a guidance jump / resolution switch / prompt
/// change. It exists to exercise the side-car warnings on a known event.
#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct Disturbance {
    /// Output state index (0-based) where the jump is applied.
    pub step: usize,
    /// Scale of the additive noise (multiplies the standard-normal draw).
    pub scale: f32,
    /// How many leading latent tokens the jump touches (a patch).
    pub patch: usize,
}

/// The synthetic host: a rectified-flow MMDiT trajectory generator.
#[derive(Clone, Debug)]
pub struct SyntheticFluxHost {
    /// Number of MMDiT latent tokens per step (batch = 1).
    pub seq_len: usize,
    /// Latent channels per token.
    pub dim: usize,
    /// Number of denoising steps (the sampler produces `steps + 1` states).
    pub steps: usize,
    /// Amplitude of the random-weight model term (mid-trajectory signature).
    pub amp: f32,
    /// Seed of the whole run (target, noise, weights, disturbance).
    pub seed: u64,
    /// Optional known jump in the trajectory.
    pub disturbance: Option<Disturbance>,
    /// Target latent `x0` (the synthetic image being generated).
    pub x0: Vec<Vec<f32>>,
    /// Initial noise `z` — the S(0) state.
    pub z: Vec<Vec<f32>>,
    /// Random weights `W` (`dim × dim`) — the MMDiT block stand-in, shared
    /// across latent tokens (a per-token dense layer).
    pub w: Vec<Vec<f32>>,
    /// Time-conditioning bias `b` (`dim`).
    pub b: Vec<f32>,
}

impl SyntheticFluxHost {
    /// The default shim: 256 latent tokens × 64 channels, 28 steps, seed 42.
    pub fn new(seq_len: usize, dim: usize, steps: usize, seed: u64) -> Self {
        Self::with_opts(seq_len, dim, steps, seed, 2.0, None)
    }

    /// Full control: model amplitude and an optional disturbance.
    pub fn with_opts(
        seq_len: usize,
        dim: usize,
        steps: usize,
        seed: u64,
        amp: f32,
        disturbance: Option<Disturbance>,
    ) -> Self {
        assert!(seq_len > 0, "seq_len must be positive");
        assert!(dim > 0, "dim must be positive");
        assert!(steps > 0, "steps must be positive");
        let mut state = seed | 1;

        // Target latent: low-rank (rank 4) structure + a smooth positional
        // field + a little noise, so neighboring tokens are correlated like
        // real image latents while every token stays distinct.
        let rank = 4usize;
        let u: Vec<Vec<f32>> = (0..seq_len)
            .map(|_| (0..rank).map(|_| next_std_normal(&mut state)).collect())
            .collect();
        let v: Vec<Vec<f32>> = (0..rank)
            .map(|_| (0..dim).map(|_| next_std_normal(&mut state)).collect())
            .collect();
        let x0 = (0..seq_len)
            .map(|i| {
                (0..dim)
                    .map(|d| {
                        let low_rank: f32 = (0..rank).map(|r| u[i][r] * v[r][d]).sum();
                        let field = 0.6 * ((i as f32 * 0.047 + d as f32 * 0.11).sin());
                        let jitter = 0.08 * next_std_normal(&mut state);
                        low_rank + field + jitter
                    })
                    .collect()
            })
            .collect();

        // Initial noise and the random-weight block.
        let z: Vec<Vec<f32>> = (0..seq_len)
            .map(|_| (0..dim).map(|_| next_std_normal(&mut state)).collect())
            .collect();
        let w: Vec<Vec<f32>> = (0..dim)
            .map(|_| {
                (0..dim)
                    .map(|_| 0.35 / (dim as f32).sqrt() * next_std_normal(&mut state))
                    .collect()
            })
            .collect();
        let b: Vec<f32> = (0..dim).map(|_| 0.2 * next_std_normal(&mut state)).collect();

        Self {
            seq_len,
            dim,
            steps,
            amp,
            seed,
            disturbance,
            x0,
            z,
            w,
            b,
        }
    }

    /// All per-step latent states `S(0..=steps)` of the rectified-flow ODE.
    ///
    /// One Euler step per denoising step, deterministic for a given seed.
    /// `S(0) = z` (pure noise); `S(steps)` sits at the target latent plus the
    /// (small) integrated model detour.
    pub fn sample_states(&self) -> Vec<Vec<Vec<f32>>> {
        let mut states = Vec::with_capacity(self.steps + 1);
        let mut x = self.z.clone();
        states.push(x.clone());

        let dt = 1.0 / self.steps as f32;
        // A separate RNG stream for the disturbance so the trajectory itself
        // does not depend on whether a disturbance is configured.
        let mut dist_rng = self.seed ^ 0x9e37_79b9_7f4a_7c15;
        for k in 0..self.steps {
            let t = (k as f32 + 0.5) * dt; // mid-point time of this step
            let gate = t * (1.0 - t) * self.amp;
            for i in 0..self.seq_len {
                let wx = matvec(&self.w, &x[i]);
                for d in 0..self.dim {
                    let velocity =
                        (self.x0[i][d] - self.z[i][d]) + gate * (wx[d] + self.b[d] * t).tanh();
                    x[i][d] += velocity * dt;
                }
            }

            if let Some(dist) = &self.disturbance {
                if k + 1 == dist.step {
                    let patch = dist.patch.min(self.seq_len);
                    for i in 0..patch {
                        for d in 0..self.dim {
                            x[i][d] += dist.scale * next_std_normal(&mut dist_rng);
                        }
                    }
                }
            }
            states.push(x.clone());
        }
        states
    }
}

/// `W · x` for one latent token.
fn matvec(w: &[Vec<f32>], x: &[f32]) -> Vec<f32> {
    let dim = w.len();
    let mut out = vec![0.0f32; dim];
    for (row, out_d) in w.iter().zip(out.iter_mut()) {
        let acc: f32 = row.iter().zip(x).map(|(wv, xv)| wv * xv).sum();
        *out_d = acc;
    }
    out
}

/// xorshift64 — deterministic stream for the shim.
fn next_u64(state: &mut u64) -> u64 {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    x
}

/// Standard normal via Box–Muller over `next_u64` (same generator family as
/// the ewm-app demo codebook, so the whole run is reproducible).
fn next_std_normal(state: &mut u64) -> f32 {
    let u1 = (next_u64(state) as f64 / u64::MAX as f64).max(1e-12);
    let u2 = next_u64(state) as f64 / u64::MAX as f64;
    ((-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()) as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn same_seed_gives_the_same_trajectory() {
        let a = SyntheticFluxHost::new(32, 16, 8, 7).sample_states();
        let b = SyntheticFluxHost::new(32, 16, 8, 7).sample_states();
        assert_eq!(a.len(), 9);
        assert_eq!(a, b, "the shim is deterministic for a seed");
    }

    #[test]
    fn trajectory_runs_from_noise_to_target() {
        let host = SyntheticFluxHost::new(64, 32, 20, 3);
        let states = host.sample_states();
        let dist = |x: &[Vec<f32>], y: &[Vec<f32>]| -> f64 {
            let mut acc = 0.0f64;
            for i in 0..x.len() {
                for d in 0..x[i].len() {
                    let diff = (x[i][d] - y[i][d]) as f64;
                    acc += diff * diff;
                }
            }
            acc.sqrt()
        };
        // The first state is exactly the noise; the last state is much closer
        // to the target than the noise is.
        assert_eq!(states[0], host.z);
        assert!(
            dist(&states[host.steps], &host.x0) < dist(&host.z, &host.x0),
            "the sampler converges toward the target latent"
        );
    }

    #[test]
    fn disturbance_only_hits_its_step_and_patch() {
        let host = SyntheticFluxHost::with_opts(
            16,
            8,
            10,
            5,
            1.0,
            Some(Disturbance {
                step: 4,
                scale: 100.0,
                patch: 2,
            }),
        );
        let states = host.sample_states();
        let clean = SyntheticFluxHost::with_opts(16, 8, 10, 5, 1.0, None).sample_states();
        for s in 0..states.len() {
            let max_diff = (0..16)
                .map(|i| {
                    (0..8)
                        .map(|d| (states[s][i][d] - clean[s][i][d]).abs())
                        .fold(0.0f32, f32::max)
                })
                .fold(0.0f32, f32::max);
            if s < 4 {
                // Before the disturbance the trajectories are bit-identical.
                assert!(max_diff < 1e-6, "pre-disturbance steps are untouched");
            } else {
                // The jump lands at step 4 and then propagates through the
                // remaining Euler steps of the sampler.
                assert!(max_diff > 1.0, "step {s} carries the disturbance");
            }
        }
    }
}
