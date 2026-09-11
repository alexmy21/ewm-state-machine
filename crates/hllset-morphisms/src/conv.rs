//! The `conv(n, dim)` channel model.
//!
//! n-grams are 1D convolutions over a token sequence; grids generalize the
//! same morphisms to 2D convolutions over a token matrix:
//!
//! ```text
//! conv(1, 1)   1×1   seed 0   G1     (shared: 1-gram = 1-conv)
//! conv(2, 1)   2×1   seed 1   G2_1d
//! conv(3, 1)   3×1   seed 2   G3_1d
//! conv(4, 1)   4×1   seed 3   G4_1d  (1D order side channel)
//! conv(2, 2)   2×2   seed 4   G2_2d
//! conv(3, 2)   3×3   seed 5   G3_2d
//! conv(4, 2)   4×4   seed 6   G4_2d  (2D order side channel)
//! ```
//!
//! Seed rule: `seed(1, ·) = 0` — the 1×1 channel is dimension-independent,
//! so G1 stays the shared gate. For `n ≥ 2`:
//! `seed(n, dim) = (dim − 1)·3 + (n − 1)`.

/// One convolution specification: an `n × … × n` window in `dim` dimensions.
///
/// `dim == 1` is the n-gram regime (window `n×1`); `dim == 2` is the grid
/// regime (window `n×n`). `n == 1` is the shared token channel in any
/// dimension.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ConvSpec {
    pub n: u8,
    pub dim: u8,
}

impl ConvSpec {
    pub const fn new(n: u8, dim: u8) -> Self {
        Self { n, dim }
    }
}

/// The channel seed of a convolution specification.
///
/// `seed(1, ·) = 0` (shared G1); for `n ≥ 2`:
/// `seed(n, dim) = (dim − 1)·3 + (n − 1)`.
pub const fn seed(spec: ConvSpec) -> u64 {
    if spec.n <= 1 {
        0
    } else {
        ((spec.dim.saturating_sub(1) * 3) + (spec.n - 1)) as u64
    }
}

/// The hllsetLUT name of a channel: `G1` for the shared 1×1 channel,
/// `G{n}_{dim}d` otherwise (e.g. `G2_1d`, `G3_2d`).
pub fn channel_name(n: u8, dim: u8) -> String {
    if n <= 1 {
        "G1".to_string()
    } else {
        format!("G{n}_{dim}d")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn seed_scheme_shares_g1_and_partitions_the_rest() {
        // The 1×1 channel is the same atom space in every dimension.
        assert_eq!(seed(ConvSpec::new(1, 1)), 0);
        assert_eq!(seed(ConvSpec::new(1, 2)), 0);
        // dim=1: n-grams (and the 4-gram order channel).
        assert_eq!(seed(ConvSpec::new(2, 1)), 1);
        assert_eq!(seed(ConvSpec::new(3, 1)), 2);
        assert_eq!(seed(ConvSpec::new(4, 1)), 3);
        // dim=2: 2D convolutions.
        assert_eq!(seed(ConvSpec::new(2, 2)), 4);
        assert_eq!(seed(ConvSpec::new(3, 2)), 5);
        assert_eq!(seed(ConvSpec::new(4, 2)), 6);
    }

    #[test]
    fn channel_names_follow_the_table() {
        assert_eq!(channel_name(1, 1), "G1");
        assert_eq!(channel_name(1, 2), "G1");
        assert_eq!(channel_name(2, 1), "G2_1d");
        assert_eq!(channel_name(3, 2), "G3_2d");
        assert_eq!(channel_name(4, 2), "G4_2d");
    }
}
