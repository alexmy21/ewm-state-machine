//! Cardinality estimation for HLLSet.
//!
//! Uses the **Horvitz-Thompson estimator** for bitmap registers where we
//! store the SET of all observed trailing-zero counts (not just the maximum).
//!
//! ## Derivation
//!
//! For each state `s` (bit position 0..31), we count `c_s` = number of
//! registers that have bit `s` set. Then:
//!
//! ```text
//! f̂_s = -n × ln(1 - c_s/n)        (for c_s < M)
//! f̂_s = n × ln(n)                 (for c_s = M, saturated)
//! ```
//!
//! Total cardinality = Σ f̂_s for all states.

use crate::core::hllset::{HLLSet, BITS_PER_REG, M};

impl HLLSet {
    /// Estimate cardinality using the Horvitz-Thompson estimator.
    ///
    /// This is the correct estimator for bitmap registers where we store
    /// the SET of all observed trailing-zero counts.
    pub fn cardinality(&self) -> f64 {
        self.cardinality_ht()
    }

    /// Horvitz-Thompson cardinality estimator.
    ///
    /// For saturated register positions (where every register has the bit
    /// set), we extrapolate from the last non-saturated bit position using
    /// a factor of 2^(distance).
    pub fn cardinality_ht(&self) -> f64 {
        let n = M as f64;
        let m = M as u32;

        // Count registers per bit position
        let mut c_values = [0u32; BITS_PER_REG as usize];
        for bit_pos in 0..BITS_PER_REG {
            c_values[bit_pos as usize] = self.count_registers_with_bit(bit_pos);
        }

        // Find the highest non-saturated bit
        let mut last_non_sat: i32 = -1;
        let mut last_f_hat = 0.0f64;

        for bit_pos in 0..BITS_PER_REG {
            let c_s = c_values[bit_pos as usize];
            if c_s > 0 && c_s < m {
                last_non_sat = bit_pos as i32;
                let ratio = c_s as f64 / n;
                last_f_hat = -n * (1.0 - ratio).ln();
                break;
            }
        }

        let mut total = 0.0f64;
        for bit_pos in 0..BITS_PER_REG {
            let c_s = c_values[bit_pos as usize];

            if c_s == 0 {
                continue;
            } else if c_s < m {
                let ratio = c_s as f64 / n;
                total += -n * (1.0 - ratio).ln();
            } else {
                // Saturated — extrapolate
                if last_non_sat > bit_pos as i32 {
                    total += last_f_hat * 2.0f64.powi(last_non_sat - bit_pos as i32);
                } else {
                    total += n * n.ln();
                }
            }
        }

        total.round().max(0.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_cardinality() {
        let h = HLLSet::new();
        assert_eq!(h.cardinality(), 0.0);
    }

    #[test]
    fn test_single_token() {
        let mut h = HLLSet::new();
        h.add_token(b"hello");
        let card = h.cardinality();
        assert!(card > 0.0);
        // With a single token, cardinality should be close to 1
        assert!((card - 1.0).abs() < 5.0, "card = {card}");
    }

    #[test]
    fn test_many_tokens() {
        let mut h = HLLSet::new();
        for i in 0..1000u32 {
            h.add_token(&i.to_le_bytes());
        }
        let card = h.cardinality();
        // Should be in the right ballpark (within ~10%)
        let error = (card - 1000.0).abs() / 1000.0;
        assert!(error < 0.15, "card={card}, error={error:.2}");
    }

    #[test]
    fn test_cardinality_monotonic() {
        let mut h = HLLSet::new();
        let mut prev = 0.0;
        for i in 0..100u32 {
            h.add_token(&i.to_le_bytes());
            let cur = h.cardinality();
            assert!(cur >= prev, "cardinality decreased at i={i}");
            prev = cur;
        }
    }
}
