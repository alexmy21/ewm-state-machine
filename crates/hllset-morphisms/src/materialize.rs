//! Materialization: LUT-first, keep every reference.
//!
//! For each active bit of each sketch, gather candidate tokens from the
//! **corresponding pointed LUT** (every encoding). A bit with several
//! candidates restores **all** of them — collisions are normal in large
//! token collections, and dropping candidates would be an incorrect filter.
//! TF is never consulted for materialization; it exists for ranking only.
//! This is **probabilistic restoration**: the LUTs give every token that
//! could have set the bit; the sketch only says the bit is active.

use hllset_core::HLLSet;
use ::hllset_lut::LutIndex;
use std::collections::BTreeSet;

/// `M(H, {L_j})` — materialize across encodings.
///
/// Takes one `(sketch, LUT)` pair per encoding; the LUT of a pair is
/// addressed by the same seed as its sketch. Every candidate referenced by
/// an active bit is restored; a collided bit therefore contributes **all**
/// of its referenced tokens (probabilistic restoration — no TF filtering).
pub fn materialize(pairs: &[(&HLLSet, &LutIndex)]) -> BTreeSet<Vec<u8>> {
    let mut out = BTreeSet::new();
    for (hllset, lut) in pairs {
        for addr in hllset.bit_addresses() {
            out.extend(lut.fiber(addr.bit()));
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ingest::Ingest;
    use hllset_contracts::{token_in_bytes, token_in_bytes_le, BitAddress};

    #[test]
    fn collided_bit_keeps_all_candidates() {
        let mut ingest = Ingest::new();
        // tid262LE and tid48300LE collide at (759, 0) under seed 0.
        let a = token_in_bytes_le(262).to_vec();
        let b = token_in_bytes_le(48_300).to_vec();
        ingest.ingest_token(&a);
        ingest.ingest_token(&a); // TF(a) = 2
        ingest.ingest_token(&b); // TF(b) = 1

        let mut sketch = hllset_core::HLLSet::new();
        sketch.add_bit(759 * 32 + 0);

        // No TF filtering: both references to the collided bit survive.
        let restored = materialize(&[(&sketch, ingest.lut(0))]);
        assert_eq!(
            restored,
            BTreeSet::from([a.clone(), b.clone()]),
            "a collided bit restores every candidate"
        );
    }

    #[test]
    fn unambiguous_bit_restores_the_pointed_token() {
        let mut ingest = Ingest::new();
        let rare = token_in_bytes(1);
        let frequent = token_in_bytes(2);
        ingest.ingest_token(&rare);
        for _ in 0..100 {
            ingest.ingest_token(&frequent);
        }

        // A sketch containing only `rare`'s seed-0 atom restores `rare`.
        let mut sketch = hllset_core::HLLSet::new();
        let rare_bit = BitAddress::of_token_seeded(&rare, 0).bit();
        sketch.add_bit(rare_bit);

        let restored = materialize(&[(&sketch, ingest.lut(0))]);
        assert_eq!(restored, BTreeSet::from([rare]));
    }

    #[test]
    fn tokens_outside_the_luts_never_appear() {
        let mut ingest = Ingest::new();
        let in_lut = token_in_bytes(5);
        let outside = token_in_bytes(6);
        ingest.ingest_token(&in_lut);
        // `outside` gets a huge TF but is removed from all LUTs.
        for _ in 0..1000 {
            ingest.ingest_token(&outside);
        }
        // Rebuild a LUT containing only `in_lut`.
        let mut lut_only_in = ::hllset_lut::LutIndex::default();
        lut_only_in.insert_token(in_lut.clone());

        let mut sketch = hllset_core::HLLSet::new();
        let bit = BitAddress::of_token_seeded(&in_lut, 0).bit();
        sketch.add_bit(bit);

        let restored = materialize(&[(&sketch, &lut_only_in)]);
        assert_eq!(restored, BTreeSet::from([in_lut]), "TF cannot conjure tokens absent from the LUTs");
    }

    #[test]
    fn candidates_come_from_all_pointed_luts() {
        let ingest = Ingest::new();
        let t0 = token_in_bytes(10);
        let t1 = token_in_bytes(20);
        let t2 = token_in_bytes(30);

        // One sketch + LUT pair per encoding; each token appears in its own
        // encoding's LUT only.
        let mut s0 = hllset_core::HLLSet::new();
        s0.add_bit(BitAddress::of_token_seeded(&t0, 0).bit());
        let mut l0 = ::hllset_lut::LutIndex::default();
        l0.insert_token_seeded(t0.clone(), 0);

        let mut s1 = hllset_core::HLLSet::new();
        s1.add_bit(BitAddress::of_token_seeded(&t1, 1).bit());
        let mut l1 = ::hllset_lut::LutIndex::default();
        l1.insert_token_seeded(t1.clone(), 1);

        let mut s2 = hllset_core::HLLSet::new();
        s2.add_bit(BitAddress::of_token_seeded(&t2, 2).bit());
        let mut l2 = ::hllset_lut::LutIndex::default();
        l2.insert_token_seeded(t2.clone(), 2);

        // Only when all three LUTs are pointed are all three restored.
        let all = materialize(&[(&s0, &l0), (&s1, &l1), (&s2, &l2)]);
        assert_eq!(all, BTreeSet::from([t0.clone(), t1.clone(), t2.clone()]));
        let only_l0 = materialize(&[(&s0, &l0)]);
        assert_eq!(only_l0, BTreeSet::from([t0]));
    }
}
