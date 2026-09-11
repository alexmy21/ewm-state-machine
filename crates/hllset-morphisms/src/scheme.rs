//! Bootstrap-scheme prefixes for token LUTs.
//!
//! n-grams and n-seeds are two ways of **bootstrapping token presentation**
//! in an HLLSet. The HLLSet itself is bootstrap-scheme agnostic, and there is
//! **one G1, one G2, one G3** — the same channels serve both schemes.
//!
//! **Bits are anonymous**: a bit does not remember which token or which
//! scheme set it. The token LUTs restore the origin of each bit in the
//! context of a specific HLLSet. The LUTs are kept **separate per scheme**,
//! and the scheme prefix lives on the **LUT name**, not on the Gn HLLSet key:
//!
//! ```text
//! G1/G2/G3         h:<sha1>    scheme-agnostic channel HLLSets (shared)
//! ng:G1 … ng:G3    n-gram LUTs — order can be restored (window chain)
//! ns:G1 … ns:G3    n-seed LUTs — plain set only (seeded hashes are orderless)
//! ```
//!
//! Given a recovered Gx, materialization picks the LUT whose name prefix
//! matches the requested bootstrap scheme.

/// The n-gram bootstrap prefix (order-preserving LUTs).
pub const NG: &str = "ng";

/// The n-seed bootstrap prefix (orderless LUTs).
pub const NS: &str = "ns";

/// The scheme-prefixed name of a Gn channel's LUT: `ng:G1`, `ns:G3`, …
///
/// `channel` is 0-based (0 = G1, 1 = G2, 2 = G3).
pub fn lut_name(scheme: &str, channel: usize) -> String {
    format!("{scheme}:G{}", channel + 1)
}

/// The scheme prefix of a LUT name (`"ng"` / `"ns"`), if the name is a
/// scheme-prefixed Gn LUT name.
pub fn lut_scheme(name: &str) -> Option<&str> {
    let (scheme, gn) = name.split_once(':')?;
    if (scheme == NG || scheme == NS) && matches!(gn, "G1" | "G2" | "G3") {
        Some(scheme)
    } else {
        None
    }
}

/// The 0-based channel index of a scheme-prefixed LUT name, if valid.
pub fn lut_channel(name: &str) -> Option<usize> {
    let (scheme, gn) = name.split_once(':')?;
    if scheme != NG && scheme != NS {
        return None;
    }
    match gn {
        "G1" => Some(0),
        "G2" => Some(1),
        "G3" => Some(2),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lut_names_carry_the_scheme_and_channel() {
        assert_eq!(lut_name(NG, 0), "ng:G1");
        assert_eq!(lut_name(NS, 2), "ns:G3");
        assert_eq!(lut_scheme("ng:G1"), Some(NG));
        assert_eq!(lut_scheme("ns:G2"), Some(NS));
        assert_eq!(lut_channel("ns:G2"), Some(1));
    }

    #[test]
    fn bare_or_malformed_names_are_rejected() {
        assert_eq!(lut_scheme("G1"), None);
        assert_eq!(lut_scheme("xx:G1"), None);
        assert_eq!(lut_channel("ng:G4"), None);
        assert_eq!(lut_channel("h:ng:abc"), None);
    }
}
