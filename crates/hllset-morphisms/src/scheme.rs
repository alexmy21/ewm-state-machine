//! Bootstrap-scheme prefixes on HLLSet keys.
//!
//! n-grams and n-seeds are two ways of **bootstrapping token presentation**
//! in an HLLSet. The HLLSet itself is bootstrap-scheme agnostic — both
//! schemes set bits in the same Gn channels (G1/G2/G3).
//!
//! The token LUTs, however, are kept **separate per scheme**: the n-gram LUT
//! stores n-gram window relations (order-preserving), the n-seed LUT stores
//! seeded-hash relations (orderless). The SHA1 prefix on a Gn key records
//! which scheme's LUT materialization must use:
//!
//! ```text
//! h:ng:<sha1>   n-gram bootstrapped  — order can be restored
//! h:ns:<sha1>   n-seed bootstrapped  — plain set only
//! ```

/// The n-gram bootstrap prefix (order-preserving LUT).
pub const NG: &str = "ng";

/// The n-seed bootstrap prefix (orderless LUT).
pub const NS: &str = "ns";

/// Build a scheme-prefixed HLLSet key from a scheme prefix and a bare SHA1.
pub fn scheme_key(prefix: &str, sha1: &str) -> String {
    format!("h:{prefix}:{sha1}")
}

/// The bootstrap scheme of a prefixed key (`"ng"` or `"ns"`), if any.
pub fn key_scheme(key: &str) -> Option<&str> {
    let rest = key.strip_prefix("h:")?;
    let (prefix, sha1) = rest.split_once(':')?;
    if sha1.len() == 40 && sha1.chars().all(|c| c.is_ascii_hexdigit()) {
        Some(prefix)
    } else {
        None
    }
}

/// The bare 40-hex SHA1 of a prefixed key, if the key is prefixed.
pub fn key_sha1(key: &str) -> Option<&str> {
    let rest = key.strip_prefix("h:")?;
    let (prefix, sha1) = rest.split_once(':')?;
    if prefix == NG || prefix == NS {
        (sha1.len() == 40 && sha1.chars().all(|c| c.is_ascii_hexdigit())).then_some(sha1)
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prefixed_keys_roundtrip() {
        let key = scheme_key(NG, "a".repeat(40).as_str());
        assert_eq!(key_scheme(&key), Some(NG));
        assert_eq!(key_sha1(&key), Some("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"));
    }

    #[test]
    fn bare_and_malformed_keys_are_rejected() {
        assert_eq!(key_scheme("h:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"), None);
        assert_eq!(key_scheme("h:xx:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"), Some("xx"));
        assert_eq!(key_sha1("not-a-key"), None);
    }
}
