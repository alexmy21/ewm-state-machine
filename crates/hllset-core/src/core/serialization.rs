//! Serialization utilities for HLLSet.
//!
//! Uses Roaring bitmap's native serialization for compact binary
//! representation. Also provides content hashing for content-addressing.

use crate::core::hashing::sha1_hex;
use crate::core::hllset::HLLSet;
use roaring::RoaringBitmap;

impl HLLSet {
    /// Serialize to bytes (Roaring bitmap format).
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(self.memory_usage());
        self.bitmap()
            .serialize_into(&mut bytes)
            .expect("serialize_into Vec<u8> is infallible");
        bytes
    }

    /// Deserialize from bytes.
    pub fn from_bytes(bytes: &[u8]) -> Option<Self> {
        RoaringBitmap::deserialize_from(bytes)
            .ok()
            .map(HLLSet::from_bitmap)
    }

    /// Generate SHA-1 content hash of the serialized HLLSet.
    ///
    /// The hash is deterministic — same bit pattern always produces
    /// the same hash. Used as the content-addressable identifier.
    pub fn content_hash(&self) -> String {
        sha1_hex(&self.to_bytes())
    }

    /// Generate content-addressable key: `h:<sha1>`.
    ///
    /// The `h:` prefix denotes a heterogeneous HLLSet (n-gram tokenized).
    pub fn content_key(&self) -> String {
        format!("h:{}", self.content_hash())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_roundtrip_bytes() {
        let mut h = HLLSet::new();
        h.add_token(b"hello");
        h.add_token(b"world");

        let bytes = h.to_bytes();
        let h2 = HLLSet::from_bytes(&bytes).unwrap();
        assert_eq!(h.popcount(), h2.popcount());
        assert_eq!(h.content_hash(), h2.content_hash());
    }

    #[test]
    fn test_content_key_prefix() {
        let mut h = HLLSet::new();
        h.add_token(b"test");
        let key = h.content_key();
        assert!(key.starts_with("h:"), "key = {key}");
        assert_eq!(key.len(), 42); // "h:" + 40 hex chars
    }

    #[test]
    fn test_content_key_deterministic() {
        let a = HLLSet::from_tokens(&["a", "b"]);
        let b = HLLSet::from_tokens(&["a", "b"]);
        assert_eq!(a.content_key(), b.content_key());
    }

    #[test]
    fn test_different_sets_different_keys() {
        let a = HLLSet::from_tokens(&["x"]);
        let b = HLLSet::from_tokens(&["y"]);
        assert_ne!(a.content_key(), b.content_key());
    }

    #[test]
    fn test_empty_set_roundtrip() {
        let h = HLLSet::new();
        let bytes = h.to_bytes();
        let h2 = HLLSet::from_bytes(&bytes).unwrap();
        assert!(h2.is_empty());
    }
}
