//! Git objects: content-addressed blobs and commits.
//!
//! Serialization is deterministic and self-delimiting (length-prefixed), so
//! the SHA1 of the serialized bytes is the content address — the IICA
//! property the whole store relies on.

use std::fmt;

use hllset_core::core::hashing::sha1_hex;

/// A SHA1 content address (40 hex chars).
#[derive(Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Debug)]
pub struct ObjectId(pub String);

impl ObjectId {
    /// Content-address some bytes.
    pub fn of_bytes(data: &[u8]) -> Self {
        Self(sha1_hex(data))
    }

    /// Validate a 40-char hex string at a boundary.
    pub fn validated(s: String) -> Option<Self> {
        (s.len() == 40 && s.bytes().all(|b| b.is_ascii_hexdigit()))
            .then_some(Self(s))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }

    /// First 8 hex chars, for human-readable logs.
    pub fn short(&self) -> &str {
        &self.0[..8.min(self.0.len())]
    }
}

impl fmt::Display for ObjectId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.short())
    }
}

/// The global-state channels.
///
/// `G1` = bits mapped from 1-gram / seed-0, `G2` = 2-gram / seed-1,
/// `G3` = 3-gram / seed-2. n-grams and seeds are interchangeable — they are
/// just bits, and it does not matter where they come from. All three are
/// commit-linked (one commit carries all three snapshots), which is what
/// enables time-travel projection.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Gx {
    G1,
    G2,
    G3,
}

impl Gx {
    pub const ALL: [Gx; 3] = [Gx::G1, Gx::G2, Gx::G3];

    /// Channel index (0..3).
    pub fn index(self) -> usize {
        match self {
            Gx::G1 => 0,
            Gx::G2 => 1,
            Gx::G3 => 2,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Gx::G1 => "G1",
            Gx::G2 => "G2",
            Gx::G3 => "G3",
        }
    }
}

impl fmt::Display for Gx {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.name())
    }
}

/// A commit: three channel trees (G1/G2/G3 state blobs), the bit-TF vector
/// blob, parents, and a message.
///
/// Wall-clock time is deliberately absent: **the commit hash is the timer**.
#[derive(Clone, Debug, PartialEq)]
pub struct Commit {
    /// Content addresses of the G1, G2, G3 state blobs.
    pub trees: [ObjectId; 3],
    /// Content address of the bit-TF vector (32K TF over the lattice top).
    pub tf: ObjectId,
    /// Parent commits (empty for the root, several for merges).
    pub parents: Vec<ObjectId>,
    pub message: String,
}

impl Commit {
    /// Deterministic serialization.
    pub fn serialize(&self) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(b"commit");
        out.push(0);
        for tree in &self.trees {
            out.extend_from_slice(tree.0.as_bytes());
        }
        out.extend_from_slice(self.tf.0.as_bytes());
        out.push(0);
        out.extend_from_slice(&(self.parents.len() as u32).to_le_bytes());
        for p in &self.parents {
            out.extend_from_slice(p.0.as_bytes());
        }
        out.extend_from_slice(&(self.message.len() as u32).to_le_bytes());
        out.extend_from_slice(self.message.as_bytes());
        out
    }

    /// Inverse of [`Self::serialize`].
    pub fn deserialize(bytes: &[u8]) -> Option<Self> {
        let mut pos = 0;
        let mut take = |n: usize| -> Option<&[u8]> {
            if pos + n <= bytes.len() {
                let s = &bytes[pos..pos + n];
                pos += n;
                Some(s)
            } else {
                None
            }
        };
        if take(6)? != b"commit" || take(1)? != [0] {
            return None;
        }
        let mut trees = Vec::with_capacity(3);
        for _ in 0..3 {
            trees.push(ObjectId::validated(String::from_utf8(take(40)?.to_vec()).ok()?)?);
        }
        let trees: [ObjectId; 3] = trees.try_into().ok()?;
        let tf = ObjectId::validated(String::from_utf8(take(40)?.to_vec()).ok()?)?;
        if take(1)? != [0] {
            return None;
        }
        let n = u32::from_le_bytes(take(4)?.try_into().ok()?) as usize;
        let mut parents = Vec::with_capacity(n);
        for _ in 0..n {
            parents.push(ObjectId::validated(String::from_utf8(take(40)?.to_vec()).ok()?)?);
        }
        let mlen = u32::from_le_bytes(take(4)?.try_into().ok()?) as usize;
        let message = String::from_utf8(take(mlen)?.to_vec()).ok()?;
        Some(Commit {
            trees,
            tf,
            parents,
            message,
        })
    }

    /// Content address of this commit object.
    pub fn id(&self) -> ObjectId {
        ObjectId::of_bytes(&self.serialize())
    }
}

/// A stored object.
#[derive(Clone, Debug, PartialEq)]
pub enum Object {
    /// A serialized HLLSet (one channel's commit state).
    Blob(Vec<u8>),
    Commit(Box<Commit>),
}

impl Object {
    /// Deterministic serialization with a type tag.
    pub fn serialize(&self) -> Vec<u8> {
        let mut out = Vec::new();
        match self {
            Object::Blob(data) => {
                out.extend_from_slice(b"blob");
                out.push(0);
                out.extend_from_slice(&(data.len() as u32).to_le_bytes());
                out.extend_from_slice(data);
            }
            Object::Commit(commit) => out.extend_from_slice(&commit.serialize()),
        }
        out
    }

    /// Inverse of [`Self::serialize`].
    pub fn deserialize(bytes: &[u8]) -> Option<Self> {
        if bytes.starts_with(b"commit") {
            return Commit::deserialize(bytes).map(|c| Object::Commit(Box::new(c)));
        }
        let rest = bytes.strip_prefix(b"blob")?;
        let rest = rest.strip_prefix(&[0])?;
        let len = u32::from_le_bytes(rest.get(0..4)?.try_into().ok()?) as usize;
        Some(Object::Blob(rest.get(4..4 + len)?.to_vec()))
    }

    /// Content address of this object.
    pub fn id(&self) -> ObjectId {
        ObjectId::of_bytes(&self.serialize())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn commit_serialization_roundtrip_with_three_trees() {
        let commit = Commit {
            trees: [
                ObjectId::of_bytes(b"g1"),
                ObjectId::of_bytes(b"g2"),
                ObjectId::of_bytes(b"g3"),
            ],
            tf: ObjectId::of_bytes(b"tf"),
            parents: vec![ObjectId::of_bytes(b"p1"), ObjectId::of_bytes(b"p2")],
            message: "merge".into(),
        };
        let bytes = commit.serialize();
        assert_eq!(Commit::deserialize(&bytes), Some(commit.clone()));
        assert_eq!(commit.id(), ObjectId::of_bytes(&bytes));
    }

    #[test]
    fn blob_serialization_roundtrip() {
        let blob = Object::Blob(vec![1, 2, 3, 4]);
        let bytes = blob.serialize();
        assert_eq!(Object::deserialize(&bytes), Some(blob));
    }

    #[test]
    fn gx_channels_index() {
        assert_eq!(Gx::G1.index(), 0);
        assert_eq!(Gx::G2.index(), 1);
        assert_eq!(Gx::G3.index(), 2);
        assert_eq!(Gx::ALL.len(), 3);
    }
}
