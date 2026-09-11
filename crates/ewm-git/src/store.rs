//! Object stores: in-memory (tests) and loose-file (2005-Git style).
//!
//! The loose store writes plain (uncompressed) files under
//! `objects/xx/yyyy…` — the original Git layout without zlib — and persists
//! `HEAD` as a file in the repository root.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use crate::object::{Object, ObjectId};

/// Store errors.
#[derive(Debug)]
pub enum StoreError {
    NotFound(ObjectId),
    Io(std::io::Error),
}

impl std::fmt::Display for StoreError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            StoreError::NotFound(id) => write!(f, "object not found: {id}"),
            StoreError::Io(e) => write!(f, "io error: {e}"),
        }
    }
}

impl std::error::Error for StoreError {}

pub type Result<T> = std::result::Result<T, StoreError>;

/// A content-addressed object store.
pub trait ObjectStore {
    fn put(&mut self, id: &ObjectId, object: &Object) -> Result<()>;
    fn get(&self, id: &ObjectId) -> Result<Object>;
    fn contains(&self, id: &ObjectId) -> bool;
    fn ids(&self) -> Vec<ObjectId>;
    fn len(&self) -> usize;

    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Remove an object (used by GC). Default: unsupported.
    fn delete(&mut self, _id: &ObjectId) -> Result<()> {
        Err(StoreError::Io(std::io::Error::new(
            std::io::ErrorKind::Unsupported,
            "delete unsupported",
        )))
    }

    /// Persist `HEAD` (no-op for stores without ref persistence).
    fn write_head(&self, _id: &ObjectId) -> Result<()> {
        Ok(())
    }

    /// Read the persisted `HEAD`, if any.
    fn read_head(&self) -> Option<ObjectId> {
        None
    }
}

// ── In-memory store ────────────────────────────────────────────────────────

#[derive(Clone, Debug, Default)]
pub struct MemoryStore {
    map: HashMap<ObjectId, Object>,
}

impl ObjectStore for MemoryStore {
    fn put(&mut self, id: &ObjectId, object: &Object) -> Result<()> {
        self.map.insert(id.clone(), object.clone());
        Ok(())
    }

    fn get(&self, id: &ObjectId) -> Result<Object> {
        self.map.get(id).cloned().ok_or_else(|| StoreError::NotFound(id.clone()))
    }

    fn contains(&self, id: &ObjectId) -> bool {
        self.map.contains_key(id)
    }

    fn ids(&self) -> Vec<ObjectId> {
        self.map.keys().cloned().collect()
    }

    fn len(&self) -> usize {
        self.map.len()
    }

    fn delete(&mut self, id: &ObjectId) -> Result<()> {
        self.map
            .remove(id)
            .map(|_| ())
            .ok_or_else(|| StoreError::NotFound(id.clone()))
    }
}

// ── Loose-file store (2005 Git layout, plain bytes) ─────────────────────────

/// `root/objects/xx/rest` with uncompressed bytes, plus `root/HEAD`.
#[derive(Clone, Debug)]
pub struct LooseStore {
    root: PathBuf,
}

impl Default for LooseStore {
    fn default() -> Self {
        Self::new(".")
    }
}

impl LooseStore {
    /// Use `root` as the repository directory (created on demand).
    pub fn new(root: impl AsRef<Path>) -> Self {
        Self {
            root: root.as_ref().to_path_buf(),
        }
    }

    /// The object directory path.
    pub fn objects_dir(&self) -> PathBuf {
        self.root.join("objects")
    }

    fn path_for(&self, id: &ObjectId) -> PathBuf {
        let (dir, rest) = id.as_str().split_at(2);
        self.objects_dir().join(dir).join(rest)
    }

    fn head_path(&self) -> PathBuf {
        self.root.join("HEAD")
    }
}

impl ObjectStore for LooseStore {
    fn put(&mut self, id: &ObjectId, object: &Object) -> Result<()> {
        let path = self.path_for(id);
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).map_err(StoreError::Io)?;
        }
        std::fs::write(&path, object.serialize()).map_err(StoreError::Io)
    }

    fn get(&self, id: &ObjectId) -> Result<Object> {
        let bytes = match std::fs::read(self.path_for(id)) {
            Ok(bytes) => bytes,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                return Err(StoreError::NotFound(id.clone()));
            }
            Err(e) => return Err(StoreError::Io(e)),
        };
        Object::deserialize(&bytes).ok_or_else(|| {
            StoreError::Io(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "corrupt object",
            ))
        })
    }

    fn contains(&self, id: &ObjectId) -> bool {
        self.path_for(id).is_file()
    }

    fn ids(&self) -> Vec<ObjectId> {
        let mut ids = Vec::new();
        let Ok(entries) = std::fs::read_dir(self.objects_dir()) else {
            return ids;
        };
        for entry in entries.flatten() {
            let Ok(rest) = std::fs::read_dir(entry.path()) else {
                continue;
            };
            for f in rest.flatten() {
                let name = f.file_name();
                let name = name.to_string_lossy();
                let full = format!("{}{}", entry.file_name().to_string_lossy(), name);
                ids.push(ObjectId(full));
            }
        }
        ids
    }

    fn len(&self) -> usize {
        self.ids().len()
    }

    fn delete(&mut self, id: &ObjectId) -> Result<()> {
        match std::fs::remove_file(self.path_for(id)) {
            Ok(()) => Ok(()),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                Err(StoreError::NotFound(id.clone()))
            }
            Err(e) => Err(StoreError::Io(e)),
        }
    }

    fn write_head(&self, id: &ObjectId) -> Result<()> {
        std::fs::write(self.head_path(), id.as_str()).map_err(StoreError::Io)
    }

    fn read_head(&self) -> Option<ObjectId> {
        let text = std::fs::read_to_string(self.head_path()).ok()?;
        let trimmed = text.trim();
        if trimmed.is_empty() {
            None
        } else {
            ObjectId::validated(trimmed.to_string())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn memory_store_roundtrip_and_delete() {
        let mut store = MemoryStore::default();
        let obj = Object::Blob(b"hello".to_vec());
        let id = obj.id();
        store.put(&id, &obj).unwrap();
        assert!(store.contains(&id));
        assert_eq!(store.get(&id).unwrap(), obj);
        store.delete(&id).unwrap();
        assert!(!store.contains(&id));
    }

    #[test]
    fn loose_store_roundtrip_with_head() {
        let dir = std::env::temp_dir().join(format!("ewm-git-test-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let mut store = LooseStore::new(&dir);
        let obj = Object::Blob(b"persisted".to_vec());
        let id = obj.id();
        store.put(&id, &obj).unwrap();
        store.write_head(&id).unwrap();

        let reload = LooseStore::new(&dir);
        assert!(reload.contains(&id));
        assert_eq!(reload.get(&id).unwrap().id(), id);
        assert_eq!(reload.read_head(), Some(id));
        let _ = std::fs::remove_dir_all(&dir);
    }
}
