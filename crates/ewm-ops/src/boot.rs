//! The boot store — the "default IPFS location" of the operational graph.
//!
//! A boot file is a [`crate::dsl`] script addressed by the SHA1 of its text
//! (`b:<sha1>`), stored as a blob. A `default` pointer names the boot file
//! the machine starts from, and a `state` file holds the persisted value
//! stack (bottom → top; the top is the current state) plus the boot CID it
//! was produced under. Persistence is optional: if the directory has
//! nothing, the machine boots an empty script.

use std::fs;
use std::io;
use std::path::{Path, PathBuf};

use hllset_contracts::sha1_hex;

use crate::graph::ValueCid;

/// The content-addressed id of a boot file: `b:<sha1 of the script text>`.
pub fn boot_cid(text: &str) -> String {
    format!("b:{}", sha1_hex(text.as_bytes()))
}

/// A directory-backed boot store: blobs by CID, a default pointer, and the
/// persisted stack state.
pub struct BootStore {
    dir: PathBuf,
}

impl BootStore {
    /// The default boot location: `~/.cache/ewm-ops`.
    pub fn default_dir() -> PathBuf {
        std::env::var_os("HOME")
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from("."))
            .join(".cache")
            .join("ewm-ops")
    }

    pub fn open(dir: impl AsRef<Path>) -> io::Result<Self> {
        let dir = dir.as_ref().to_path_buf();
        fs::create_dir_all(dir.join("blobs"))?;
        Ok(Self { dir })
    }

    pub fn dir(&self) -> &Path {
        &self.dir
    }

    /// Store a boot script as a content-addressed blob; returns its CID.
    pub fn put_boot(&self, text: &str) -> io::Result<String> {
        let cid = boot_cid(text);
        fs::write(self.dir.join("blobs").join(&cid), text)?;
        Ok(cid)
    }

    /// Point the default boot at a CID (the boot-file analog of HEAD).
    pub fn set_default(&self, cid: &str) -> io::Result<()> {
        fs::write(self.dir.join("default"), cid)
    }

    /// The default boot CID, if one is set.
    pub fn default_cid(&self) -> io::Result<Option<String>> {
        let ptr = self.dir.join("default");
        if !ptr.exists() {
            return Ok(None);
        }
        let cid = fs::read_to_string(ptr)?;
        let cid = cid.trim().to_string();
        if cid.is_empty() || !self.dir.join("blobs").join(&cid).exists() {
            return Ok(None);
        }
        Ok(Some(cid))
    }

    /// Load the default boot script text.
    pub fn default_boot(&self) -> io::Result<Option<String>> {
        match self.default_cid()? {
            Some(cid) => Ok(Some(fs::read_to_string(
                self.dir.join("blobs").join(&cid),
            )?)),
            None => Ok(None),
        }
    }

    /// Persist the value stack (bottom → top) with the boot CID it was
    /// produced under. The top of the stack is the current state.
    pub fn save_state(&self, boot: &str, stack: &[ValueCid]) -> io::Result<()> {
        let mut text = String::from(boot);
        for cid in stack {
            text.push('\n');
            text.push_str(cid);
        }
        fs::write(self.dir.join("state"), text)
    }

    /// Load the persisted state: `(boot cid, stack bottom → top)`.
    pub fn load_state(&self) -> io::Result<Option<(String, Vec<ValueCid>)>> {
        let p = self.dir.join("state");
        if !p.exists() {
            return Ok(None);
        }
        let text = fs::read_to_string(p)?;
        let mut lines = text.lines().map(|l| l.trim().to_string());
        let boot = match lines.next() {
            Some(b) if !b.is_empty() => b,
            _ => return Ok(None),
        };
        let stack: Vec<ValueCid> = lines.filter(|l| !l.is_empty()).collect();
        Ok(Some((boot, stack)))
    }
}
