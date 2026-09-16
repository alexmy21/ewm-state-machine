//! The boot store — the "default IPFS location" of the operational graph.
//!
//! A boot file is a [`crate::dsl`] script addressed by the SHA1 of its text
//! (`b:<sha1>`), stored as a blob. The `latest` pointer names the boot file
//! the machine starts from, and a `state` file holds the persisted value
//! stack (bottom → top; the top is the current state) plus the boot CID it
//! was produced under. Persistence is optional: if the directory has
//! nothing, the machine boots an empty script.
//!
//! OS-like extras:
//!
//! - **boot log** (`boot.log`) — append-only JSONL: every boot records its
//!   boot CID, timestamp, state top, fire count, and commit-point reason;
//! - **latest-pointer semantics** — `latest` is a mutable pointer into the
//!   content-addressed blob store; [`BootStore::rollback`] re-points it at
//!   any known boot, and [`BootStore::rollback_previous`] re-points it at
//!   the previous boot in the log (the "boot previous known-good" path).

use std::fs;
use std::io;
use std::path::{Path, PathBuf};

use hllset_contracts::sha1_hex;
use serde::{Deserialize, Serialize};

use crate::graph::ValueCid;

/// The content-addressed id of a boot file: `b:<sha1 of the script text>`.
pub fn boot_cid(text: &str) -> String {
    format!("b:{}", sha1_hex(text.as_bytes()))
}

/// One boot-log entry (append-only JSONL).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct BootRecord {
    /// The boot-file CID that was booted.
    pub boot: String,
    /// Unix seconds of the boot.
    pub booted_at: String,
    /// The top of the persisted stack after the boot (the current state).
    pub state_top: Option<ValueCid>,
    /// How many [UM]s fired during the boot.
    pub fires: usize,
    /// The dispatcher stop reason (`Quiescence` / `Predicate` / `FireBudget`).
    pub reason: String,
}

/// A directory-backed boot store: blobs by CID, a latest pointer, the
/// persisted stack state, and the append-only boot log.
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

    /// True when a boot blob exists for `cid`.
    pub fn has_boot(&self, cid: &str) -> bool {
        self.dir.join("blobs").join(cid).exists()
    }

    /// Point the latest pointer at a boot CID (the boot-file analog of
    /// HEAD). Returns `NotFound` unless the blob exists — the pointer may
    /// only reference content that is already in the store.
    pub fn set_latest(&self, cid: &str) -> io::Result<()> {
        if !self.has_boot(cid) {
            return Err(io::Error::new(
                io::ErrorKind::NotFound,
                format!("no boot blob for {cid}"),
            ));
        }
        fs::write(self.dir.join("latest"), cid)
    }

    /// Backward-compatible alias of [`set_latest`](Self::set_latest).
    pub fn set_default(&self, cid: &str) -> io::Result<()> {
        self.set_latest(cid)
    }

    /// The latest boot CID, if one is set and resolvable. Falls back to the
    /// pre-vocabulary `default` pointer file for backward compatibility.
    pub fn latest_cid(&self) -> io::Result<Option<String>> {
        for name in ["latest", "default"] {
            let ptr = self.dir.join(name);
            if !ptr.exists() {
                continue;
            }
            let cid = fs::read_to_string(ptr)?;
            let cid = cid.trim().to_string();
            if cid.is_empty() || !self.has_boot(&cid) {
                continue;
            }
            return Ok(Some(cid));
        }
        Ok(None)
    }

    /// Backward-compatible alias of [`latest_cid`](Self::latest_cid).
    pub fn default_cid(&self) -> io::Result<Option<String>> {
        self.latest_cid()
    }

    /// Load the latest boot script text.
    pub fn default_boot(&self) -> io::Result<Option<String>> {
        match self.latest_cid()? {
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

    // ── boot log ────────────────────────────────────────────────────────

    /// Append one boot record to the append-only `boot.log` (JSONL).
    pub fn log_boot(&self, record: &BootRecord) -> io::Result<()> {
        use std::io::Write;
        let mut f = fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(self.dir.join("boot.log"))?;
        serde_json::to_writer(&mut f, record)?;
        f.write_all(b"\n")?;
        Ok(())
    }

    /// Read the boot log (oldest first).
    pub fn read_log(&self) -> io::Result<Vec<BootRecord>> {
        let p = self.dir.join("boot.log");
        if !p.exists() {
            return Ok(Vec::new());
        }
        let text = fs::read_to_string(p)?;
        let mut out = Vec::new();
        for line in text.lines() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            if let Ok(rec) = serde_json::from_str::<BootRecord>(line) {
                out.push(rec);
            }
        }
        Ok(out)
    }

    // ── latest-pointer semantics ────────────────────────────────────────

    /// Re-point `latest` at a known boot CID (validated). This is the
    /// rollback primitive: content stays immutable, only the pointer moves.
    pub fn rollback(&self, cid: &str) -> io::Result<()> {
        self.set_latest(cid)
    }

    /// Roll `latest` back to the most recent boot in the log whose boot CID
    /// differs from the current latest. Returns the new latest CID, or
    /// `None` when there is no earlier boot to fall back to.
    pub fn rollback_previous(&self) -> io::Result<Option<String>> {
        let current = match self.latest_cid()? {
            Some(cid) => cid,
            None => return Ok(None),
        };
        for rec in self.read_log()?.iter().rev() {
            if rec.boot != current && self.has_boot(&rec.boot) {
                self.rollback(&rec.boot)?;
                return Ok(Some(rec.boot.clone()));
            }
        }
        Ok(None)
    }
}
