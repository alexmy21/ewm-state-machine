//! `ewm-sm-explore` — the read-only explorer of the ewm-state-machine
//! data structure.
//!
//! The state machine is one data structure spread over three locations:
//!
//! ```text
//! 1. S(t) run-time  (ewm-app::StateCache)   — working set, tree, turns, tip
//! 2. cache          (designed, Arrow)        — LUTs, TF, hllsetLUT, Gx versions
//! 3. persistent     (ewm-git::Repository)    — commit DAG, blobs, HEAD, hllsetLUT
//! ```
//!
//! The explorer projects these locations read-only:
//!
//! - [`render_store`] — the persistent layer over an `ewm-git` repository;
//! - [`render_snapshot`] — the S(t) run-time area from a `StateSnapshot`;
//! - [`store_json`] — the persistent layer as JSON for tooling.

use ewm_app::StateSnapshot;
use ewm_git::{view, Gx, ObjectStore, Repository};

/// Render the persistent layer of an open repository, tip-first.
pub fn render_store<S: ObjectStore>(repo: &Repository<S>, store_label: &str) -> String {
    let mut out = String::new();
    out.push_str(&format!("=== ewm-state-machine explorer: persistent layer ===\n"));
    out.push_str(&format!("store : {store_label}\n"));
    out.push_str(&format!(
        "tip   : {}\n",
        repo.head()
            .map(|h| h.to_string())
            .unwrap_or_else(|| "<none>".to_string())
    ));
    out.push_str(&format!("context bits (G1 lattice top): {}\n", repo.context_size()));

    out.push_str("\nlattice tops:\n");
    for gx in Gx::ALL {
        out.push_str(&format!(
            "  {} bits = {}\n",
            gx.name(),
            repo.lattice_top(gx).popcount()
        ));
    }

    out.push_str("\nhllsetLUT (repo registry, ranked by TH):\n");
    let ranked = repo.hllset_lut().ranked();
    if ranked.is_empty() {
        out.push_str("  (empty)\n");
    } else {
        for (id, th) in ranked.iter().take(20) {
            out.push_str(&format!("  h:{}  TH = {th}\n", id.as_str()));
        }
        if ranked.len() > 20 {
            out.push_str(&format!("  ... {} more\n", ranked.len() - 20));
        }
    }

    out.push_str("\ncommits (tip-first):\n");
    match repo.log() {
        Ok(log) if log.is_empty() => out.push_str("  (no commits)\n"),
        Ok(log) => {
            for (n, cid) in log.iter().enumerate().rev() {
                let keys = repo
                    .state_keys(cid)
                    .map(|k| k.map(|s| short(&s, 12)))
                    .unwrap_or_else(|_| ["?".to_string(), "?".to_string(), "?".to_string()]);
                let v = view(repo, cid)
                    .map(|v| format!("D={} R={} N={}", v.departed.popcount(), v.retained.popcount(), v.new.popcount()))
                    .unwrap_or_else(|_| "unreadable".to_string());
                let commit = repo
                    .read_commit(cid)
                    .map(|c| {
                        let parents: Vec<String> = c.parents.iter().map(|p| short(&p.to_string(), 8)).collect();
                        format!(
                            "  #{n} {}  parents=[{}]\n      message: {}\n      G1={} G2={} G3={}\n      G1 view: {v}",
                            short(&cid.to_string(), 8),
                            parents.join(", "),
                            truncate(&c.message, 60),
                            keys[0],
                            keys[1],
                            keys[2],
                        )
                    })
                    .unwrap_or_else(|_| format!("  #{n} {} (unreadable)", short(&cid.to_string(), 8)));
                out.push_str(&commit);
                out.push('\n');
            }
        }
        Err(_) => out.push_str("  (log unreadable)\n"),
    }
    out
}

/// Render a `StateSnapshot` (the S(t) run-time area).
pub fn render_snapshot(snap: &StateSnapshot) -> String {
    let mut out = String::new();
    out.push_str(&format!("=== ewm-state-machine explorer: {} ===\n", snap.layer));
    out.push_str(&format!(
        "tip         : {}\n",
        snap.tip.as_deref().unwrap_or("<none>")
    ));
    out.push_str(&format!("turn_count  : {}\n", snap.turn_count));
    out.push_str(&format!(
        "working     : G1={} G2={} G3={} bits\n",
        snap.working_bits[0], snap.working_bits[1], snap.working_bits[2]
    ));
    out.push_str(&format!(
        "tree        : root={} leaves={}\n",
        short(&snap.tree_root, 12),
        snap.tree_leaves
    ));
    out.push_str(&format!("tf_base     : {} entries\n", snap.tf_base_entries));

    out.push_str("\nturns:\n");
    if snap.turns.is_empty() {
        out.push_str("  (no turns)\n");
    }
    for t in &snap.turns {
        out.push_str(&format!(
            "  #{} ids={:?} g1={} commit={}\n",
            t.turn,
            t.ids,
            short(&t.g1_key, 12),
            t.commit.as_deref().map(|c| short(c, 8)).unwrap_or_else(|| "-".to_string())
        ));
    }

    out.push_str(&format!("\ncache: {}\n", snap.cache.status));
    out.push_str(&format!("  batches: {}\n", snap.cache.batches.join(", ")));
    out
}

/// The persistent layer as JSON for tooling.
pub fn store_json<S: ObjectStore>(repo: &Repository<S>, store_label: &str) -> serde_json::Value {
    let commits = repo
        .log()
        .map(|log| {
            log.iter()
                .enumerate()
                .rev()
                .map(|(n, cid)| {
                    let keys = repo.state_keys(cid).unwrap_or_else(|_| {
                        ["?".to_string(), "?".to_string(), "?".to_string()]
                    });
                    let v = view(repo, cid);
                    serde_json::json!({
                        "index": n,
                        "id": cid.to_string(),
                        "message": repo.read_commit(cid).map(|c| c.message).unwrap_or_default(),
                        "parents": repo.read_commit(cid).map(|c| c.parents.iter().map(|p| p.to_string()).collect::<Vec<_>>()).unwrap_or_default(),
                        "gx_keys": { "G1": keys[0], "G2": keys[1], "G3": keys[2] },
                        "G1": v.map(|v| serde_json::json!({
                            "D": v.departed.popcount(),
                            "R": v.retained.popcount(),
                            "N": v.new.popcount(),
                        })).unwrap_or(serde_json::json!(null)),
                    })
                })
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();

    serde_json::json!({
        "store": store_label,
        "tip": repo.head().map(|h| h.to_string()),
        "context_bits": repo.context_size(),
        "lattice_tops": Gx::ALL.iter().map(|gx| serde_json::json!({
            "channel": gx.name(),
            "bits": repo.lattice_top(*gx).popcount(),
        })).collect::<Vec<_>>(),
        "hllset_lut": repo.hllset_lut().ranked().iter().map(|(id, th)| serde_json::json!({
            "id": format!("h:{}", id.as_str()),
            "th": th,
        })).collect::<Vec<_>>(),
        "commits": commits,
    })
}

fn short(s: &str, n: usize) -> String {
    if s.len() <= n {
        s.to_string()
    } else {
        format!("{}…", &s[..n])
    }
}

fn truncate(s: &str, n: usize) -> String {
    if s.chars().count() <= n {
        s.to_string()
    } else {
        let cut: String = s.chars().take(n).collect();
        format!("{cut}…")
    }
}
