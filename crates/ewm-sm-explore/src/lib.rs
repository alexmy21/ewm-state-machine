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
use ewm_git::{view, Gx, ObjectId, ObjectStore, Repository};

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
                let g1_bits = repo
                    .state(cid)
                    .map(|hll| hll.popcount().to_string())
                    .unwrap_or_else(|_| "?".to_string());
                let commit = repo
                    .read_commit(cid)
                    .map(|c| {
                        let parents: Vec<String> = c.parents.iter().map(|p| short(&p.to_string(), 8)).collect();
                        format!(
                            "  #{n} {}  parents=[{}]\n      message: {}\n      G1 bits = {g1_bits}  (monotone gate; candidates available here)\n      G1={} G2={} G3={}\n      G1 view: {v}",
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
    out.push_str(&format!(
        "ring        : window={}/{} dimension={}\n",
        snap.ring.window_len, snap.ring.capacity, snap.ring.dimension
    ));

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

/// Per-user projection of one commit: the codebook gate `G_u` intersected
/// with the commit's shared lattice `c(t) = G1 ∪ G2 ∪ G3` (lab notebook 11).
///
/// The store commits unordered token collections, so the gate-HLLSet is the
/// **n-seed projection** `G1 ∪ G2 ∪ G3` over seeds 0/1/2 — the same shared
/// channels the n-gram scheme uses for ordered streams (1-gram/seed-0 → G1,
/// 2-gram/seed-1 → G2, 3-gram/seed-2 → G3); a Gx-HLLSet is one channel only.
/// The scheme mark is explicit in the keys: the catalog gate and its gated
/// projection carry `c:<sha1>` (n-seed), the commit lattice carries `h:<sha1>`
/// (n-gram), mirroring the `ns:`/`ng:` LUT names.
/// The materializer runs against the LUT rebuilt from the codebook hash list
/// — hashes ↔ tokens are isomorphic under the shared soldered rule, so the
/// per-user LUT degrades to the token/hash list; when a persisted shared
/// reverse LUT lands (the Arrow cache), the same call runs against it.
/// Returns `gated_pop`/`gated_key` (`c(t) ∩ G_u`) and the restored tokens.
pub fn project_user_json<S: ObjectStore>(
    repo: &Repository<S>,
    user: &str,
    cid: &ObjectId,
    codebook: &[String],
) -> Result<serde_json::Value, String> {
    // Flat n-seed gate: G1/G2/G3 = seed-0/1/2 token bits, matching the store.
    let mut gate_ing = hllset_morphisms::Ingest::new();
    gate_ing.ingest_tokens(codebook.iter().map(|t| t.as_bytes()));
    let gate_hll = gate_ing.projection();
    let codebook_set: std::collections::BTreeSet<Vec<u8>> =
        codebook.iter().map(|t| t.as_bytes().to_vec()).collect();

    let states = repo.states(cid).map_err(|e| e.to_string())?;
    let shared = states[0].union(&states[1]).union(&states[2]);
    let gated = hllset_morphisms::gate(&gate_hll, &shared);

    // Materialize against the gate's own per-seed LUTs (derived from the
    // hash list), filtered by the codebook — exact per-user restoration up
    // to hash collisions.
    let restored: Vec<String> = hllset_morphisms::ingest_gated_set(
        &gate_ing,
        &gated,
        Some(&codebook_set),
    )
    .into_iter()
    .map(|v| String::from_utf8_lossy(&v).into_owned())
    .collect();

    Ok(serde_json::json!({
        "user": user,
        "commit": cid.as_str(),
        "commit_pop": shared.popcount(),
        "commit_key": shared.content_key(),
        "gate_pop": gate_hll.popcount(),
        "gate_key": gate_ing.key(),
        "gated_pop": gated.popcount(),
        "gated_key": gated.content_key_c(),
        "n_restored": restored.len(),
        "restored_tokens": restored,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ewm_git::{LatticeState, LooseStore, MemoryStore, Repository};

    #[test]
    fn project_user_gates_the_commit_lattice() {
        let mut repo = Repository::new(MemoryStore::default());

        // User A's codebook, and a commit whose shared lattice mixes A and B
        // tokens — the gated projection keeps only the A-component.
        let codebook: Vec<String> = ["fin_a", "fin_b", "fin_c", "fin_d"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        let state_a = LatticeState::single(&hllset_core::HLLSet::from_tokens(
            ["fin_a", "fin_b"].iter(),
        ));
        let commit = repo.commit(&state_a, &[], "user A").unwrap();

        let doc = project_user_json(&repo, "acme-finance", &commit, &codebook).unwrap();
        let gate_pop = doc["gate_pop"].as_u64().unwrap();
        let gated_pop = doc["gated_pop"].as_u64().unwrap();
        assert!(gate_pop >= gated_pop, "gated bits are a subset of the gate");
        assert!(gated_pop > 0, "the commit carries some of the user's bits");
        assert_eq!(doc["restored_tokens"], serde_json::json!(["fin_a", "fin_b"]));

        // The gated key is stable: same projection twice, same address.
        let doc2 = project_user_json(&repo, "acme-finance", &commit, &codebook).unwrap();
        assert_eq!(doc["gated_key"], doc2["gated_key"]);
    }

    #[test]
    fn project_user_reads_a_loose_store() {
        // Exercise the same path the CLI uses: a persisted LooseStore.
        let root = std::env::temp_dir().join(format!(
            "ewm-project-user-smoke-{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&root);
        let mut repo = Repository::new(LooseStore::new(&root));

        let mixed = LatticeState::single(&hllset_core::HLLSet::from_tokens(
            ["fin_a", "fin_b", "med_x"].iter(),
        ));
        let commit = repo.commit(&mixed, &[], "mixed").unwrap();

        let codebook: Vec<String> = ["fin_a", "fin_b", "fin_c"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        let doc = project_user_json(&repo, "acme-finance", &commit, &codebook).unwrap();
        assert_eq!(doc["user"], "acme-finance");
        assert!(doc["gated_pop"].as_u64().unwrap() > 0);
        // The foreign med_x is outside the gate and never restored.
        assert_eq!(doc["restored_tokens"], serde_json::json!(["fin_a", "fin_b"]));

        let _ = std::fs::remove_dir_all(&root);
    }
}

