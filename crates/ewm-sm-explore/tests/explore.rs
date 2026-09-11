//! ewm-sm-explore integration tests — the explorer projects the layers.

use ewm_app::{StateCache, StateMachine};
use ewm_git::{LatticeState, LooseStore, MemoryStore, Repository};

fn temp_dir(name: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "ewm-sm-explore-test-{}-{}-{}",
        std::process::id(),
        name,
        std::thread::current().name().unwrap_or("main")
    ));
    let _ = std::fs::remove_dir_all(&dir);
    dir
}

#[test]
fn store_render_shows_tip_lattice_tops_and_commits() {
    let dir = temp_dir("store");
    let mut app = StateMachine::new(LooseStore::new(&dir));
    let mut cache = StateCache::empty();
    app.run_turn(&mut cache, &[10u32, 20, 30]).expect("turn 1");
    app.run_turn(&mut cache, &[20u32, 30, 40]).expect("turn 2");

    let repo = Repository::open(LooseStore::new(&dir));
    let text = ewm_sm_explore::render_store(&repo, dir.to_str().unwrap());

    assert!(text.contains("persistent layer"), "text = {text}");
    assert!(text.contains("tip   :"), "text = {text}");
    assert!(text.contains("lattice tops:"), "text = {text}");
    assert!(text.contains("G1 bits"), "text = {text}");
    assert!(text.contains("hllsetLUT (repo registry"), "text = {text}");
    assert!(text.contains("commits (tip-first):"), "text = {text}");
    assert!(text.contains("#0 ") && text.contains("#1 "), "two commits shown: {text}");
    assert!(text.contains("G1=h:"), "Gx keys shown: {text}");
    assert!(text.contains("D="), "D/R/N shown: {text}");

    let doc = ewm_sm_explore::store_json(&repo, "test-store");
    assert_eq!(doc["commits"].as_array().unwrap().len(), 2);
    assert!(doc["commits"][0]["gx_keys"]["G1"].as_str().is_some());
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn snapshot_render_shows_st_area_and_cache_stubs() {
    let mut app = StateMachine::new(MemoryStore::default());
    let mut cache = StateCache::empty();
    app.run_turn(&mut cache, &[1u32, 2, 3]).expect("turn 1");

    let snap = app.snapshot(&cache);
    let json = snap.to_json();
    let parsed: ewm_app::StateSnapshot = serde_json::from_str(&json).expect("roundtrip");

    let text = ewm_sm_explore::render_snapshot(&parsed);
    assert!(text.contains("S(t) run-time"), "text = {text}");
    assert!(text.contains("working     : G1="), "text = {text}");
    assert!(text.contains("tree        : root="), "text = {text}");
    assert!(text.contains("#0 ids=[1, 2, 3]"), "turn shown: {text}");
    assert!(text.contains("cache: designed (Arrow)"), "cache stub: {text}");
    assert!(text.contains("MANIFEST"), "batch names: {text}");
}

#[test]
fn store_snapshot_roundtrip_via_json_file() {
    // The ewm-app exports a snapshot file; the explorer reads it back.
    let dir = temp_dir("roundtrip");
    let mut app = StateMachine::new(LooseStore::new(&dir));
    let mut cache = StateCache::empty();
    app.run_turn(&mut cache, &[7u32, 8]).expect("turn");

    let snap = app.snapshot(&cache);
    let file = dir.join("snapshot.json");
    std::fs::write(&file, snap.to_json()).expect("write");

    let raw = std::fs::read_to_string(&file).expect("read");
    let parsed: ewm_app::StateSnapshot = serde_json::from_str(&raw).expect("parse");
    assert_eq!(parsed, snap, "snapshot survives the JSON roundtrip");

    let text = ewm_sm_explore::render_snapshot(&parsed);
    assert!(text.contains("tip         :"), "text = {text}");
    let _ = std::fs::remove_dir_all(&dir);
}
