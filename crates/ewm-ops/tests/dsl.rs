//! Tests for the boot-script DSL, the boot store, and the ewm-git commit
//! bridge.

use ewm_ops::{boot_cid, compile_boot, commit_fire_log, BootStore, Dispatcher};
use std::path::PathBuf;

fn tmp_dir(tag: &str) -> PathBuf {
    let mut p = std::env::temp_dir();
    p.push(format!(
        "ewm-ops-test-{tag}-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir_all(&p).unwrap();
    p
}

const FANOUT_BOOT: &str = "\
# a small fan-out graph
value a apple
value b banana cherry
def union2 ( 2 -- 1 ) union
def dup2 ( 1 -- 2 ) dup
link value:@a -> in:union2.0
link value:@b -> in:union2.1
link out:union2.0 -> in:dup2.0
stack @a @b
";

#[test]
fn boot_script_compiles_and_runs_a_fanout() {
    let program = compile_boot(FANOUT_BOOT).unwrap();
    assert_eq!(program.graph.ops.len(), 2);
    assert_eq!(program.graph.edges.len(), 3);
    assert_eq!(program.stack.len(), 2);

    let mut graph = program.graph.clone();
    let mut d = Dispatcher::new(&mut graph);
    d.seed_values(&program.stack);
    let log = d.run().unwrap().clone();

    assert_eq!(log.records.len(), 2, "union2 fires, then dup2");
    assert_eq!(log.records[0].outputs.len(), 1);
    assert_eq!(log.records[1].outputs.len(), 2);
    // dup2 consumed exactly the union output — fan-out by the same reference.
    assert_eq!(log.records[1].inputs, vec![log.records[0].outputs[0].clone()]);
    assert_eq!(log.produced_values().len(), 3);
}

#[test]
fn boot_script_resolves_call_by_name() {
    let script = "\
value a apple
value b banana
def union2 ( 2 -- 1 ) union
def caller ( 2 -- 1 ) call:union2
stack @a @b
";
    let program = compile_boot(script).unwrap();
    let caller = program
        .graph
        .ops
        .iter()
        .find(|(_, s)| s.expr.source.starts_with("2 1 call:p:"))
        .expect("caller resolves call:union2 to a program CID");
    assert!(caller.0.starts_with("p:"));
}

#[test]
fn boot_store_roundtrips_blobs_default_and_state() {
    let dir = tmp_dir("store");
    let store = BootStore::open(&dir).unwrap();

    let cid = store.put_boot(FANOUT_BOOT).unwrap();
    assert_eq!(cid, boot_cid(FANOUT_BOOT));
    assert!(cid.starts_with("b:"));

    store.set_default(&cid).unwrap();
    assert_eq!(store.default_cid().unwrap().as_deref(), Some(cid.as_str()));
    assert_eq!(store.default_boot().unwrap().as_deref(), Some(FANOUT_BOOT));

    let stack = vec!["h:aaa".to_string(), "h:bbb".to_string()];
    store.save_state(&cid, &stack).unwrap();
    let (boot, loaded) = store.load_state().unwrap().unwrap();
    assert_eq!(boot, cid);
    assert_eq!(loaded, stack);
    assert_eq!(loaded.last(), Some(&"h:bbb".to_string()), "top of stack = state");

    let _ = std::fs::remove_dir_all(&dir);
}

#[cfg(feature = "git")]
#[test]
fn commit_bridge_commits_every_record() {
    use ewm_git::{LooseStore, Repository};

    let program = compile_boot(FANOUT_BOOT).unwrap();
    let mut graph = program.graph.clone();
    let mut d = Dispatcher::new(&mut graph);
    d.seed_values(&program.stack);
    let log = d.run().unwrap().clone();

    let dir = tmp_dir("repo");
    let mut repo = Repository::new(LooseStore::new(&dir));
    let ids = commit_fire_log(&mut repo, &graph, &log).unwrap();
    assert_eq!(ids.len(), log.records.len());
    assert!(repo.head().is_some(), "HEAD advanced to the last commit");

    let _ = std::fs::remove_dir_all(&dir);
}
