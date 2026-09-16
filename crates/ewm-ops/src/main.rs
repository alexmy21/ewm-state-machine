//! `ewm-ops` CLI — boot the operational graph.
//!
//! ```bash
//! ewm-ops boot [--store DIR] [--boot FILE|CID] [--fires N] [--repo DIR]
//! ewm-ops log [--store DIR]           # the append-only boot log (JSONL)
//! ewm-ops list [--store DIR]          # distinct boots with their metadata
//! ewm-ops rollback [--store DIR] CID  # re-point latest at a known boot
//! ewm-ops prev [--store DIR]          # roll latest back to the previous boot
//! ```
//!
//! Boot sequence (the OS metaphor):
//! 1. open the boot store (default `~/.cache/ewm-ops`);
//! 2. load the latest boot file (content-addressed, `b:<sha1>`);
//! 3. compile it into the operational graph + its `v:<sha1>` vocabulary;
//! 4. pick up the persisted state — the value stack from the last run, whose
//!    top is the current state (falls back to the script's `stack` on first
//!    boot or when the boot file changed);
//! 5. run the dispatcher (to quiescence or the fire budget);
//! 6. commit the fire log into ewm-git if `--repo` is given, persist the new
//!    stack, append the boot-log entry, and print the boot report as JSON.

use std::path::PathBuf;

use ewm_git::{LooseStore, Repository};
use ewm_ops::{compile_boot, commit_fire_log, BootRecord, BootStore, Dispatcher};
use serde_json::json;

/// The built-in boot script used when the store has no latest yet.
const DEFAULT_BOOT: &str = "\
# default ewm-ops boot — an empty machine; define values, ops and links
# in a boot file and point the store at it with: ewm-ops boot --boot file
";

fn main() {
    if let Err(e) = run() {
        eprintln!("ewm-ops: {e}");
        std::process::exit(2);
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() || args.iter().any(|a| a == "--help" || a == "-h") {
        print_help();
        return Ok(());
    }
    // First positional is the subcommand; `--flag`-first stays backward
    // compatible with the original single-command CLI.
    let (cmd, rest): (&str, &[String]) = if args[0].starts_with("--") {
        ("boot", &args[..])
    } else {
        (args[0].as_str(), &args[1..])
    };
    match cmd {
        "boot" => cmd_boot(rest),
        "log" => cmd_log(rest),
        "list" => cmd_list(rest),
        "rollback" => cmd_rollback(rest),
        "prev" => cmd_prev(rest),
        other => Err(format!("unknown subcommand: {other}").into()),
    }
}

/// `--store DIR` (optional) from the remaining args.
fn store_dir(rest: &[String]) -> Result<PathBuf, Box<dyn std::error::Error>> {
    let mut dir = BootStore::default_dir();
    let mut i = 0;
    while i < rest.len() {
        match rest[i].as_str() {
            "--store" => {
                i += 1;
                dir = PathBuf::from(rest.get(i).cloned().ok_or("--store: missing value")?);
            }
            other => return Err(format!("unknown argument: {other}").into()),
        }
        i += 1;
    }
    Ok(dir)
}

fn now_secs() -> String {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs().to_string())
        .unwrap_or_else(|_| "0".to_string())
}

fn cmd_boot(rest: &[String]) -> Result<(), Box<dyn std::error::Error>> {
    let mut dir = BootStore::default_dir();
    let mut boot_arg: Option<String> = None;
    let mut fires = 0usize;
    let mut repo_path: Option<PathBuf> = None;

    let mut i = 0;
    while i < rest.len() {
        match rest[i].as_str() {
            "--store" => {
                i += 1;
                dir = PathBuf::from(rest.get(i).cloned().ok_or("--store: missing value")?);
            }
            "--boot" => {
                i += 1;
                boot_arg = Some(rest.get(i).cloned().ok_or("--boot: missing value")?);
            }
            "--fires" => {
                i += 1;
                fires = rest
                    .get(i)
                    .and_then(|v| v.parse().ok())
                    .ok_or("--fires: not an integer")?;
            }
            "--repo" => {
                i += 1;
                repo_path = Some(PathBuf::from(rest.get(i).cloned().ok_or("--repo: missing value")?));
            }
            other => return Err(format!("unknown argument: {other}").into()),
        }
        i += 1;
    }

    let store = BootStore::open(&dir)?;

    // 1–2. Boot file: install a new one (file or already-stored CID), then
    // resolve the latest.
    if let Some(boot) = &boot_arg {
        let cid = if boot.starts_with("b:") && std::path::Path::new(boot).extension().is_none() {
            boot.clone()
        } else {
            let text = std::fs::read_to_string(boot).map_err(|e| format!("{boot}: {e}"))?;
            store.put_boot(&text)?
        };
        store.set_latest(&cid)?;
    }
    let boot_cid = store.latest_cid()?.ok_or("no latest boot file — set one with --boot")?;
    let script = store
        .default_boot()?
        .unwrap_or_else(|| DEFAULT_BOOT.to_string());

    // 3. Compile the boot script into the graph + vocabulary.
    let program = compile_boot(&script)?;
    let vocab_cid = program.vocab.cid();

    // 4. Pick up the state: persisted stack if it belongs to this boot file,
    // otherwise the script's own stack (first boot / boot changed).
    let seed_stack: Vec<String> = match store.load_state()? {
        Some((state_boot, stack)) if state_boot == boot_cid => stack,
        _ => program.stack.clone(),
    };

    // 5. Run the dispatcher.
    let mut graph = program.graph.clone();
    let mut dispatcher = Dispatcher::new(&mut graph);
    dispatcher.seed_values(&seed_stack);
    let commit_point = if program.fires > 0 || fires > 0 {
        dispatcher.run_until(fires.max(program.fires), |_| false)?
    } else {
        dispatcher.run_until(usize::MAX, |_| false)?
    };
    let log = dispatcher.log().clone();

    // 6. Commit (optional) + persist state + boot log + report.
    let commits = match &repo_path {
        Some(path) => {
            let loose = LooseStore::new(path);
            let mut repo = Repository::new(loose);
            let ids = commit_fire_log(&mut repo, &graph, &log)
                .map_err(|e| format!("commit: {e}"))?;
            Some(ids.iter().map(|id| id.to_string()).collect::<Vec<_>>())
        }
        None => None,
    };

    let produced = log.produced_values();
    let new_stack: Vec<String> = if produced.is_empty() {
        seed_stack.clone()
    } else {
        produced.clone()
    };
    store.save_state(&boot_cid, &new_stack)?;
    store.log_boot(&BootRecord {
        boot: boot_cid.clone(),
        booted_at: now_secs(),
        state_top: new_stack.last().cloned(),
        fires: commit_point.fired,
        reason: format!("{:?}", commit_point.reason),
    })?;

    let report = json!({
        "boot": boot_cid,
        "store": store.dir(),
        "vocab": serde_json::to_value(&program.vocab)?,
        "vocab_cid": vocab_cid,
        "graph": serde_json::to_value(&graph)?,
        "seed_stack": seed_stack,
        "commit_point": serde_json::to_value(&commit_point)?,
        "fire_log": serde_json::to_value(&log)?,
        "stack": new_stack,
        "state": new_stack.last().cloned(),
        "commits": commits,
    });
    let mut stdout = std::io::stdout().lock();
    serde_json::to_writer_pretty(&mut stdout, &report)?;
    use std::io::Write;
    stdout.write_all(b"\n")?;
    Ok(())
}

fn cmd_log(rest: &[String]) -> Result<(), Box<dyn std::error::Error>> {
    let store = BootStore::open(store_dir(rest)?)?;
    let log = store.read_log()?;
    print_json(&serde_json::to_value(&log)?)
}

fn cmd_list(rest: &[String]) -> Result<(), Box<dyn std::error::Error>> {
    let store = BootStore::open(store_dir(rest)?)?;
    let log = store.read_log()?;
    let mut boots: Vec<serde_json::Value> = Vec::new();
    let mut seen = std::collections::BTreeSet::new();
    for rec in log.iter().rev() {
        if seen.insert(rec.boot.clone()) {
            boots.push(json!({
                "boot": rec.boot,
                "last_booted_at": rec.booted_at,
                "state_top": rec.state_top,
                "last_fires": rec.fires,
                "is_latest": store.latest_cid()?.as_deref() == Some(rec.boot.as_str()),
            }));
        }
    }
    boots.reverse();
    print_json(&serde_json::Value::Array(boots))
}

fn cmd_rollback(rest: &[String]) -> Result<(), Box<dyn std::error::Error>> {
    let mut cid: Option<String> = None;
    let mut dir = BootStore::default_dir();
    let mut i = 0;
    while i < rest.len() {
        match rest[i].as_str() {
            "--store" => {
                i += 1;
                dir = PathBuf::from(rest.get(i).cloned().ok_or("--store: missing value")?);
            }
            other if !other.starts_with("--") => cid = Some(other.to_string()),
            other => return Err(format!("unknown argument: {other}").into()),
        }
        i += 1;
    }
    let store = BootStore::open(&dir)?;
    let cid = cid.ok_or("rollback: missing boot CID")?;
    store.rollback(&cid)?;
    print_json(&json!({ "latest": cid }))
}

fn cmd_prev(rest: &[String]) -> Result<(), Box<dyn std::error::Error>> {
    let store = BootStore::open(store_dir(rest)?)?;
    let previous = store.rollback_previous()?;
    print_json(&json!({ "latest": previous }))
}

fn print_json(value: &serde_json::Value) -> Result<(), Box<dyn std::error::Error>> {
    let mut stdout = std::io::stdout().lock();
    serde_json::to_writer_pretty(&mut stdout, value)?;
    use std::io::Write;
    stdout.write_all(b"\n")?;
    Ok(())
}

fn print_help() {
    println!(
        "ewm-ops — boot the operational graph\n\
         \n\
         Usage:\n\
         \x20 ewm-ops boot [options]              boot (default subcommand)\n\
         \x20 ewm-ops log [--store DIR]           print the append-only boot log\n\
         \x20 ewm-ops list [--store DIR]          list distinct boots with metadata\n\
         \x20 ewm-ops rollback [--store DIR] CID  re-point latest at a known boot\n\
         \x20 ewm-ops prev [--store DIR]          roll latest back to the previous boot\n\
         \n\
         Boot options:\n\
         \x20 --store DIR    boot store directory (default ~/.cache/ewm-ops)\n\
         \x20 --boot F|CID   install a boot file (path or b:<sha1>) and set it latest\n\
         \x20 --fires N      fire budget (default: the script's `fires`, 0 = quiescence)\n\
         \x20 --repo DIR     commit the fire log into an ewm-git loose store\n\
         \n\
         Boot sequence: latest boot file -> compile (+ v:<sha1> vocabulary) ->\n\
         pick up state on top of the stack -> dispatch -> commit -> persist\n\
         state -> append boot log.\n\
         \n\
         Output: a single JSON report on stdout."
    );
}
