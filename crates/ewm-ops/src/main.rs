//! `ewm-ops` CLI — boot the operational graph.
//!
//! ```bash
//! ewm-ops boot [--store DIR] [--boot FILE|CID] [--fires N] [--repo DIR]
//! ```
//!
//! Boot sequence (the OS metaphor):
//! 1. open the boot store (default `~/.cache/ewm-ops`);
//! 2. load the default boot file (content-addressed, `b:<sha1>`);
//! 3. compile it into the operational graph;
//! 4. pick up the persisted state — the value stack from the last run, whose
//!    top is the current state (falls back to the script's `stack` on first
//!    boot or when the boot file changed);
//! 5. run the dispatcher (to quiescence or the fire budget);
//! 6. commit the fire log into ewm-git if `--repo` is given, persist the new
//!    stack, and print the boot report as JSON.

use std::path::PathBuf;

use ewm_git::{LooseStore, Repository};
use ewm_ops::{compile_boot, commit_fire_log, BootStore, Dispatcher};
use serde_json::json;

/// The built-in boot script used when the store has no default yet.
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
    if args.iter().any(|a| a == "--help" || a == "-h") {
        print_help();
        return Ok(());
    }

    let mut store_dir = BootStore::default_dir();
    let mut boot_arg: Option<String> = None;
    let mut fires = 0usize;
    let mut repo_path: Option<PathBuf> = None;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--store" => {
                i += 1;
                store_dir = args.get(i).cloned().map(PathBuf::from).ok_or("--store: missing value")?;
            }
            "--boot" => {
                i += 1;
                boot_arg = Some(args.get(i).cloned().ok_or("--boot: missing value")?);
            }
            "--fires" => {
                i += 1;
                fires = args.get(i).and_then(|v| v.parse().ok()).ok_or("--fires: not an integer")?;
            }
            "--repo" => {
                i += 1;
                repo_path = Some(PathBuf::from(args.get(i).cloned().ok_or("--repo: missing value")?));
            }
            other => return Err(format!("unknown argument: {other}").into()),
        }
        i += 1;
    }

    let store = BootStore::open(&store_dir)?;

    // 1–2. Boot file: install a new one (file or already-stored CID), then
    // resolve the default.
    if let Some(boot) = &boot_arg {
        let cid = if boot.starts_with("b:") && std::path::Path::new(boot).extension().is_none() {
            boot.clone()
        } else {
            let text = std::fs::read_to_string(boot).map_err(|e| format!("{boot}: {e}"))?;
            store.put_boot(&text)?
        };
        store.set_default(&cid)?;
    }
    let boot_cid = store.default_cid()?.ok_or("no default boot file — set one with --boot")?;
    let script = store
        .default_boot()?
        .unwrap_or_else(|| DEFAULT_BOOT.to_string());

    // 3. Compile the boot script into the graph.
    let program = compile_boot(&script)?;

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

    // 6. Commit (optional) + persist state + report.
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

    let report = json!({
        "boot": boot_cid,
        "store": store.dir(),
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

fn print_help() {
    println!(
        "ewm-ops — boot the operational graph\n\
         \n\
         Usage:\n\
         \x20 ewm-ops boot [options]\n\
         \n\
         Options:\n\
         \x20 --store DIR    boot store directory (default ~/.cache/ewm-ops)\n\
         \x20 --boot F|CID   install a boot file (path or b:<sha1>) and set it default\n\
         \x20 --fires N      fire budget (default: the script's `fires`, 0 = quiescence)\n\
         \x20 --repo DIR     commit the fire log into an ewm-git loose store\n\
         \n\
         Boot sequence: default boot file -> compile -> pick up state on top of\n\
         the stack -> dispatch -> commit -> persist state.\n\
         \n\
         Output: a single JSON boot report on stdout."
    );
}
