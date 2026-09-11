//! `ewm-sm-explore` CLI — the read-only explorer of the ewm-state-machine.
//!
//! ```text
//! ewm-sm-explore store <path> [--json]      persistent layer (ewm-git)
//! ewm-sm-explore snapshot <file> [--json]   S(t) run-time area (StateSnapshot)
//! ewm-sm-explore help
//! ```

use std::process::ExitCode;

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match run(&args) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("ewm-sm-explore: {e}");
            eprintln!("usage: ewm-sm-explore store <path> [--json] | snapshot <file> [--json]");
            ExitCode::FAILURE
        }
    }
}

fn run(args: &[String]) -> Result<(), String> {
    if args.is_empty() || args[0] == "help" || args[0] == "--help" || args[0] == "-h" {
        print_usage();
        return Ok(());
    }

    match args[0].as_str() {
        "store" => {
            let path = args.get(1).ok_or("store: missing <path>")?;
            let json = args.get(2).map(|s| s == "--json").unwrap_or(false);
            cmd_store(path, json)
        }
        "snapshot" => {
            let file = args.get(1).ok_or("snapshot: missing <file>")?;
            let json = args.get(2).map(|s| s == "--json").unwrap_or(false);
            cmd_snapshot(file, json)
        }
        other => Err(format!("unknown command: {other}")),
    }
}

fn cmd_store(path: &str, json: bool) -> Result<(), String> {
    let repo = ewm_git::Repository::open(ewm_git::LooseStore::new(path));
    if json {
        let doc = ewm_sm_explore::store_json(&repo, path);
        println!("{}", serde_json::to_string_pretty(&doc).map_err(|e| e.to_string())?);
    } else {
        print!("{}", ewm_sm_explore::render_store(&repo, path));
    }
    Ok(())
}

fn cmd_snapshot(file: &str, json: bool) -> Result<(), String> {
    let raw = std::fs::read_to_string(file).map_err(|e| format!("{file}: {e}"))?;
    let snap: ewm_app::StateSnapshot =
        serde_json::from_str(&raw).map_err(|e| format!("{file}: {e}"))?;
    if json {
        println!("{}", snap.to_json());
    } else {
        print!("{}", ewm_sm_explore::render_snapshot(&snap));
    }
    Ok(())
}

fn print_usage() {
    println!(
        "ewm-sm-explore — read-only explorer of the ewm-state-machine

The state machine is one data structure over three locations:
  1. S(t) run-time  (ewm-app::StateCache)  — working set, tree, turns, tip
  2. cache          (designed, Arrow)       — LUTs, TF, hllsetLUT, Gx versions
  3. persistent     (ewm-git::Repository)   — commit DAG, blobs, HEAD, hllsetLUT

Commands:
  ewm-sm-explore store <path> [--json]      project the persistent layer
  ewm-sm-explore snapshot <file> [--json]   project the S(t) run-time area
                                            (exported by ewm-app --snapshot)
  ewm-sm-explore help
"
    );
}
