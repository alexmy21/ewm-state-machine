//! `ewm-sm-explore` CLI — the read-only explorer of the ewm-state-machine.
//!
//! ```text
//! ewm-sm-explore store <path> [--json]      persistent layer (ewm-git)
//! ewm-sm-explore snapshot <file> [--json]   S(t) run-time area (StateSnapshot)
//! ewm-sm-explore project-user --store <path> --user <id> --commit <hex|head>
//!                              --codebook <file.json>    per-user gate projection
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
            eprintln!("       ewm-sm-explore project-user --store <path> --user <id> --commit <hex|head> [--codebook <file.json>]");
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
        "project-user" => {
            let store_path = arg_val(args, "--store")?;
            let user = arg_val(args, "--user")?;
            let commit = arg_val(args, "--commit")?;
            let codebook_path = arg_opt(args, "--codebook")?;
            cmd_project_user(store_path, user, commit, codebook_path)
        }
        other => Err(format!("unknown command: {other}")),
    }
}

fn arg_val<'a>(args: &'a [String], name: &str) -> Result<&'a str, String> {
    for (i, a) in args.iter().enumerate() {
        if a == name {
            return args
                .get(i + 1)
                .map(|s| s.as_str())
                .ok_or_else(|| format!("{name}: missing value"));
        }
    }
    Err(format!("{name}: missing"))
}

fn arg_opt<'a>(args: &'a [String], name: &str) -> Result<Option<&'a str>, String> {
    Ok(arg_val(args, name).ok())
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

fn cmd_project_user(
    store_path: &str,
    user: &str,
    commit: &str,
    codebook_path: Option<&str>,
) -> Result<(), String> {
    let repo = ewm_git::Repository::open(ewm_git::LooseStore::new(store_path));
    let cid = if commit == "head" {
        repo.head()
            .cloned()
            .ok_or_else(|| format!("{store_path}: no HEAD"))?
    } else {
        ewm_git::ObjectId::validated(commit.to_string())
            .ok_or_else(|| "commit must be 40 hex chars or 'head'".to_string())?
    };

    // The codebook either comes from an explicit file or from the store's
    // gates section (the per-user catalog registry).
    let (tokens, gate_source) = match codebook_path {
        Some(path) => {
            let raw = std::fs::read_to_string(path).map_err(|e| format!("{path}: {e}"))?;
            let v: serde_json::Value =
                serde_json::from_str(&raw).map_err(|e| format!("{path}: {e}"))?;
            let tokens: Vec<String> = v["tokens"]
                .as_array()
                .ok_or_else(|| format!("{path}: missing tokens array"))?
                .iter()
                .map(|t| {
                    t.as_str()
                        .map(|s| s.to_string())
                        .ok_or_else(|| format!("{path}: token is not a string"))
                })
                .collect::<Result<Vec<_>, _>>()?;
            (tokens, "file".to_string())
        }
        None => {
            let (_key, tokens) = repo
                .gate(user)
                .map_err(|e| format!("gates section: {e}"))?;
            (tokens, "gates-section".to_string())
        }
    };

    let mut doc = ewm_sm_explore::project_user_json(&repo, user, &cid, &tokens)?;
    doc["gate_source"] = serde_json::Value::String(gate_source);
    println!("{}", serde_json::to_string_pretty(&doc).map_err(|e| e.to_string())?);
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
  ewm-sm-explore project-user --store <path> --user <id> --commit <hex|head>
                              [--codebook <file.json>]
                                            per-user codebook gate projection:
                                            c(t) n G_u + restored tokens
                                            (codebook from the file, or from
                                            the store's gates/ section when
                                            --codebook is omitted)
  ewm-sm-explore help
"
    );
}
