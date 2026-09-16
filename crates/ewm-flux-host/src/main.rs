//! `ewm-flux-host` CLI — run the Flux side-car loop over a synthetic
//! random-weight MMDiT trajectory and print the JSON report.
//!
//! ```bash
//! cargo run -p ewm-flux-host
//! cargo run -p ewm-flux-host -- --seq-len 256 --dim 64 --steps 28 --codebook 4096 --seed 42
//! cargo run -p ewm-flux-host -- --disturb-step 16 --disturb-scale 8 --disturb-patch 32
//! cargo run -p ewm-flux-host -- --no-disturb
//! ```
//!
//! Output: a single JSON value on stdout (the `ewm-scene` convention).

use ewm_app::CodebookEncoder;
use ewm_flux_host::{run, CodebookProbe, Disturbance, SyntheticFluxHost};

#[derive(Clone, Copy, Debug)]
struct Cli {
    seq_len: usize,
    dim: usize,
    steps: usize,
    codebook: usize,
    seed: u64,
    amp: f32,
    disturbance: Option<Disturbance>,
}

impl Default for Cli {
    fn default() -> Self {
        Self {
            seq_len: 256,
            dim: 64,
            steps: 28,
            codebook: 4096,
            seed: 42,
            amp: 2.0,
            // A known jump at step 16 so the default run demonstrates the
            // side-car warnings on a synthetic guidance change.
            disturbance: Some(Disturbance {
                step: 16,
                scale: 12.0,
                patch: 32,
            }),
        }
    }
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.iter().any(|a| a == "--help" || a == "-h") {
        print_help();
        return;
    }
    let cli = match parse_args(&args) {
        Ok(cli) => cli,
        Err(e) => {
            eprintln!("ewm-flux-host: {e}");
            print_help();
            std::process::exit(2);
        }
    };

    let host = SyntheticFluxHost::with_opts(
        cli.seq_len,
        cli.dim,
        cli.steps,
        cli.seed,
        cli.amp,
        cli.disturbance,
    );
    // A separate codebook seed so the codebook and the trajectory are
    // independent draws of the same deterministic stream family.
    let codebook = CodebookEncoder::new(cli.dim, cli.codebook, cli.seed ^ 0x1111_2222_3333_4444);
    let mut probe = CodebookProbe::new(codebook);

    let report = run(&host, &mut probe);
    let mut stdout = std::io::stdout().lock();
    serde_json::to_writer_pretty(&mut stdout, &report).expect("write JSON report");
    use std::io::Write;
    stdout.write_all(b"\n").expect("write newline");
}

fn parse_args(args: &[String]) -> Result<Cli, String> {
    let mut cli = Cli::default();
    let mut disturb_step = cli.disturbance.map(|d| d.step).unwrap_or(0);
    let mut disturb_scale = cli.disturbance.map(|d| d.scale).unwrap_or(0.0);
    let mut disturb_patch = cli.disturbance.map(|d| d.patch).unwrap_or(0);
    let mut no_disturb = false;

    let mut i = 0;
    while i < args.len() {
        let arg = args[i].as_str();
        let mut value = |name: &str| -> Result<String, String> {
            i += 1;
            args.get(i)
                .cloned()
                .ok_or_else(|| format!("{name}: missing value"))
        };
        match arg {
            "--seq-len" => cli.seq_len = value("--seq-len")?.parse().map_err(|_| "--seq-len: not an integer".to_string())?,
            "--dim" => cli.dim = value("--dim")?.parse().map_err(|_| "--dim: not an integer".to_string())?,
            "--steps" => cli.steps = value("--steps")?.parse().map_err(|_| "--steps: not an integer".to_string())?,
            "--codebook" => cli.codebook = value("--codebook")?.parse().map_err(|_| "--codebook: not an integer".to_string())?,
            "--seed" => cli.seed = value("--seed")?.parse().map_err(|_| "--seed: not an integer".to_string())?,
            "--amp" => cli.amp = value("--amp")?.parse().map_err(|_| "--amp: not a float".to_string())?,
            "--disturb-step" => disturb_step = value("--disturb-step")?.parse().map_err(|_| "--disturb-step: not an integer".to_string())?,
            "--disturb-scale" => disturb_scale = value("--disturb-scale")?.parse().map_err(|_| "--disturb-scale: not a float".to_string())?,
            "--disturb-patch" => disturb_patch = value("--disturb-patch")?.parse().map_err(|_| "--disturb-patch: not an integer".to_string())?,
            "--no-disturb" => no_disturb = true,
            other => return Err(format!("unknown argument: {other}")),
        }
        i += 1;
    }

    if cli.seq_len == 0 || cli.dim == 0 || cli.steps == 0 || cli.codebook == 0 {
        return Err("--seq-len/--dim/--steps/--codebook must be positive".into());
    }
    if no_disturb {
        cli.disturbance = None;
    } else {
        cli.disturbance = Some(Disturbance {
            step: disturb_step,
            scale: disturb_scale,
            patch: disturb_patch,
        });
    }
    Ok(cli)
}

fn print_help() {
    println!(
        "ewm-flux-host — Flux/MMDiT side-car over a synthetic latent trajectory\n\
         \n\
         Usage:\n\
         \x20 ewm-flux-host [options]\n\
         \n\
         Options:\n\
         \x20 --seq-len N       latent tokens per step (default 256)\n\
         \x20 --dim N           latent channels per token (default 64)\n\
         \x20 --steps N         denoising steps (default 28)\n\
         \x20 --codebook N      shared codebook anchors (default 4096)\n\
         \x20 --seed N          run seed (default 42)\n\
         \x20 --amp F           random-weight model amplitude (default 2.0)\n\
         \x20 --disturb-step N  inject a jump at output state N (default 16)\n\
         \x20 --disturb-scale F jump noise scale (default 12.0)\n\
         \x20 --disturb-patch N leading tokens the jump touches (default 32)\n\
         \x20 --no-disturb      run the clean trajectory (no injected jump)\n\
         \n\
         Output: a single JSON report on stdout."
    );
}
