//! `ewm-scene` CLI — JSONL frames in, HLLSet statistics out.
//!
//! ```text
//! ewm-scene ingest <frames.jsonl>
//! ewm-scene bss <frames.jsonl>
//! ewm-scene ma <frames.jsonl> --short 1 --long 5
//! ewm-scene noether <frames.jsonl>
//! ewm-scene materialize <frames.jsonl>
//! ewm-scene grid <frames.jsonl> [--beam N]
//! ewm-scene tensor <frames.jsonl> [--beam N]
//! ewm-scene sidecar <frames.jsonl> [--cap N]
//! ```
//!
//! Input: one JSON object per line, `{"id": 1, "tokens": ["tid12", ...]}`.
//! Output: a single JSON value on stdout.

use std::io::{BufRead, Write};

use ewm_boolring::InsertResult;
use ewm_scene::{
    grid_restore_with, project, pyramid, restore_with, subframes, tensor_restore_with, Dimension,
    Frame, FrameSet, GridFrame, PerceptronTokens, PyramidFrame, TensorFrame,
};

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if let Err(e) = run(&args) {
        eprintln!("ewm-scene: {e}");
        eprintln!("usage: ewm-scene <ingest|bss|ma|noether|materialize|grid|tensor|sidecar|pyramid|project> <frames.jsonl> [options]");
        std::process::exit(2);
    }
}

fn run(args: &[String]) -> Result<(), String> {
    if args.is_empty() || args[0] == "help" || args[0] == "--help" {
        println!(
            "ewm-scene — direct morphisms path for the LLM <-> HLLSet side-car\n\
             \n\
             Commands:\n\
             \x20 ingest <file>                  frame keys + popcounts\n\
             \x20 bss <file>                     consecutive BSSτ / Jaccard\n\
             \x20 ma <file> --short N --long N   HLLSet moving averages\n\
             \x20 noether <file>                 D/R/N + three indicators\n\
             \x20 materialize <file>             ordered / set / beam-2 restoration
             \x20 grid <file> [--beam N]          2D morphisms (conv dim=2) restoration
             \x20 tensor <file> [--beam N]        N-d morphisms (conv dim=N) restoration
             \x20 subframes <file> --i N --j N    D/R/N subframes of a transition
             \x20 boolring <file>                 GF(2) span novelty/dimension series
             \x20 sidecar <file> [--cap N] [--freeze N]  Phase-1 trajectory: soft/hard keys, steps, jumps
             \x20 pyramid <file> [--cap N] [--freeze N]  Phase-2: union top perceptron + D/R/N, joined components, u-ring
             \x20 project <file> --frame <frame.json>   BSS trajectory of the stream over a named frame of dimensions"
        );
        return Ok(());
    }

    let cmd = args[0].as_str();
    let path = args.get(1).ok_or("missing <frames.jsonl>")?;
    // Only the flat-frame commands share the default reader; grid / tensor /
    // pyramid / subframes read their own format.
    let flat = matches!(
        cmd,
        "ingest" | "bss" | "ma" | "noether" | "materialize" | "sidecar" | "boolring" | "project"
    );
    let frames = if flat { read_frames(path)? } else { Vec::new() };
    let fs = FrameSet::from_frames(frames);

    let out = match cmd {
        "ingest" => {
            let frames: Vec<serde_json::Value> = fs
                .frames
                .iter()
                .enumerate()
                .map(|(i, f)| {
                    serde_json::json!({
                        "id": f.id,
                        "key": fs.keys[i],
                        "pop": fs.pops[i],
                    })
                })
                .collect();
            serde_json::json!({ "frames": frames })
        }
        "bss" => {
            let (tau, jac) = fs.bss_series();
            serde_json::json!({ "tau": tau, "jaccard": jac, "pop": fs.pops, "keys": fs.keys })
        }
        "ma" => {
            let short = arg_usize(args, "--short")?.unwrap_or(1);
            let long = arg_usize(args, "--long")?.unwrap_or(5);
            let (t0, fast, slow) = fs.moving_averages(short, long);
            serde_json::json!({ "t0": t0, "fast": fast, "slow": slow })
        }
        "noether" => {
            let n = fs.noether();
            serde_json::json!({
                "dp": n.dp, "rp": n.rp, "np": n.np,
                "ind1": n.ind1, "ind2": n.ind2, "ind3": n.ind3,
            })
        }
        "boolring" => {
            let fs = FrameSet::from_frames(read_frames(path)?);
            let mut basis = ewm_boolring::BoolBasis::new();
            let mut novelty = Vec::new();
            let mut dim = Vec::new();
            let mut coords_len = Vec::new();
            for hll in &fs.hllsets {
                match basis.insert(hll) {
                    InsertResult::InSpan { coords } => {
                        novelty.push(0u64);
                        dim.push(basis.dimension());
                        coords_len.push(coords.len());
                    }
                    InsertResult::Added { residual, .. } => {
                        novelty.push(residual.popcount());
                        dim.push(basis.dimension());
                        coords_len.push(0);
                    }
                }
            }
            serde_json::json!({
                "novelty": novelty,
                "dim": dim,
                "coords_len": coords_len,
            })
        }
        "subframes" => {
            let i = arg_usize(args, "--i")?.ok_or("subframes: missing --i")?;
            let j = arg_usize(args, "--j")?.ok_or("subframes: missing --j")?;
            let frames = read_grid_frames(path)?;
            let a = frames
                .get(i - 1)
                .ok_or_else(|| format!("--i {i} out of range ({} frames)", frames.len()))?;
            let b = frames
                .get(j - 1)
                .ok_or_else(|| format!("--j {j} out of range ({} frames)", frames.len()))?;
            let sf = subframes(a, b);
            serde_json::json!({
                "pair": sf.pair,
                "departed": sf.departed,
                "retained": sf.retained,
                "new": sf.new,
                "dp": sf.dp,
                "rp": sf.rp,
                "np": sf.np,
            })
        }
        "grid" => {
            let beam = arg_usize(args, "--beam")?.unwrap_or(2);
            let frames: Vec<serde_json::Value> = read_grid_frames(path)?
                .iter()
                .map(|f| {
                    let r = grid_restore_with(f, beam);
                    serde_json::json!({
                        "id": r.id,
                        "width": r.width,
                        "height": r.height,
                        "ordered": r.ordered,
                        "set": r.set,
                        "beam2": r.beam2,
                    })
                })
                .collect();
            serde_json::json!({ "frames": frames })
        }
        "tensor" => {
            let beam = arg_usize(args, "--beam")?.unwrap_or(2);
            let frames: Vec<serde_json::Value> = read_tensor_frames(path)?
                .iter()
                .map(|f| {
                    let r = tensor_restore_with(f, beam);
                    serde_json::json!({
                        "id": r.id,
                        "shape": r.shape,
                        "ordered": r.ordered,
                        "set": r.set,
                        "beam2": r.beam2,
                    })
                })
                .collect();
            serde_json::json!({ "frames": frames })
        }
        "materialize" => {
            let beam = arg_usize(args, "--beam")?.unwrap_or(2);
            let frames: Vec<serde_json::Value> = fs
                .frames
                .iter()
                .map(|f| {
                    let r = restore_with(f, beam);
                    serde_json::json!({
                        "id": r.id,
                        "ordered": r.ordered,
                        "set": r.set,
                        "beam2": r.beam2,
                    })
                })
                .collect();
            serde_json::json!({ "frames": frames })
        }
        "sidecar" => {
            let cap = arg_usize(args, "--cap")?.unwrap_or(ewm_app::RING_CAPACITY);
            let freeze = arg_usize(args, "--freeze")?;
            let out = fs.sidecar_with_cap_freeze(cap, freeze);
            let frames: Vec<serde_json::Value> = out
                .frames
                .iter()
                .enumerate()
                .map(|(i, f)| {
                    serde_json::json!({
                        "id": fs.frames[i].id,
                        "soft": f.soft,
                        "hard": f.hard,
                        "basis_pop": f.basis_pop,
                        "residual": f.residual,
                        "in_span": f.in_span,
                        "dim": f.dim,
                        "rotation_count": f.rotation_count,
                        "rotation_mass": f.rotation_mass,
                        "step": f.step,
                        "soft_first": f.soft_first,
                        "spill_first": f.spill_first,
                    })
                })
                .collect();
            serde_json::json!({
                "frames": frames,
                "jumps": out.jumps,
                "threshold": out.threshold,
                "freeze": freeze,
                "basis_history": out.basis_history,
            })
        }
        "pyramid" => {
            let cap = arg_usize(args, "--cap")?.unwrap_or(ewm_app::RING_CAPACITY);
            let freeze = arg_usize(args, "--freeze")?;
            let pf = read_pyramid_frames(path)?;
            let out = pyramid(&pf, cap, freeze);
            let frames: Vec<serde_json::Value> = out
                .frames
                .iter()
                .map(|f| {
                    let perceptrons: Vec<serde_json::Value> = f
                        .names
                        .iter()
                        .zip(&f.pops)
                        .zip(&f.keys)
                        .map(|((name, pop), key)| {
                            serde_json::json!({
                                "name": name,
                                "pop": pop,
                                "key": key,
                            })
                        })
                        .collect();
                    serde_json::json!({
                        "id": f.id,
                        "perceptrons": perceptrons,
                        "union_pop": f.union_pop,
                        "union_key": f.union_key,
                    })
                })
                .collect();
            let ring_frames: Vec<serde_json::Value> = out
                .ring
                .frames
                .iter()
                .map(|f| {
                    serde_json::json!({
                        "soft": f.soft,
                        "hard": f.hard,
                        "basis_pop": f.basis_pop,
                        "residual": f.residual,
                        "in_span": f.in_span,
                        "dim": f.dim,
                        "rotation_count": f.rotation_count,
                        "rotation_mass": f.rotation_mass,
                        "step": f.step,
                        "soft_first": f.soft_first,
                        "spill_first": f.spill_first,
                    })
                })
                .collect();
            serde_json::json!({
                "names": out.names,
                "frames": frames,
                "drn": {
                    "dp": out.drn.dp,
                    "rp": out.drn.rp,
                    "np": out.drn.np,
                    "ind1": out.drn.ind1,
                    "ind2": out.drn.ind2,
                    "ind3": out.drn.ind3,
                },
                "ring": {
                    "frames": ring_frames,
                    "jumps": out.ring.jumps,
                    "threshold": out.ring.threshold,
                    "basis_history": out.ring.basis_history,
                },
                "freeze": freeze,
            })
        }
        "project" => {
            let frame_path = arg_str(args, "--frame")?.ok_or("project: missing --frame <frame.json>")?;
            let dims = read_dimensions(frame_path)?;
            let ids: Vec<u64> = fs.frames.iter().map(|f| f.id).collect();
            let out = project(&fs.hllsets, &ids, &dims);
            let frames: Vec<serde_json::Value> = out
                .frames
                .iter()
                .map(|f| {
                    serde_json::json!({
                        "id": f.id,
                        "intersections": f.intersections,
                        "bss": f.bss,
                    })
                })
                .collect();
            serde_json::json!({
                "names": out.names,
                "pops": out.pops,
                "frames": frames,
            })
        }
        other => return Err(format!("unknown command: {other}")),
    };

    let mut stdout = std::io::stdout().lock();
    serde_json::to_writer_pretty(&mut stdout, &out).map_err(|e| e.to_string())?;
    stdout.write_all(b"\n").map_err(|e| e.to_string())?;
    Ok(())
}

fn arg_usize(args: &[String], name: &str) -> Result<Option<usize>, String> {
    for (i, a) in args.iter().enumerate() {
        if a == name {
            let v = args
                .get(i + 1)
                .ok_or_else(|| format!("{name}: missing value"))?
                .parse()
                .map_err(|_| format!("{name}: not an integer"))?;
            return Ok(Some(v));
        }
    }
    Ok(None)
}

fn arg_str<'a>(args: &'a [String], name: &str) -> Result<Option<&'a str>, String> {
    for (i, a) in args.iter().enumerate() {
        if a == name {
            let v = args
                .get(i + 1)
                .ok_or_else(|| format!("{name}: missing value"))?;
            return Ok(Some(v.as_str()));
        }
    }
    Ok(None)
}

/// Read a projection frame: a JSON object
/// `{"dimensions": [{"name": "D", "tokens": [...]}, ...]}`.
fn read_dimensions(path: &str) -> Result<Vec<Dimension>, String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("{path}: {e}"))?;
    let v: serde_json::Value =
        serde_json::from_str(&text).map_err(|e| format!("{path}: {e}"))?;
    let dims = v["dimensions"]
        .as_array()
        .ok_or_else(|| format!("{path}: missing dimensions"))?;
    let mut out = Vec::with_capacity(dims.len());
    for (i, d) in dims.iter().enumerate() {
        let name = d["name"]
            .as_str()
            .ok_or_else(|| format!("{path}: dimension {i} missing name"))?
            .to_string();
        let tokens: Vec<String> = d["tokens"]
            .as_array()
            .ok_or_else(|| format!("{path}: dimension {name} missing tokens"))?
            .iter()
            .map(|t| {
                t.as_str()
                    .map(|s| s.to_string())
                    .ok_or_else(|| format!("{path}: dimension {name} token is not a string"))
            })
            .collect::<Result<Vec<_>, _>>()?;
        out.push(Dimension::from_tokens(name, &tokens));
    }
    Ok(out)
}

fn read_grid_frames(path: &str) -> Result<Vec<GridFrame>, String> {
    let file = std::fs::File::open(path).map_err(|e| format!("{path}: {e}"))?;
    let mut frames = Vec::new();
    for (lineno, line) in std::io::BufReader::new(file).lines().enumerate() {
        let line = line.map_err(|e| format!("{path}:{lineno}: {e}"))?;
        if line.trim().is_empty() {
            continue;
        }
        let v: serde_json::Value =
            serde_json::from_str(&line).map_err(|e| format!("{path}:{lineno}: {e}"))?;
        let id = v["id"].as_u64().ok_or_else(|| format!("{path}:{lineno}: missing id"))?;
        let width = v["width"]
            .as_u64()
            .ok_or_else(|| format!("{path}:{lineno}: missing width"))?
            as usize;
        let height = v["height"]
            .as_u64()
            .ok_or_else(|| format!("{path}:{lineno}: missing height"))?
            as usize;
        let tokens = v["tokens"]
            .as_array()
            .ok_or_else(|| format!("{path}:{lineno}: missing tokens"))?
            .iter()
            .map(|t| {
                t.as_str()
                    .map(|s| s.to_string())
                    .ok_or_else(|| format!("{path}:{lineno}: token is not a string"))
            })
            .collect::<Result<Vec<_>, _>>()?;
        frames.push(GridFrame {
            id,
            width,
            height,
            tokens,
        });
    }
    Ok(frames)
}

fn read_tensor_frames(path: &str) -> Result<Vec<TensorFrame>, String> {
    let file = std::fs::File::open(path).map_err(|e| format!("{path}: {e}"))?;
    let mut frames = Vec::new();
    for (lineno, line) in std::io::BufReader::new(file).lines().enumerate() {
        let line = line.map_err(|e| format!("{path}:{lineno}: {e}"))?;
        if line.trim().is_empty() {
            continue;
        }
        let v: serde_json::Value =
            serde_json::from_str(&line).map_err(|e| format!("{path}:{lineno}: {e}"))?;
        let id = v["id"].as_u64().ok_or_else(|| format!("{path}:{lineno}: missing id"))?;
        let shape = v["shape"]
            .as_array()
            .ok_or_else(|| format!("{path}:{lineno}: missing shape"))?
            .iter()
            .map(|d| {
                d.as_u64()
                    .map(|n| n as usize)
                    .ok_or_else(|| format!("{path}:{lineno}: shape is not a list of integers"))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let tokens = v["tokens"]
            .as_array()
            .ok_or_else(|| format!("{path}:{lineno}: missing tokens"))?
            .iter()
            .map(|t| {
                t.as_str()
                    .map(|s| s.to_string())
                    .ok_or_else(|| format!("{path}:{lineno}: token is not a string"))
            })
            .collect::<Result<Vec<_>, _>>()?;
        frames.push(TensorFrame { id, shape, tokens });
    }
    Ok(frames)
}

/// Read pyramid frames: `{"id": 1, "perceptrons": {"perception": [...], ...}}`
/// or `{"id": 1, "perceptrons": [[...], [...]]}` (auto-named p1..pm).
fn read_pyramid_frames(path: &str) -> Result<Vec<PyramidFrame>, String> {
    let file = std::fs::File::open(path).map_err(|e| format!("{path}: {e}"))?;
    let mut frames = Vec::new();
    for (lineno, line) in std::io::BufReader::new(file).lines().enumerate() {
        let line = line.map_err(|e| format!("{path}:{lineno}: {e}"))?;
        if line.trim().is_empty() {
            continue;
        }
        let v: serde_json::Value =
            serde_json::from_str(&line).map_err(|e| format!("{path}:{lineno}: {e}"))?;
        let id = v["id"].as_u64().ok_or_else(|| format!("{path}:{lineno}: missing id"))?;
        let ps = v["perceptrons"]
            .as_object()
            .ok_or_else(|| format!("{path}:{lineno}: perceptrons must be an object"))?;
        let mut perceptrons = Vec::with_capacity(ps.len());
        for (name, tokens) in ps {
            let tokens = tokens
                .as_array()
                .ok_or_else(|| format!("{path}:{lineno}: perceptron {name} is not a list"))?
                .iter()
                .map(|t| {
                    t.as_str()
                        .map(|s| s.to_string())
                        .ok_or_else(|| format!("{path}:{lineno}: token is not a string"))
                })
                .collect::<Result<Vec<_>, _>>()?;
            perceptrons.push(PerceptronTokens {
                name: name.clone(),
                tokens,
            });
        }
        frames.push(PyramidFrame { id, perceptrons });
    }
    Ok(frames)
}

fn read_frames(path: &str) -> Result<Vec<Frame>, String> {
    let file = std::fs::File::open(path).map_err(|e| format!("{path}: {e}"))?;
    let mut frames = Vec::new();
    for (lineno, line) in std::io::BufReader::new(file).lines().enumerate() {
        let line = line.map_err(|e| format!("{path}:{lineno}: {e}"))?;
        if line.trim().is_empty() {
            continue;
        }
        let v: serde_json::Value =
            serde_json::from_str(&line).map_err(|e| format!("{path}:{lineno}: {e}"))?;
        let id = v["id"].as_u64().ok_or_else(|| format!("{path}:{lineno}: missing id"))?;
        let tokens = v["tokens"]
            .as_array()
            .ok_or_else(|| format!("{path}:{lineno}: missing tokens"))?
            .iter()
            .map(|t| {
                t.as_str()
                    .map(|s| s.to_string())
                    .ok_or_else(|| format!("{path}:{lineno}: token is not a string"))
            })
            .collect::<Result<Vec<_>, _>>()?;
        frames.push(Frame { id, tokens });
    }
    Ok(frames)
}
