# Assignment — Qwen-Drive test bench for the ewm-sm side-car

**Status:** Phase 1 implemented and green — notebook `notebooks/03_qwendrive_sidecar.ipynb`
runs the loop on real Qwen-Drive-1.0-4B perception tokens (48 camera frames from the
bundled demo scenes), with the two-score evaluation and the soft-key trajectory.
The `ewm-scene` helper gained `sidecar` (soft/hard keys, step lengths, jump detector,
rotation component, `--cap`, `--freeze` stable-coordinate mode) and `tensor`
(conv dim=N restoration) commands; the notebook includes the frozen-basis DFT
decomposition and the tangible D/R/N warning hand-off. Phase 2 started: the
simple model is in — `ewm-scene pyramid` (union top perceptron + D/R/N, joined
components, u-ring) and notebook `notebooks/04_perceptron_pyramid.ipynb`.

## Goal

Validate the full side-car loop — **encodings → ewm-sm → encodings** — on a
real vision-language-action model, using **Qwen-Drive-1.0** perception
tokens as the host encodings, with a **Qwen3-family 4B** planner as the
host decoder side.

> Model note: Qwen-Drive-1.0 is a VLA stack — Qwen3-VL vision tower →
> **perception tokens** (continuous latent embeddings of the scene) →
> driving planner. We tap the perception-token interface: that is the
> "encodings" our side-car consumes. If the exact release is Qwen3.5-4B,
> use it; the interface is what matters, not the release number.

## The settled design (from this session)

Two measurement-level addresses over the Boolean-ring basis
`B = {B_1 … B_k}`:

```text
hard key  = GF(2) coordinates   A = B_1 Δ B_3 Δ B_7 ⇔ (1,0,1,0,…,1)
                                (span members only; exact structural match)

soft key  = BSS weights         w_i = |A ∩ B_i| / |B_i|
                                (any HLLSet; similarity over the basis)
```

Both are semantics-free — the HLLSet is the measurement (IICA). Semantics
live only in the host encoder/decoder.

## Phase 1 — single perceptron

```text
perception tokens (f32 vectors, per time step t)
   │  CodebookEncoder (shared codebook)
   ▼
tid ids → ingest → S(t), H(t-1), D/R/N, ring (BoolWindow)
   │
   ▼
ordered materialize → restored_ids → host planner/decoder → actions
```

Deliverables:

- a notebook app (`notebooks/03_qwendrive_sidecar.ipynb`) that loads
  Qwen-Drive, extracts perception tokens from driving video, and runs the
  loop per frame/turn;
- the two-score evaluation: **loop accuracy** (hash-level restoration,
  expected 1.0) and **decode quality** (driving-relevant metric, separate);
- the trajectory: per-frame BSS weight vectors against the ring basis
  (soft key), step lengths, jump detector; cross-check with the
  Boolean-ring residual (linear novelty) and D/R/N.

## Phase 2 — pyramid of perceptrons

```text
S(t)_i  = state of perceptron i at time t = i-HLLSet
          (perception, planner, ego-state, navigation, action …)

perceptron-level Ring over {S(t)_1 … S(t)_n}
   │  hard/soft keys over the perceptron basis
   ▼
system trajectory in k-dim — the whole-system state path
```

- the **perceptron** is a processing unit that owns one HLLSet state per
  time step; the pyramid is a hierarchy of such units (levels can nest);
- the system Ring is exactly the Boolean-ring window over the perceptron
  states — same algebra, one level up;
- start with one perceptron (the perception-token stream), then add the
  planner/action perceptrons, then nest.

## Constraints

- ewm-state-machine remains the main line; no new top-level workspace.
- Keep the morphisms vocabulary-agnostic: Qwen-Drive awareness lives only
  in the notebook + the encoder adapter.
- GPU work runs in a dedicated conda env (the `deepseek-ocr` pattern);
  kernel restarts free GPU memory when a run OOMs.

## Entry points for the new session

- this file (the assignment);
- `docs/SEPARATION.md` (the bit is the fiber; restoration contract);
- `docs/BOOLRING.md` (the windowed ring, hard/soft keys);
- `notebooks/02_scene_sidecar.ipynb` (the same loop on the scene bench);
- `crates/ewm-app/src/encoder.rs` (the encoder adapter);
- `crates/ewm-boolring/src/lib.rs` (BoolWindow, coordinates, residual).
