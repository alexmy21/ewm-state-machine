# ewm-state-machine

The **state-centric** home of the fractal_manifold_gen2 collection. The
accent here is not HLLSet — it is the **state of the system, S(t)**.

## Lineage — this is the main line

```text
hllset-next-v2            foundation (core / contracts / lut / morphisms)
ewm-cortex-fpga-v2        FPGA PoC + DSL proof  ── superseded for app work;
                          parked until tangible FPGA development results
hllset-fpga-simulator-v2  FPGA simulator reference (kept)
ewm-state-machine         ◄── MAIN LINE — the balanced separation:
                          HLLSet state plane first, operations around it
```

The FPGA line proved its two points — an FPGA can serve an EWM system, and
ewm applications can be developed in a high-level DSL. This workspace
inherits the conclusions, not the bridge: the **HLLSet structure (the state
plane) is prior; the operational part is organized around the states.**

## The fundamental equation

```text
H(t) = ( S(t), H(t-1), D, R, N )

  S(t)    = current emerging state — work in progress, the tip of the stack
  H(t-1)  = previous committed state — the cache (required to commit)
  N(t)    = S(t) \ H(t-1)   new information
  R(t)    = S(t) ∩ H(t-1)   retained (the R-link)
  D(t)    = H(t-1) \ S(t)   departed information
```

**Everything in this equation is an HLLSet** — H(t), S(t), H(t-1), D, R, N,
and every set operation on them are bit sets. The state machine's algebra
runs entirely at the bit level. **Tokens are restored only by
materialization** (LUT-first, keeping every candidate — probabilistic
restoration, no TF filtering): ingest lifts tokens
into bits, materialize lowers bits back into tokens.

## The map — four ideas, not a forest

The whole model hangs on four ideas, in order of depth. When lost, come back
here:

```text
1. One universe.        Every HLLSet is a node of the same Boolean lattice
                        over the bit plane (⊆ order, ∪ join, ∩ meet, Δ as
                        the ring addition). Nothing new is created — a
                        computation names a lattice expression and
                        discovers the node.                  → docs/BOOLRING.md

2. The morphism is the  The token→bit connection is the chosen morphism
   contract (IICA).     μ: token collections → bit vectors. Only IICA
                        (Idempotence, Immutability, Content Addressability)
                        is required; the bit vector forgets provenance by
                        design, and any IICA algorithm is interchangeable.
                                                           → docs/BOOLRING.md

3. The fiber is global. The same bit in any HLLSet points to the same LUT
                        fiber — the fiber is a property of the bit address,
                        never of the HLLSet. Materialization is exact on
                        the builders, probabilistic on the extras.
                                                         → docs/SEPARATION.md

4. Decompositions are   D/R/N, joined perceptrons, ring basis — each is a
   frames (readings).   named set of dimensions; φ_D(X)=(|X∩D_i|/|D_i|) is
                        the coordinate map, and φ_D(X(t)) over t is the
                        HLLSet trajectory. Each frame's scope is set by how
                        it is built.         → docs/ASSIGNMENT_QWENDRIVE.md
```

`ewm-scene` makes all four ideas executable: `ingest`/`materialize` (2, 3),
`noether`/`subframes` (D/R/N frame), `pyramid` (joined perceptrons),
`sidecar` (ring frame), `project` (any named frame).

S(t) is the **state in a stateless system**. IICA — Idempotence,
Immutability, Content Addressability — removes the contradiction: S(t) is an
immutable, content-addressed value, so it is safe to share. The [UM]
(processing unit) that works on S(t) is stateless and disposable; it reads
the tip, processes incoming tokens, and proposes the next state.

S(t) and H(t-1) live **outside the [UM]**, in a shareable cache
(`ewm-app::StateCache`), backed by the persistent store (`ewm-git`):

```text
## The side-car loop

The [UM] consumes host encodings through an **encoder** — the only
vocabulary-aware step (`ewm-app::CodebookEncoder`, a shared codebook
quantizer). Everything below it is vocabulary-agnostic:

```text
host encodings (f32 vectors)
        │  CodebookEncoder::encode   (codebook quantization)
        ▼
tid{n} ids → ingest → HLLSets (S(t), H(t-1), D/R/N, ring)
        │
        ▼
ordered materialize → restored_ids → back to the host
```

```text
persistent store (ewm-git)     the committed stack; tip = head
        ▲  │
   load │  │ commit
        │  ▼
shared cache (StateCache)      S(t) working set + H(t-1) materialized
        ▲  │
   read │  │ propose
        │  ▼
     [UM] (ewm-app)            stateless, disposable — owns only the store handle
```

Recovery is a **pop, not a rebuild**: drop the broken [UM] and its cache,
restore the cache from the tip, and continue with a fresh [UM]. Uncommitted
work is reprocessed — idempotent by IICA.

## Vocabulary

| Term | Meaning |
| ---- | ------- |
| S(t) | current emerging state (work in progress, shared) |
| H(t-1) | previous committed state (the cache, required to commit) |
| cache | the shareable in-memory layer holding S(t) and H(t-1), outside the [UM] |
| D / R / N | departed / retained / new — derived at commit |
| tip | the committed head of the state stack |
| commit | the atomic advance of the tip, recording H(t) |
| [UM] | the stateless, disposable processing unit |
| IICA | Idempotence, Immutability, Content Addressability |
| conv(n, dim) | the channel model: n-grams are conv(n, dim=1); grids conv(n, dim=2); tensors conv(n, dim=N) — shared 1×1 G1, seed(n,dim) = (dim−1)·3 + (n−1) |

Terminology reference:
`../hllset-next-v2/_DOCS/dev/STANDARD.md`.

## Layout

```text
ewm-state-machine/
├── Cargo.toml                 # workspace manifest
├── docs/
│   ├── ARROW_CACHE.md         # design note: Arrow as the cache-layer substrate
│   ├── EXPLORER.md            # the read-only explorer contract
│   ├── SEPARATION.md          # separation of concerns — the bit is the fiber
│   ├── BOOLRING.md            # GF(2) Boolean-ring context index (spike, positive)
│   ├── BASIS_FRAMES.md        # basis frames: HLLSet interpretation, time travel, commit conditions
│   ├── TRAINER.md             # trainer layer: separation map, protocol interfaces, DSL roadmap
│   ├── PODMAN.md              # podman packaging + notebook/container setup
│   └── ASSIGNMENT_QWENDRIVE.md # next assignment — Qwen-Drive test bench
├── trainer/                   # the trainer/controller reference implementation
│   ├── protocol.py            # probe config / trajectory / predictor+selector shapes
│   ├── predictors.py          # persistence / linear / dft-periodic / ridge portfolio
│   ├── selector.py            # EWMA surprise + epsilon-greedy meta-controller
│   ├── adapters.py            # probe adapters (SyntheticAdapter, LlmAdapter)
│   ├── ewm.py                 # ewm-scene client (the only Rust-facing module)
│   ├── loop.py                # open loop + closed loop (memory + curiosity)
│   └── smoke_test.py          # full-pipeline self-check (no torch needed)
├── captures/                  # capture-first front-end scripts (OCR / VLA / JEPA)
└── crates/
    ├── hllset-contracts/      # soldered invariants (leaf)
    ├── hllset-cid/            # embedded SHA-1 CIDs
    ├── hllset-core/           # HLLSet bit-plane — the state representation
    ├── hllset-lut/            # the LUT lattice (token interpretation)
    ├── hllset-morphisms/      # ingest/materialize + default api (SHA1 + hllsetLUT)
    ├── hllset-storage/        # memory + sled content-addressed storage
    ├── context-tree/          # Merkle tree over the working set (S(t) presentation)
    ├── ewm-git/               # the state stack: commit DAG, tip, recovery
    ├── ewm-app/               # the [UM] harness: stateless driver + StateCache
    │                          # (S(t) and H(t-1) live in the cache, not the [UM])
    ├── ewm-sm-explore/        # read-only explorer of the three-layer structure
    ├── ewm-scene/             # LLM <-> HLLSet side-car helper (direct morphisms)
    ├── ewm-flux-host/         # Flux/MMDiT side-car host adapter (synthetic MMDiT
    │                          # shim, SidecarProbe, in-process loop, two-score eval)
    ├── ewm-ops/               # the operational graph: content-addressed values +
    │                          # programs + stack dispatcher (value/program lattice)
    └── ewm-cache/             # Arrow-backed extended cache (docs/ARROW_CACHE.md):
                               # IPC spill/restore of the derived cache objects
```

`ewm-cache` is the Arrow cache layer (designed in
[`docs/ARROW_CACHE.md`](docs/ARROW_CACHE.md)): HLLSets stay Roaring-native;
Arrow IPC files spill/restore the derived cache objects behind a validated
`MANIFEST.arrow`, and the cache is content-addressed as `c:<sha1>`.

## Explorer

The state machine is one data structure over three locations — S(t)
run-time, cache, persistent layer. `ewm-sm-explore` projects them read-only:

```bash
# Persistent layer (commit DAG, per-commit Gx keys, D/R/N, hllsetLUT, tops)
cargo run -p ewm-sm-explore -- store /tmp/ewm-sm-demo [--json]

# S(t) run-time area (exported by the app after the loop)
cargo run -p ewm-app -- --stub "1,2,3;2,3,4" --repo /tmp/ewm-sm-demo \
    --snapshot /tmp/ewm-sm-demo/snapshot.json
cargo run -p ewm-sm-explore -- snapshot /tmp/ewm-sm-demo/snapshot.json
```

See [`docs/EXPLORER.md`](docs/EXPLORER.md).

## Flux side-car

`ewm-flux-host` is the side-car host adapter for aarambh-vision-studio's
rectified-flow MMDiT sampler (developed here first against a synthetic
random-weight shim; port the adapter crate later — see
`docs/notes/flux-notes.md`). One call closes the whole loop on a synthetic
latent trajectory:

```bash
cargo run -p ewm-flux-host -- --seq-len 256 --dim 64 --steps 28 --codebook 4096
cargo run -p ewm-flux-host -- --no-disturb        # clean trajectory
cargo test -p ewm-flux-host
```

The JSON report carries, per denoising step: `S(t)` popcount + content key,
D/R/N, the Boolean-ring record (soft/hard key, residual, rotation), the
warning flags, and the two-score evaluation — loop accuracy (ordered + set
latent token restoration, expected 1.0) and decode quality (latent
reconstruction cosine/MSE). Because the ring basis changes as the run grows,
the report also carries the **lazy back-propagated** view: `soft_final` /
`step_final` re-project every step onto the final basis (each cached vector
is stamped with the basis generation; only the entries queried are
recomputed — never the whole history per basis change).

## Operational graph

`ewm-ops` standardizes the state-machine representation: a **content-addressed
operational graph** whose two sides are the value lattice (`h:<sha1>` HLLSets)
and the program lattice (`p:<sha1>` expressions — any DSL expression is a
[UM], persistence optional), tied by directed edges and traversed by a
stack-pop dispatcher (fan-out by reference, deterministic fire sequence,
feedback cycles under a fire budget). The boot file compiles into the graph
plus a **content-addressed vocabulary** (`v:<sha1>`), and the CLI boots it
like an OS:

```bash
cargo run -p ewm-ops -- --store /tmp/ewm-ops-demo --boot path/to/boot.ops \
    --fires 12 --repo /tmp/ewm-ops-demo/repo
cargo run -p ewm-ops -- --store /tmp/ewm-ops-demo   # picks up state on top of stack
cargo run -p ewm-ops log  --store /tmp/ewm-ops-demo # append-only boot log
cargo run -p ewm-ops list --store /tmp/ewm-ops-demo # distinct boots + latest flag
cargo run -p ewm-ops prev --store /tmp/ewm-ops-demo # roll latest back one boot
cargo test -p ewm-ops
```

## Podman

The workspace ships a multi-stage `Containerfile` (see
[`docs/PODMAN.md`](docs/PODMAN.md)) that builds all six binaries and packs
them into a small non-root Debian image:

```bash
podman build -t ewm-state-machine .
podman run --rm ewm-state-machine ewm-ops --help
podman run --rm -v /tmp/ewm-podman:/var/lib/ewm ewm-state-machine \
    ewm-ops --store /var/lib/ewm --boot /var/lib/ewm/boot.ops
```

## Notebooks

The notebook is the application: each code cell is a step, and the [UM] runs
the cells.

| # | Notebook | Description |
| - | -------- | ----------- |
| 01 | `ingest_materialize_um` | morphisms step by step — ingest (SHA1 + three pointers + hllsetLUT), materialize (ordered / `no_order` / beam), then the [UM] loop and recovery |
| 02 | `scene_sidecar` | the LLM ↔ HLLSet side-car application rebuilt on ewm-state-machine: vLLM host line + `ewm-scene` direct-morphism side-car (BSSτ, D/R/N, exact roundtrip on the conv(n, dim=2) grid path, restore-from-HLLSet pixel demo) |
| 03 | `qwendrive_sidecar` | Phase 1 Qwen-Drive test bench: real Qwen-Drive-1.0-4B perception tokens → shared codebook → the side-car loop per camera frame (S(t), H(t-1), D/R/N, ring) → ordered materialize → two-score evaluation (loop accuracy, decode quality) + the soft-key trajectory with jump detector, residual and D/R/N cross-check |
| 04 | `perceptron_pyramid` | Phase 2 simple model: m perceptrons per frame, the union top perceptron `u-HLLSet(t)`, and its three decompositions — D/R/N of the union stream, the joined per-perceptron components, and the u-ring basis decomposition |
| 05 | `flux_sidecar` | Flux Phase-1 test bench on a synthetic random-weight MMDiT shim: `ewm-flux-host` runs the in-process loop per denoising step (S(t), D/R/N, ring, warnings), the two-score evaluation (loop accuracy = latent token restoration 1.0; decode quality = latent reconstruction cosine/MSE), and the trajectory plots with the injected-jump warning |
| 06 | `ewm_ops_boot` | the operational graph (`ewm-ops`) boots like an OS: content-addressed boot file → compile → pick up the persisted state from the top of the stack → run the stack-pop dispatcher (fan-out by reference, deterministic fire sequence, feedback loop under a fire budget) → commit the fire log into `ewm-git` |
| 07 | `jepa_ewm_state_machine_cooperation` | the six `ewm-jepa` demos consolidated into one cooperation story: V-JEPA encoder/predictor ⇄ the Rust lattice — pipeline/IICA, three-LUT unification, recursive IICA chain, [UM]-Net agent fan-out, holographic D/R/N, and the grounding proof (fidelity + one-sided gate) |
| 08 | `multi_llm_sidecar` | the multi-LLM side-car environment: three synthetic LLM probes feed `ewm-scene pyramid`, the union state `S(t) = ∪ S_i(t)`, the m-dim BSS system trajectory over the per-LLM ring, the pattern matrix `(LLM, Ring)` → `(LLM, Ring, DRN)` → `(Ring, Ring, DRN)` (ring states as virtual LLMs), and the first trajectory training targets (next BSS vector, next commit signal) |
| 09 | `multi_llm_sidecar_real` | the same side-car with three **real small LLMs** (DeepSeek-R1-Distill-Qwen-1.5B, Qwen2.5-1.5B-Instruct, gpt2) in the `ewm-nanolm` env: real `tid{n}` streams → union state, the pattern matrix `(LLM, Ring)` → `(LLM, Ring, DRN)` → `(Ring, Ring, DRN)`, and the discovered 16-step query cycle (`bss(t+16) == bss(t)`, rings plateau at dim 16) |
| 10 | `multi_llm_sidecar_loop` | closes the loop **around** the unchanged apparatus: DFT observer detects the query period from the BSS trajectory and frozen soft keys, then a controller feeds the materialized union state (restored order) back into the LLM prompts as memory and uses a DFT-periodic predictor to pick the next query (curiosity); the prediction error is the surprise training signal |
| 11 | `multi_llm_sidecar_predictor` | the predictor as a swappable **portfolio** — persistence / linear / DFT-periodic / online ridge behind one interface — and a meta-controller that tracks EWMA surprise per predictor and chooses ε-greedily online (the predictor layer trains itself to pick the most appropriate predictor); LeCun's encoder→context→predictor separation with HLLSets as the contexts and no decoder |
| 12 | `multi_llm_sidecar_learned_selector` | the meta-controller learns from trajectory features: `LearnedSelector` conditions on `[bss, Δbss, DFT period, EWMA surprises, phase]`, fits one linear model per predictor online to reward = −surprise, and drives the same memory+curiosity loop through the extracted `trainer/` package with the real LLMs; EWMA is still the stronger selector on the 16-step loop (0.95 vs 1.11 meta/best) |
| 13 | `multi_llm_sidecar_jev_router` | the first non-LLM actor in the loop: TypeSafe AI's **Jev** (System One) as router — the controller sends query + materialized memory + BSS to `JevAdapter`, Jev returns a typed `DecisionRecord` (choice + probabilities + confidence), only the chosen LLM answers, and ewm-sm ingests the answer unchanged; mock fallback runs when `TYPESAFE_API_KEY` is unset |
| 14 | `multi_llm_sidecar_jev_patterns` | the two Jev patterns that matter for ewm-sm, built with the mock while the API key is waitlisted: **confidence-gated routing** (below a floor, route to a fallback LLM) and **decision-in-state** (the `DecisionRecord` is lowered to `jev_*` tokens and ingested into `S(t)`, so the decision joins the memory); live-Jev ready by setting `TYPESAFE_API_KEY` |
| 15 | `hetero_sidecar_ocr_vla_jepa` | the realistic heterogeneous side-car: three non-LLM front-ends — **DeepSeek-OCR** (vision-encoder token ids), **NVIDIA Cosmos-Policy VLA** (quantized action chunk), **V-JEPA** (quantized patch ids) — captured first in their native envs (`captures/capture_*.py`), then one union `S(t) = S_ocr ∪ S_vla ∪ S_jepa` through the unchanged ewm-sm: `(Frontend, Ring)` pattern matrix, Noether D/R/N, materialized memory |
| 16 | `multi_llm_sidecar_rust_laya` | the open-source decision model in the loop: **Laya** (Apache-2.0, ungated, 421M, non-autoregressive typed decisions) runs as a persistent pure-Rust candle daemon behind the same `DecisionRouter` seam as Jev — no API key, no Python model; confidence gating + decision-in-state reuse notebook 14 with real calibrated probabilities (here the near-uniform confidences gate all steps to the fallback — the gate working as designed) |
| 17 | `multi_llm_sidecar_rust_laya_prompts` | the article's one failure mode, measured and fixed: an A/B over the eight queries (model-name options vs answer-type phrases, prose state) roughly doubles Laya's confidence and rebalances its choices; the improved router loop on the real LLMs keeps gating low-confidence steps to the fallback |
| 18 | `hetero_sidecar_rust_laya_router` | Laya routes the **heterogeneous** front-ends of notebook 15: each step the typed decision model picks OCR / VLA / JEPA for the query and that front-end's captured frame joins the union — options are genuinely different worlds, so Laya separates them (ocr 9 / vla 2 / jepa 13) with the same gating + decision-in-state |

Notebooks 01–04 were updated with the aarambh-vision-studio revisions:
**01** adds structural commits (basis change) and the `ewm-ops` operational
graph; **02–04** add **basis frames / time travel** — every frame is also
projected into the first interpretation with its spill
(`docs/BASIS_FRAMES.md`); **04** additionally rebuilds the union as an
operational graph.

### JEPA cooperation (notebook 07)

Notebook 07 pairs this repo with the
[`ewm-jepa`](https://github.com/alexmy21/ewm-jepa.git) project: V-JEPA
produces the `tid{n}` streams, and the Rust binaries (`ewm-scene`,
`ewm-app`, `ewm-ops`) provide the lattice side.

#### **Prerequisites**

- Clone `ewm-jepa` somewhere near this repo:

  ```bash
  git clone https://github.com/alexmy21/ewm-jepa.git ../ewm-jepa
  ```

- A JEPA-capable Python environment with `torch`, `ewm_jepa`, and
  `hllset_py` (the `ewm-jepa` conda env if available) registered as a
  Jupyter kernel named `ewm-jepa`.

#### **Run**

```bash
jupyter notebook notebooks/07_jepa_ewm_state_machine_cooperation.ipynb
```

The notebook locates both projects automatically (marker search over the
notebook root, its ancestors and their subdirectories). Overrides are
available if the projects live elsewhere:

```bash
EWM_JEPA=/path/to/ewm-jepa EWM_SM=/path/to/ewm-state-machine \
  jupyter notebook notebooks/07_jepa_ewm_state_machine_cooperation.ipynb
```

Without the JEPA side the setup cell stops with instructions on where to
clone it and what the `EWM_JEPA` variable should point at.

### Multi-LLM side-car real (notebooks 09–18)

Notebook 09 runs the same side-car as notebook 08 with three **real small
LLMs** on the `ewm-nanolm` kernel. Notebook 10 keeps the same probes and
closes the loop around the apparatus with a DFT observer + memory/curiosity
controller. Notebook 11 isolates the predictor as a portfolio
(persistence / linear / DFT-periodic / online ridge) with an ε-greedy
meta-controller that learns which predictor to trust. Notebook 12 replaces
the EWMA selector with a `LearnedSelector` that conditions on trajectory
features and refits one linear model per predictor online. Notebook 13 adds
TypeSafe AI's **Jev** (System One) as a typed-decision router, notebook
14 adds its confidence-gated routing + decision-in-state patterns, notebook
15 generalizes the side-car to **three heterogeneous non-LLM front-ends**
(DeepSeek-OCR + NVIDIA VLA + V-JEPA) via capture-first JSONLs, notebook
16 swaps in the open-source **Laya** decision model on a pure-Rust candle
daemon, notebook 17 measures and fixes the prompt failure mode, and
notebook 18 lets Laya route the heterogeneous front-ends. The 09–14 models
are loaded from the local HuggingFace cache on the RTX 3060
(`CUDA_VISIBLE_DEVICES=1`):

- `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`
- `Qwen/Qwen2.5-1.5B-Instruct`
- `gpt2`

#### **Prerequisites**

- The `ewm-nanolm` conda env (PyTorch cu124 + transformers), registered as a
  Jupyter kernel named `ewm-nanolm`:

  ```bash
  /home/alexmy/.conda/envs/ewm-nanolm/bin/python -m ipykernel install \
      --user --name ewm-nanolm --display-name "Python 3 (ewm-nanolm)"
  ```

- The three models cached under `~/.cache/huggingface/hub` (the notebooks use
  `local_files_only=True`).
- A GPU with ≥ 8 GB free (the three fp16 models total ~5.6 GB).
- For notebooks 13–14 with the real Jev model: `pip install typesafe-sdk` and
  `export TYPESAFE_API_KEY=...` (without the key the notebooks use the
  deterministic mock router).
- For notebook 15: the three captures under `~/.cache/ewm-hetero/`, produced
  by `captures/capture_*.py` in their native envs (see the capture scripts'
  docstrings); the notebook itself runs on the plain `python3` kernel.
- For notebooks 16–18: the Rust Laya daemon `laya-jsonl`
  (`/home/alexmy/tools/laya-rust`) and the checkpoint
  `/home/alexmy/.cache/laya/typed-decisions` (Apache-2.0, ungated;
  `LayaAdapter` falls back to a mock when either is missing). Notebook 18
  reuses the notebook-15 captures and runs on the plain `python3` kernel.

#### **Run**

```bash
jupyter notebook notebooks/09_multi_llm_sidecar_real.ipynb
jupyter notebook notebooks/10_multi_llm_sidecar_loop.ipynb
jupyter notebook notebooks/11_multi_llm_sidecar_predictor.ipynb
jupyter notebook notebooks/12_multi_llm_sidecar_learned_selector.ipynb
jupyter notebook notebooks/13_multi_llm_sidecar_jev_router.ipynb
jupyter notebook notebooks/14_multi_llm_sidecar_jev_patterns.ipynb
jupyter notebook notebooks/15_hetero_sidecar_ocr_vla_jepa.ipynb
jupyter notebook notebooks/16_multi_llm_sidecar_rust_laya.ipynb
jupyter notebook notebooks/17_multi_llm_sidecar_rust_laya_prompts.ipynb
jupyter notebook notebooks/18_hetero_sidecar_rust_laya_router.ipynb
```

The notebooks set `CUDA_VISIBLE_DEVICES=1` before importing torch; override
it in the first cell if the RTX 3060 is not the target GPU.
