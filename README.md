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
│   └── ASSIGNMENT_QWENDRIVE.md # next assignment — Qwen-Drive test bench
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
