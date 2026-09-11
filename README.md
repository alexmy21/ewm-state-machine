# ewm-state-machine

The **state-centric** home of the fractal_manifold_gen2 collection. The
accent here is not HLLSet — it is the **state of the system, S(t)**.

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
materialization** (LUT-first, with TF disambiguation): ingest lifts tokens
into bits, materialize lowers bits back into tokens.

S(t) is the **state in a stateless system**. IICA — Idempotence,
Immutability, Content Addressability — removes the contradiction: S(t) is an
immutable, content-addressed value, so it is safe to share. The [UM]
(processing unit) that works on S(t) is stateless and disposable; it reads
the tip, processes incoming tokens, and proposes the next state.

S(t) and H(t-1) live **outside the [UM]**, in a shareable cache
(`ewm-app::StateCache`), backed by the persistent store (`ewm-git`):

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

Terminology reference:
`../hllset-next-v2/_DOCS/dev/STANDARD.md`.

## Layout

```text
ewm-state-machine/
├── Cargo.toml                 # workspace manifest
├── docs/
│   ├── ARROW_CACHE.md         # design note: Arrow as the cache-layer substrate
│   └── EXPLORER.md            # the read-only explorer contract
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
    └── (planned) ewm-cache/   # Arrow-backed cache layer, per docs/ARROW_CACHE.md
```

The Arrow cache layer is designed but not implemented; see
[`docs/ARROW_CACHE.md`](docs/ARROW_CACHE.md) for the boundary, schemas, and
the IPC extended-cache layout.

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

## Notebooks

The notebook is the application: each code cell is a step, and the [UM] runs
the cells.

| # | Notebook | Description |
| - | -------- | ----------- |
| 01 | `ingest_materialize_um` | morphisms step by step — ingest (SHA1 + three pointers + hllsetLUT), materialize (ordered / `no_order`), then the [UM] loop and recovery |
