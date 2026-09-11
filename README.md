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

S(t) is the **state in a stateless system**. IICA — Idempotence,
Immutability, Content Addressability — removes the contradiction: S(t) is an
immutable, content-addressed value, so it is safe to share. The [UM]
(processing unit) that works on S(t) is stateless and disposable; it reads
the tip, processes incoming tokens, and proposes the next state.

Recovery is a **pop, not a rebuild**: drop the broken top of the stack, the
previous committed state becomes the tip, a fresh [UM] resumes processing.

## Vocabulary

| Term | Meaning |
| ---- | ------- |
| S(t) | current emerging state (work in progress, shared) |
| H(t-1) | previous committed state (the cache) |
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
└── crates/
    ├── hllset-contracts/      # soldered invariants (leaf)
    ├── hllset-cid/            # embedded SHA-1 CIDs
    ├── hllset-core/           # HLLSet bit-plane — the state representation
    ├── hllset-lut/            # the LUT lattice (token interpretation)
    ├── hllset-morphisms/      # ingest/materialize + default api (SHA1 + hllsetLUT)
    ├── hllset-storage/        # memory + sled content-addressed storage
    ├── context-tree/          # Merkle tree over the working set (S(t) presentation)
    ├── ewm-git/               # the state stack: commit DAG, tip, recovery
    └── ewm-app/               # the [UM] harness: stateless driver loop
```
