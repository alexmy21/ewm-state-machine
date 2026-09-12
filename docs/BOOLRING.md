# Boolean ring over HLLSets — context-index spike

**Status:** spike complete, result positive (see below). This is a *local*
context index, not a foundation change and not a global content address.

## Structure

An HLLSet is a subset of the bit plane, hence a vector over GF(2):

```text
symmetric difference Δ = addition      x + x = 0
intersection        ∩ = multiplication  x · x = x
A ∪ B = A Δ B Δ (A ∩ B)
```

`ewm-boolring` provides three primitives over HLLSets:

- `symmetric_difference(a, b)` — GF(2) addition;
- `BoolBasis::insert(set)` — Gaussian elimination into a **reduced
  row-echelon** basis; returns `InSpan { coords }` (the set is an XOR of
  existing basis elements) or `Added { residual, pivot }`;
- `BoolBasis::{coordinates, residual, dimension}` — the unique GF(2)
  coordinates of a span member, and the **residual** of any set: its linear
  novelty, the part not expressible from the context so far.

## Scope limits (the basis-matching problem)

The basis depends on insertion order and pivot choice — there is **no
canonical GF(2) basis**, so coordinates only mean something inside one
context. The structure is scoped to the ewm-sm context by design. (A
canonical alternative exists — the atoms of the Boolean algebra generated
by the sets — but it is exponential in the number of generators.)

Coordinates also do not preserve `popcount(A Δ B)` distance unless the
basis is disjoint; treat them as relational structure, not a metric
embedding.

## Spike result

Pseudo-clip mirroring the scene structure (10 scenes × 10 frames, 256
tokens/frame, 200-token scene base + 56-token per-frame drift), analyzed
incrementally as frame HLLSets arrive:

```text
dimension after 100 frames       : 100   (one new direction per frame)
linear novelty at scene cuts     : mean 782.9  (min 677, max 1058)
linear novelty in-scene          : mean 271.4  (min 69, max 680)
separation (mean cut / in-scene) : 2.9x
cuts above in-scene mean + 2σ    : 9/9   [11, 21, 31, 41, 51, 61, 71, 81, 91]
```

**The residual spikes at scene cuts** — the linear-novelty series detects
all nine cuts with a simple threshold, and is conceptually a higher-order
`N`: *"cannot be assembled from anything the context has seen,"* vs the
bit-level `N = S(t) \ H(t-1)`.

## Verdict

The direction earns its place as an ewm-sm context index. Suggested
integration: maintain a `BoolBasis` per context (session), expose
`dimension`, `residual popcount`, and `coordinates` per turn in the
explorer, and let the residual feed the accident-frame selection example —
frames around a residual spike are the ones to materialize.
