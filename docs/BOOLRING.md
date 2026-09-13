# Boolean ring over HLLSets — context-index spike

**Status:** implemented — spike positive, windowed ring wired into the
[UM] cache and the explorer. This is a *local* context index, not a
foundation change and not a global content address.

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

## Windowed ring over originals

The basis problem is resolved by **fixing the generators and their order**:
only *original* HLLSets (the per-turn ingest projections) enter the ring,
in ingestion order, bounded by the cache. The basis of a window sequence is
therefore deterministic — the same sequence always reduces to the same
basis, so coordinates are comparable inside the window.

`BoolWindow` (in `ewm-boolring`):

- `push(original)` — one Gaussian step; returns `RingStats { residual,
  in_span, dimension }` (the incoming set's linear novelty against the
  window *before* insertion);
- eviction — the oldest original leaves; the basis is recomputed from the
  remaining window (cheap at cache sizes);
- `residual(set)` / `coordinates(set)` — evaluate any (compound) set
  against the window basis without inserting it.

Compounds are **evaluated, not inserted**: `A Δ B` is in the span
(coordinates); `A ∩ B` and the D/R/N differences usually are not — their
residual is the *multiplicative* novelty, the part no XOR of originals can
express.

Wiring: `StateCache.ring` (capacity `RING_CAPACITY = 64`) is pushed by
`run_turn`, rebuilt by `StateCache::restore` from the committed turns in
order, and projected as `ring { capacity, window_len, dimension }` in the
explorer snapshot.

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

**Real-clip note** (notebook 02, the CLIP-quantized synthetic clip): the
separation holds (mean cut novelty 279.9 vs 71.2 in-scene, ~3.9x) but the
in-scene variance is high, so the 2σ threshold flags only 3/9 cuts. The
reason: the quantization codebook is shared, so scenes reuse tids and the
window span absorbs some cut novelty. The residual is a complement to the
cosine/BSS cut line, not a replacement — it flags the hardest cuts.

## Verdict

The direction earns its place as an ewm-sm context index, and the windowed
ring is now implemented: originals-only generators, ingestion order, cache
bounded, deterministic per window. The residual feeds the
accident-frame selection example — frames around a residual spike are the
ones to materialize. Integration tests cover window novelty, span
membership, the snapshot projection, and deterministic rebuild on restore.
