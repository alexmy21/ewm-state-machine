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

## The HLLSet lattice — one universe, nodes discovered not created

Every HLLSet is a node of the Boolean lattice over the bit plane:

```text
order   A ⊆ B                 (bitwise subset)
join    A ∪ B                 (union)
meet    A ∩ B                 (intersection)
Δ       (A ∪ B) \ (A ∩ B)     (ring addition = symmetric difference)
```

Every result of every morphism — frame states, D/R/N sets, unions,
intersections, residuals, basis elements — is a node that **already exists**
in this lattice. Nothing new is created: a computation *discovers* a node by
naming a lattice expression, and the node's content key is its name. Two
different paths reach the same node iff their keys agree — that is the
tangible test of "discovered, not created" (notebook 04 verifies it: the
pyramid's union join and the token-concatenation ingest produce the same
keys).

The decompositions look separate only because the LLM context exposes
different aspects of the same lattice:

- the joined perceptrons are the **join** of their components,
  `u-HLLSet(t) = ⋁ p_i-HLLSet(t)`;
- the D/R/N of a transition are the three disjoint pieces of the same two
  nodes: `D = A \ B`, `R = A ∩ B`, `N = B \ A`, with `D ∪ R = A`,
  `N ∪ R = B`, `D ∩ N = ∅`;
- the ring basis picks a **sublattice frame** (a set of independent nodes)
  through which every other node is read.

The "holographic" property is literal in a Boolean lattice: any node is the
join of the atoms below it, so its full relation set (meets with all other
nodes) reconstructs it. A decomposition frame is simply a chosen set of
dimensions through which those relations are read — a partial view of one
node in one universe, not a new universe per decomposition.

### The morphism is the contract (IICA)

"Tokens are generators of lattice nodes" is true as procedure, not as
mechanism. The final bit vector **does not remember how its bits got
flipped**, and there is no physical connection between a token and "its"
bits. The connection is the chosen morphism

```text
μ : {token collections} → {bit vectors},   A ↦ μ(A)
```

and the only thing the model relies on is that `μ` satisfies **IICA** —
Idempotence, Immutability, Content Addressability: the same token collection
always maps to the same bits (deterministic), bits are never mutated in
place (only new nodes appear), and the bit vector is addressed by its content
key. Any algorithm that meets this contract is interchangeable — the LUT and
n-gram seeding are the current implementation, not part of the model.

What structure of the token-side lattice survives in the image is exactly
what the chosen morphism preserves:

- the **flat projection** is a join homomorphism — each token sets its own
  LUT bits, so `μ(A ∪ B) = μ(A) ∪ μ(B)`. Notebook 04's 48/48 union-key
  agreement is this fact;
- **windowed n-gram ingestion** is only inclusion-isotonic —
  `A ⊆ B ⟹ μ(A) ⊆ μ(B)` — so union identities survive as inclusions, not
  equalities.

Both are valid IICA morphisms; they are different *measurements*, and the
lattice facts of the previous section hold in whichever image the chosen
morphism produces.

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
  in_span, dimension, rotation_count, rotation_mass }` (the incoming set's
  linear novelty against the window *before* insertion, plus the rotation
  component of the basis change);
- eviction — the oldest original leaves; the basis is recomputed from the
  remaining window (cheap at cache sizes);
- `residual(set)` / `coordinates(set)` — evaluate any (compound) set
  against the window basis without inserting it.

### The basis change decomposes into extension + rotation

When a set is inserted outside the span, the basis change has two parts:

- **extension** — the residual `R` becomes a new basis element with a new
  pivot `p` (the span grows by one direction);
- **rotation** — every existing basis element that contains bit `p` is
  re-pivoted: `B_i ← B_i Δ R`, keeping the basis in reduced row-echelon form.

`rotation_count` is the number of existing basis elements re-pivoted;
`rotation_mass = rotation_count · residual` is the total Hamming change of
the old basis. An in-span push never changes the basis, so both are zero.
The whole basis content change is `(rotation_count + 1) · residual` — the
extension plus the rotation. The rotation is a **second-order novelty
signal**: it is zero whenever the residual is zero, and among novel frames it
measures how many existing directions re-index themselves through the new
one (the re-indexing cost of the context). For sparse per-frame HLLSets it
is a sparse, noisy re-weighting of the residual; for dense working sets it
is the stronger signal.

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
