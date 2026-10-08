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

"Tokens generate lattice nodes" is true as procedure, not as
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
- `BoolBasis::insert(set)` — Gaussian elimination into a **row-echelon**
  basis (distinct leading bits); returns `InSpan { coords }` (the set is an
  XOR of existing basis elements) or `Added { residual, pivot }`;
- `BoolBasis::{coordinates, residual, dimension}` — the unique GF(2)
  coordinates of a span member, and the **residual** of any set: a remainder
  modulo the span (zero exactly when the set is in the span) — its linear
  novelty, the part not expressible from the context so far.

## Ring over originals: monotone basis, visibility window

The basis problem is resolved by **fixing the commit HLLSets and their order**:
only *original* HLLSets (the per-turn ingest projections) enter the ring, in
ingestion order. Two roles are decoupled:

- the **basis is monotone** — every commit HLLSet ever pushed stays in the
  base's span, so `span(t) ⊆ span(t+1)`: a set that is representable
  stays representable, and every commit HLLSet remains an XOR of the basis;
- the **visibility window** (capacity `RING_CAPACITY = 64` in the [UM] cache)
  is the recent originals the ring exposes as *context*. Eviction moves this
  window only; it never removes a base HLLSet.

Because the commit sequence is fixed by ingestion, the basis is
deterministic — the same sequence always reduces to the same basis, so
coordinates are comparable across the run (and across restores).

`BoolWindow` (in `ewm-boolring`):

- `push(original)` — one Gaussian step; returns `RingStats { residual,
  in_span, dimension, rotation_count, rotation_mass }` (the incoming set's
  linear novelty against the span *before* insertion, plus the rotation
  component of the basis change);
- visibility — the oldest original leaves the window once `capacity` is
  exceeded; `window_len()` stays bounded while `total()` counts the
  base HLLSets;
- `residual(set)` / `coordinates(set)` — evaluate any (compound) set against
  the span without inserting it;
- `BoolWindow::windowed(capacity)` — the legacy scene-bounded mode, where the
  window *is* the base set and eviction rebuilds the basis. Kept for
  reproducing the earlier windowed spike; production uses the monotone mode.

### The basis change decomposes into extension + rotation

When a set is inserted outside the span, the basis change has two parts:

- **extension** — the residual `R` becomes a new basis element with a new
  pivot `p` (the span grows by one direction);
- **rotation** — every existing basis element that contains bit `p` is
  re-pivoted: `B_i ← B_i Δ R`. Leading bits stay distinct (row-echelon; the
  basis is not fully reduced — see the representation note below).

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
express. Because the span is monotone, a compound that is outside at `t` can
*become* representable once later originals grow the span; pushing it as an
original makes that permanent.

### What the ring can represent (the meet obstruction)

The span `S = span_GF(2)(originals)` is a vector subspace, so it is
closed under `Δ` but generally *not* under `∩`, `∪` or `\`. For `A, B ∈ S`,
distributivity of `∩` over `Δ` gives

```text
A ∪ B = A Δ B Δ (A ∩ B)   ->   A∪B ∈ S  ⟺  A∩B ∈ S
A \ B = A Δ (A ∩ B)       ->   A\B ∈ S  ⟺  A∩B ∈ S
A Δ B                     ->   AΔB ∈ S  always
```

So `A∪B`, `A∩B` and `A\B` are pairwise congruent modulo `S`: they lie in one
coset `X + S` and are in-span together or outside together. The **meet is the
single obstruction**. A general expression in ∪, ∩, \ over in-span sets is an
XOR of basis elements iff every meet subterm is in `S`; the exact test for
any set `X` is `residual(X) = ∅` (equivalently `coordinates(X)` is `Some`).
`S` is closed under `∩` iff it admits a pairwise-disjoint GF(2) basis — i.e.
iff it is a Boolean subalgebra, the unions of the blocks of a partition. For
generic HLLSets that essentially never holds (the lattice they generate has
up to `2^n` atoms against the ring's `n` dimensions), so the ring is a **lossy
linear projection of the lattice**. When a compound is outside the span there
is no exact representation; the only exact record is
`X = (X Δ residual(X)) Δ residual(X)`, with the first factor in the span.

**Remainder, not canonical representative.** `residual` is deterministic for
a fixed commit sequence but is only *a* representative of the coset
`X + S`: `insert` keeps distinct leading bits without fully reducing, and
`reduce` stops at the first non-pivot lowest bit, so congruent sets can return
different remainders (and a remainder may even contain a pivot bit). The
coset-invariant fact — and the one the tests rely on — is
`residual(X) = ∅ ⟺ X ∈ S`. Notebook
[`22_boolring_window_over_originals.ipynb`](../notebooks/22_boolring_window_over_originals.ipynb)
illustrates push/visibility/compound evaluation and measures these limits on
real HLLSets.

Wiring: `StateCache.ring` is a monotone `BoolWindow` with visibility capacity
`RING_CAPACITY = 64`. It is pushed by `run_turn`, rebuilt by
`StateCache::restore` from every committed turn in order (so the commit
history is complete after a restart), and projected as
`ring { capacity, window_len, total, dimension }` in the explorer snapshot.

### The ring + lattice toolkit

The ring and the lattice are kept together because each is exact at something
the other is not — the ring gives canonical coordinates, the lattice gives the
BSS measurement — and the gap between them is itself a quantity. The
primitives live on `BoolBasis` and are demonstrated in notebook
[`23_ring_lattice_tools.ipynb`](../notebooks/23_ring_lattice_tools.ipynb):

| tool | exactness | primitive |
| - | - | - |
| (a) base transform `B(t-1) -> B(t)` | exact GF(2) change-of-basis matrix; coordinates transport by `a · M` | `change_of_basis` |
| (b) BSS transform `BSS(t-1) -> BSS(t)` | **not linear** (XOR vs `∩`); exact by re-projection, or by the meet-order ladder (`order ≥ support size`); genuinely new directions must be measured, not rotated | `bss_vector` |
| (c) `union(basis) = union(originals)` | exact (`XOR ⊆ OR`, and commit HLLSets are XORs of the basis) | `cover` |
| (d) projection `X -> (D, R, N)` onto the cover | exact lattice ops; `N ⊆ residual(X)` | `project` |
| (e) minimal cover of a context set | unique coordinate support when in span; greedy (H(d)) union cover otherwise | `linear_cover` / `minimal_cover` |

The practical reading: transport **coordinates** with the change-of-basis
matrix; re-project **BSS** (do not try to map it linearly). `N` is the hard
bit-novelty no span member can contain and the ring residual is the linear
novelty (`N ⊆ residual`); the cover is the context ceiling and `X \ cover` its
spill; the minimal cover compresses a context into a few basis directions with
the residual as the irreducible remainder.

### Views vs compounds — what the store keeps

The change of basis is not only bookkeeping; it decides what we persist. Only
two things are materialized:

- **originals** — the IICA sketches of measured token streams, content-addressed
  (`h:<sha1>`), immutable;
- **base HLLSets** — the monotone basis the ring produces, themselves
  content-addressed.

Everything else — unions, `D/R/N`, residuals, soft keys, covers, projections —
is a **view**: an *expression* over source nodes plus its **provenance** (the
ordered list of SHA-1 CIDs of the HLLSets it is built from), never a copy of the
bits. The reason is the generation stamp: coordinates are basis-relative and the
basis rotates (extension + rotation), so a compound frozen as a coordinate
vector, or as a bit copy taken in one generation, is silently wrong when read in
another. The store keeps the recipe; the reader **recreates** the compound
against the base current at read time.

Recreation is safe (IICA: same recipe + same sources ⇒ same bits) and cheap (a
handful of bit operations per base HLLSet, independent of history length). Caching
is optional and subordinate: a materialized view may be cached, but the entry is
stamped with its basis generation, and a read in a new generation re-projects
only the entries queried — never the whole history. Storage is proportional to
the number of originals plus the size of the base, not to the number of
compounds ever named: the operational form of "discover, don't create".

```text
persisted (bits)            recorded (recipe + provenance)        computed (read)
h:<sha1> original   ──▶     view  p:<sha1>                    ──▶  recreate against
h:<sha1> base       ──▶       expr       = union | drn | ...       B(t) at read time
                              sources    = [h:.., h:.., ...]       (cache is stamped
                              basis_gen  = g                        with generation g)
```

In code: originals and base HLLSets are content-addressed HLLSet nodes; views are
the `ewm-ops` program lattice (`p:<sha1>`) with the content-addressed vocabulary
(`v:<sha1>`), and provenance is the list of source `h:` CIDs.

## Scope limits (the basis-matching problem)

The basis depends on insertion order and pivot choice — there is **no
canonical GF(2) basis**, so coordinates only mean something inside one fixed
commit sequence. The structure is scoped to the ewm-sm context by design.
(A canonical alternative exists — the atoms of the Boolean algebra generated
by the sets — but it is exponential in the number of commit HLLSets.) The
monotone basis keeps coordinates comparable across the whole run; they are
still not comparable to a different ingestion order.

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
span absorbs some cut novelty. The residual is a complement to the
cosine/BSS cut line, not a replacement — it flags the hardest cuts. (That
series predates the monotone basis: it is the legacy `BoolWindow::windowed`
mode. The monotone sidecar measures *cumulative* novelty against every
commit HLLSet ever pushed, so in-scene drift keeps the residual high and the
cut separation is weaker; the visibility window still bounds the context the
ring exposes.)

## Verdict

The direction earns its place as an ewm-sm context index, and the ring is now
implemented with a **monotone basis** (commit HLLSets only, ingestion
order — every commit HLLSet stays a base HLLSet) and a separate
**visibility window** (`RING_CAPACITY = 64`). The residual feeds the
accident-frame selection example — frames around a residual spike are the ones
to materialize. Integration tests cover novelty, span membership (monotone and
legacy windowed), the snapshot projection, and deterministic rebuild on
restore.
