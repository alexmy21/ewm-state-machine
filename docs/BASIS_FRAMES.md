# Basis frames — HLLSet interpretation, time travel, commit conditions

**Status:** implemented in `ewm-boolring` (generation stamp) and
`ewm-flux-host::sidecar` (`BasisSnapshot`, `BasisHistory`, `Projection`);
demonstrated by `notebooks/05_flux_sidecar.ipynb`.

This document makes formal what the Boolean-ring notes call a *frame*:
a ring basis is not just an index — it is an **interpretation** of every
HLLSet. Keeping the basis frames (instead of per-HLLSet BSS vectors) gives
two things that are usually informal in a side-car: a measurable
**time travel** (re-reading a current HLLSet through a past basis) and a
precise **commit condition** (the basis change itself is a structural
event).

---

## Part 1 — Mathematical ground of the HLLSet transformation

### 1.1 Setting

```text
𝔹  = {0, …, 32767}                the bit plane (M registers × 32 tz bits)
X ⊆ 𝔹                             an HLLSet (a bitmap over the plane)
(P(𝔹), ⊆, ∪, ∩)                  the Boolean lattice (join ∪, meet ∩)
(P(𝔹), Δ, ∩)                     the GF(2) ring, with Δ = symmetric difference
```

Every node of the lattice already exists; a computation only *names* it
(docs/BOOLRING.md). The morphism `μ : tokens → bits` satisfies IICA
(Idempotence, Immutability, Content Addressability) and is not part of the
structure below — everything here operates on the image of `μ`.

### 1.2 Frame — the basis as an interpretation

A **frame** is a named, ordered set of basis HLLSets together with their
span and cover:

```text
F = (B_1, …, B_k)                 an RREF GF(2) basis over some window
span(F) = { Δ_{i ∈ J} B_i : J ⊆ {1..k} }       the GF(2) span
cover(F) = B_1 ∪ … ∪ B_k                       the set-theoretic cover
dim(F)  = k                                    the frame dimension
```

`span(F)` is what the frame can **reconstruct**; `cover(F)` is what the frame
can **see**. They differ: a single bit can lie inside the cover but outside
the span (it is seen, but not assembled).

The **interpretation** of an HLLSet `X` under frame `F` is the pair
`(F, Π_F(X))`, where the projection is the triple:

```text
Π_F(X) = ( w, c, s )

  w_i = |X ∩ B_i| / |B_i|          soft key   ∈ [0,1]^k,  defined for ALL X
  c   ∈ GF(2)^k with               hard key   defined iff X ∈ span(F):
       X = Δ_{i : c_i = 1} B_i                 exact GF(2) coordinates
  s   = |X \ cover(F)|             spill      ∈ ℕ, the part F cannot see
```

The soft key is a **similarity profile** over the frame's directions; it
does not reconstruct. The hard key is the **structural address** — the exact
XOR decomposition, unique for span members. The spill is the **error of the
reading**: the bits of `X` that the interpretation cannot account for even
set-theoretically.

### 1.3 Properties

Let `res(X)` be the RREF residual of `X` against `F` (the part left after
Gaussian elimination; empty iff `X ∈ span(F)`).

**P1 — the soft key sees only the cover.**
`w_i(X) = w_i(X ∩ cover(F))` for every `i`. Each `B_i ⊆ cover(F)`, so the
intersections — hence the whole soft key — depend only on the part of `X`
inside the frame's field of view.

**P2 — spill is a subset of the residual.**
`X \ cover(F) ⊆ res(X)`, hence `s ≤ |res(X)|`. Proof: elimination XORs only
basis elements, and every basis element is a subset of `cover(F)`; therefore
bits of `X` outside the cover are never touched by any elimination step and
survive into the residual. The spill is the cheap, set-theoretic part of the
linear novelty.

**P3 — being in the span implies no spill.**
`X ∈ span(F) ⟹ X ⊆ cover(F) ⟹ s = 0`, because an XOR of sets is a subset of
the union of its operands. The converse is false:

```text
F = ({1,2})     span(F) = {∅, {1,2}}     cover(F) = {1,2}
X = {1}         s = 0                    but X ∉ span(F)
```

So `s = 0` is necessary, not sufficient, for span membership.

**P4 — hard-key existence is span membership.**
`c` is defined iff `res(X) = ∅` iff `X ∈ span(F)`; then `c` is unique and
`X = Δ_{i : c_i = 1} B_i`. The hard key is therefore the strongest reading,
and the soft key is its relaxation: a profile any `X` admits.

**P5 — original states are always visible to their own frame.**
Every original `O_j` of the window that built `F` satisfies
`O_j ∈ span(F)` and `O_j ⊆ cover(F)` — the basis spans the originals by
construction. Within the window, `spill = 0`; only when projecting a *new*
set into an *old* frame does the spill become nonzero.

**P6 — interpretation is deterministic.**
The frame sequence is fixed by the ingestion order of originals (IICA +
docs/BOOLRING.md): the same sequence always produces the same frames,
generations, and projections. Two readings are in the **same
interpretation** iff they carry the same basis generation.

### 1.4 Time travel

Let `F_g` be the frame recorded at basis generation `g`. Projecting a
current HLLSet `X_t` into an old frame `F_g` is **time travel**: the state
is re-read through a past interpretation.

```text
Π_{F_g}(X_t) = ( w_old, c_old?, s_old )
```

- `w_old` — how the present state lines up with the old directions; always
  defined (P1: the old frame only sees its own cover, and that is exactly
  what "old interpretation" means).
- `c_old` — defined iff the present state can be assembled **exactly** from
  the old directions; the strongest possible continuity statement.
- `s_old = |X_t \ cover(F_g)|` — the measurable error of the travel: how many
  bits of the present the past cannot see.

Time travel is not a reconstruction of the past; it is a *reading* of the
present through the past's lens, with a built-in error term. The pair
`(w_old, s_old)` is complete: profile over what the old frame sees, plus the
mass of what it misses.

### 1.5 The basis change as a structural event

A basis **changes** when a new original is outside the span (extension +
rotation) or when eviction rebuilds the basis. Each change is a structural
event — the context re-indexed itself — with two measurable magnitudes:

```text
rotation_count, rotation_mass     how many old directions re-pivot and the
                                  total Hamming change of the old basis
generation bump                   the event itself (monotone stamp)
```

The frame history `F_0, F_1, …, F_g` is the sequence of interpretations the
system passed through. Because a basis change is observable and grounded, it
is the natural **second commit condition** next to `new bits > 0`:

```text
commit when:  new bits > 0   (content changed)
           or basis changed  (interpretation changed)
```

---

## Part 2 — Implementation aspects

### 2.1 Where the pieces live

| Piece | Crate / module | Notes |
| --- | --- | --- |
| GF(2) RREF basis (`reduce`, `coordinates`, `residual`, `insert`) | `ewm-boolring::BoolBasis` | unchanged foundation |
| window over originals, eviction, **`generation()`** | `ewm-boolring::BoolWindow` | generation bumps exactly when the basis content changes: `Added` insert or `recompute()`; in-span pushes do not bump |
| `BasisSnapshot { generation, basis, cover }` | `ewm-flux-host::sidecar` | `cover` precomputed as `⋃ B_i`; `project(&self, X) -> Projection` |
| `BasisHistory { snapshots }` | `ewm-flux-host::sidecar` | `push_if_changed(&window)` records one frame per generation bump; `project_into(i, X)` |
| `Projection { soft, spill, hard }` | `ewm-flux-host::sidecar` | the `Π_F(X)` triple |
| per-step records (`basis_generation`, `basis_change`, `soft`, `hard`, `step_len`) | `ewm-flux-host::sidecar::SidecarFrame` | online view, measured against the pre-push basis |
| consistent final view (`soft_final`, `step_final`, `spill_final`) | `ewm-flux-host::sidecar::StepFrame` / `FluxReport` | every step re-projected into the final frame |
| report aggregates (`basis_changes`, `basis_generations`, `final_projection`) | `ewm-flux-host::sidecar::FluxReport` | `final_projection = { dimension, generation, basis_pop, cover_pop, spill_mean, spill_oldest, jumps, jump_threshold }` |

The same algebra exists in `ewm-scene::sidecar_series` (the Qwen/scene
bench); it currently records per-step soft keys and can be upgraded to the
frame-history design as a porting note.

### 2.2 Data structures and cost

`HLLSet` is a roaring bitmap over the 32768-bit plane. One soft-key
projection is `k` intersections + popcounts; the spill is one difference +
popcount against the precomputed cover. Measured on synthetic sets matching
the flux run (800-bit query set, `k = 64` basis elements of 100–1300 bits):

| Operation | dev | release |
| --- | --- | --- |
| BSS projection, `k = 64` | 0.57 ms | **0.41 ms** |
| cover `⋃ B_i` (once per frame) | 0.29 ms | 0.22 ms |
| spill `|X \ cover|` | 2.0 µs | 1.2 µs |

**Decision: do not store BSS vectors per HLLSet.** Storing vectors costs
`O(n·k)` floats and goes stale on every basis change. Storing frames costs
`O(g·k)` HLLSets (`g ≤ n` basis changes) and never goes stale; projections
are recomputed on demand at sub-millisecond cost. The 29-step default run
pays ~12 ms for the whole consistent matrix.

### 2.3 Algorithms

```text
BasisSnapshot::new(g, basis)
    cover ← ⋃ B_i                                  (one union fold)

BasisSnapshot::project(X)
    soft  ← [ |X ∩ B_i| / |B_i|  for i = 1..k ]    (P1: only the cover is seen)
    spill ← |X \ cover|                            (one difference + popcount)
    hard  ← basis.coordinates(X)                   (Some iff X ∈ span(F))
    return (soft, hard, spill)

BasisHistory::push_if_changed(window)
    if window.generation() != last recorded generation:
        snapshots.push(BasisSnapshot::new(window.generation(), window.basis()))

sidecar_series_with_window(originals)
    for each original X_t:
        measure soft, hard against the pre-push basis      (online reading)
        record basis_generation ← window.generation()       (pre-push)
        stats ← window.push(X_t)
        record basis_change ← window.generation() != basis_generation
        history.push_if_changed(&window)                    (frame history)

run()
    F ← history.last()                                      (final interpretation)
    for each original X_t:
        (soft_final_t, hard_final_t, spill_final_t) ← F.project(X_t)
    step_final_t ← L2(soft_final_{t-1}, soft_final_t)
    spill_oldest ← history.first().project(last original).spill
```

### 2.4 JSON report fields (the time-travel surface)

```jsonc
{
  "frames": [{
    "step": 16,
    "basis_generation": 16,       // interpretation this step was read in
    "basis_change": true,         // structural event (commit condition)
    "soft":  [ ... ],             // online soft key (pre-push basis)
    "step_len": 0.19,
    "soft_final":  [ ... ],       // soft key in the FINAL interpretation
    "step_final":  0.17,          // L2 between consecutive soft_final rows
    "spill_final":  0             // |S(t) \ cover(final)| — 0 inside the window
  }],
  "final_projection": {
    "dimension": 29, "generation": 29,
    "basis_pop": [ ... ], "cover_pop": 2656,
    "spill_mean": 0.0,            // 0: the final frame spans all steps
    "spill_oldest": 429,          // newest step read in the OLDEST frame
    "jumps": [ ], "jump_threshold": 0.3848
  },
  "basis_changes": 29,            // number of interpretation changes
  "basis_generations": [1, ..., 29]
}
```

`spill_oldest` is the canonical demonstration: the last step, read through
the first frame, spills 429 bits — the present through the past's lens, with
the error made explicit.

### 2.5 Commit condition (wiring point)

The condition is measured and recorded in the side-car, but not yet wired
into the [UM] commit rule. The intended change in `ewm-app::run_turn`:

```rust
let gen_before = cache.ring.generation();
let ring_stats  = cache.ring.push(&turn_g1);
let basis_changed = cache.ring.generation() != gen_before;

let commit = if self.repo.new_bits(&state) == 0 && !basis_changed {
    None
} else {
    Some(self.repo.commit(&state, &parents, &message)?)
};
```

This makes every structural event durable in the commit DAG, even when the
cumulative bit content did not change (e.g., a window slide that re-indexes
the context).

### 2.6 Test coverage

- `ewm-boolring`: generation bumps exactly on basis changes; eviction
  rebuild bumps; in-span pushes do not.
- `ewm-flux-host`: frame history records every basis change and skips
  in-span pushes; `project_into` gives `(soft, hard, spill)` with the
  `spill = |X \ cover|` invariant; `soft_final` rows share one dimension;
  `step_final` matches the L2 of consecutive consistent rows;
  `spill ≤ cover_pop`; the oldest-frame spill of the newest step is
  nonzero.
- `notebooks/05_flux_sidecar.ipynb`: end-to-end demonstration (basis
  changes 29/29, cover pop 2656, mean spill 0.0, oldest-frame spill 429).
