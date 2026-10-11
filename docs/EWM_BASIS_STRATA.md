# Basis strata — stratifying the ring base by birth commit

**Status:** implemented for §1–§6 in `ewm-boolring` (`BirthTable`, flag query
API) and `ewm-app` (`StateCache` wiring, restore replay). Notebook prototypes:
`notebooks/27_basis_strata.ipynb`, `notebooks/28_trajectory_optim.ipynb`,
`notebooks/29_wishful_forecast.ipynb`. §7 (the CP model) and §8 (the wishful
forecast loop) are **future directions** — notebook-only until deeper
prototyping. The design specifies a **birth table** over the monotone Boolean
ring (`docs/BOOLRING.md`) that stratifies the base HLLSets by the commit in
which each direction first occurred, and gives the two applications that
motivate it: **host LLM context management** and **historical state
exploration under trajectory ambiguity**.

Builds on `docs/BOOLRING.md` (the monotone basis, extension + rotation),
`docs/BASIS_FRAMES.md` (generation stamps, time travel, the basis-change
commit condition), and `docs/EWM_RULES.md` (stratification, world-relative
programs).

---

## 1. The one correction to the naive picture

The intuition is:

```math
\text{B(1)} \rarr \text{B(2)} \rarr \text{B(3)} \rarr … \rarr \text{B(t)}
```

Basis elements are GF(2)-independent and rotate at later births (`B_i ← B_i Δ R`). The chains
hold

```text
span(t) ⊆ span(t+1)        the GF(2) span of the base
cover(t) ⊆ cover(t+1)      the set-theoretic union of the base
U(t) ⊆ U(t+1)              the lattice top (union of committed originals)
```

The correct object for "stratify by first occurrence" is the **birth
residual**. When `BoolBasis::insert` accepts a set outside the span it returns
`Added { residual R, pivot p, rotation_count }`. The residual `R_i` of the
`i`-th such event satisfies

```text
R_i ∈ span_i \ span_{i-1}
```

so the sequence `R_1, …, R_t` is a basis **adapted to the flag**

```text
0 = span_0 ⊂ span_1 ⊂ span_2 ⊂ … ⊂ span_t.
```

The current RREF basis is a *different* basis (pivot-adapted): after
rotations, slot `i` is `R_i Δ (some later R_j)`, so it is not flag-adapted.
The birth residuals are the stable identities; the RREF slots are the working
coordinates. Both are needed, for different jobs.

---

## 2. The birth table

One record per `Added` event, recorded at `BoolWindow::push` time (the
residual is already computed there; today it is used for the rotation and
dropped):

```text
BirthRecord {
  generation:    g            // = birth order = stratum index
  commit:        t:<sha1>     // head commit at the moment of birth
  original:      h:<sha1>     // the turn HLLSet that caused the event
  residual:      h:<sha1>     // R_g — the flag basis element (stable)
  pivot:         u32          // the new leading bit
  rotation_count: usize       // how many older slots re-pivoted
}
```

Properties:

- **Additive.** `BoolWindow::push` keeps its signature; a recording wrapper
  captures the table. No existing CID, test, or caller changes.
- **Rebuildable.** `StateCache::restore` already replays every committed turn
  through `ring.push`; recording during the same replay makes the table
  commit-aligned by construction, so it can never drift from the DAG.
- **O(t) storage.** `t = dimension ≤ #originals`; each `R_g` is already
  content-addressed at birth.
- **Deterministic per lineage.** The flag inherits the insertion-order and
  pivot-choice dependence of the ring (`docs/BOOLRING.md` scope limits). It
  is canonical *given* the commit sequence, not globally canonical.

The table is a two-way index: `commit → strata born there` and
`stratum → birth commit`.

---

## 3. Use case 1 — host LLM context: strata as the context unit

The existing context surfaces are the ring's **visibility window** (the last
`RING_CAPACITY = 64` originals, `ewm-app/src/state.rs`) and the Python
`BonsaiSession` modes `full` / `compact` (last-cap materialized memory vs the
**D-part novelty stream**, `trainer/bonsai_cli.py`). The strata are the
structural version of both, with provenance:

- a **stratum** is one new degree of freedom with its birth commit — the
  right granularity, because originals are mostly in-span redundancies while
  directions are exactly the structural novelties;
- `compact` (D-part displacement tokens) is the token-level shadow of "the
  newest stratum"; the birth table makes the same idea bit-level and
  commit-linked.

Selection policies become one-liners over the table:

```text
recent(K)         last K strata                 structural recency window
skeleton+recent   strata 0..s plus last K       stable background + novelty
budget(B)         newest first, then by |R_g| / rotation_mass, until B bits
delta_since(c)    strata born after commit c    "what changed since then"
```

Present the **stratum digest**, not bit vectors:

```text
(g, t:cid, |R_g|, pivot, rotation_count, provenance)
```

This is the interpretation (frame) the LLM should see, and it keeps the
prompt compact — notebook 20's compressed-context result, at the structural
level. The visibility window and the stratum window coexist: originals give
*data* recency, strata give *structural* recency.

---

## 4. Use case 2 — trajectory ambiguity: membership and signature

Content addressing already pins *a* trajectory (the `t:` chain). The harder
ambiguity is: given a **state** `X` (an HLLSet from some run/world/time),
which lineage does it belong to, and *where in its history* is it? The flag
answers both.

**Lineage membership.** `X` is consistent with lineage `L` up to commit `c_k`
iff

```text
residual of X against R_1..R_k = ∅.
```

**Generation signature.** When `X ∈ span_t`, its coordinates in the *flag
basis* `R_1..R_t` are unique; the support

```text
sig(X) = { g : coordinate_g ≠ 0 }
```

is the set of strata needed to reconstruct `X`. Same final bits reached
through a different history → different flag → different signature. This is
the disambiguation primitive.

**Off-trajectory / cross-world test.** When `X ∉ span_t`, the residual is the
part no stratum explains. Compare residuals against candidate lineages: the
lineage with the empty (or smallest) residual wins. A wrong lineage is a
wrong logic — the BOOLRING span test doing duty as the world-relative check
of `docs/EWM_RULES.md` §5.

**Soft signature.** `w_g = |X ∩ R_g| / |R_g|` per generation is always
defined (the soft key sees only the cover, `docs/BASIS_FRAMES.md` P1) and
gives a similarity profile over time even for states outside the span. Hard
signature for exact reconstruction; soft signature for "which eras does this
state resemble".

**Exploration index.** With the birth table plus the existing
`BasisHistory::project_into(i, X)` time travel (soft / hard / spill), an
ambiguity query is a walk:

```text
for each candidate commit c_k:
    project X into F_{k-1} and F_k
    report (spill before, strata born at c_k, spill after)
```

The spill collapses at the commit that first explains the state — the
ambiguity localizes to that commit.

Exactness note: all of this is GF(2)/set algebra over the site level, where
the sketch is exact (`docs/FOL_HLLSET.md` §3.3). The token-level alias gap
does not enter the membership or signature tests.

---

## 5. What it is not

- **Not a replacement for rule-layer stratification.** `docs/EWM_RULES.md`
  strata order anti-monotone literals (negation, downward thresholds); the
  birth strata order data by generation. Orthogonal dimensions. If both are
  used, evaluation is over the product `(logical stratum, birth generation)`;
  treating "born earlier" as "lower negation stratum" would be unsound.
- **Not a global canonical basis.** The flag is lineage-relative; merges in
  the commit DAG need a chosen topological order (the ingestion order the
  ring already uses on the first-parent chain).
- **Not a metric.** Coordinates still do not preserve `popcount(A Δ B)`
  distance unless the basis is disjoint (`docs/BOOLRING.md` scope limits).

---

## 6. Build plan

1. ✅ `ewm-boolring::BirthTable` — `Vec<BirthRecord>` + the recording wrapper
   `push_recorded` around `BoolWindow::push_with_birth` (`push` unchanged).
2. ✅ Wire into `StateCache::empty` and `StateCache::restore` (record during
   the existing replay) and into `ewm-app::run_turn` (birth linked to the
   turn's commit).
3. ✅ Query API: `stratum(g)`, `generation_of(&commit)`,
   `flag_coordinates(&X)`, `signature(&X)` (hard + soft), `membership(&X, k)`,
   `cover()`.
4. ⬜ `ewm-flux-host::sidecar` — add the commit↔generation link to
   `BasisHistory` so `project_into` can be driven by commit, not just by
   generation number.
5. ◐ Integration tests ported (`ewm-boolring/tests/birth_table.rs`,
   `ewm-app` birth/restore tests); ⬜ use the table in the host-LLM context
   proposals (Python `BonsaiSession` and/or the Rust side).

---

## 7. Trajectory optimization as constraint programming

The three items of §5 are the starting point of an optimization model over
the trajectory `H(t) = (S(t), H(t-1), D, R, N)` (`ewm-app`'s commit record).
The trajectory is the decision object; the ring/lattice/commit rules are the
constraints; `H(t)` is the state descriptor whose cost is minimized. The
prototype is `notebooks/28_trajectory_optim.ipynb`.

### 7.1 Given, decided, derived

```text
given   : observations O_1..O_n (HLLSets), budgets B_bits, B_strata, B_dim
decide  : per turn — commit? retain? evict? merge order π? which strata kept
derived : S_t, H_{t-1}, D_t, R_t, N_t, rotation_mass_t   (functions of the decisions)
```

### 7.2 Constraints

```text
C1 definitional   D = H_{t-1} \ S,  R = H_{t-1} ∩ S,  N = S \ H_{t-1}
                  (D ∪ R = H_{t-1},  N ∪ R = S,  D, R, N pairwise disjoint)
C2 monotone span  span_{t-1} ⊆ span_t            (the flag: 0 ⊂ span_1 ⊂ …)
C3 immutability   committed h:/t: nodes never change          (IICA)
C4 budget         |S_t| ≤ B_bits,  dim ≤ B_dim,  strata kept ≤ B_strata
C5 stratification (logical σ, birth g) is a valid topological order — the
                  product of the two orders of §5
C6 lineage        every basis-dependent quantity is taken under one fixed
                  linearization π (the merge order decision)
C7 metric         costs only via popcount(Δ / ∩ / \) and BSS; never coordinate L2
C8 commit rule    materialize iff  N_t ≠ ∅  or  basis changed
C9 linking        spill(X) ⊆ residual(X)         (bit-novelty ⊆ linear novelty)
```

### 7.3 Objective

```text
min  Σ_t  α|D_t| + β|N_t| + γ·rotation_mass_t + δ|S_t|
```

or lexicographic / Pareto over the same terms. The two extremes are
`minimize |D|` (forget nothing — pure accumulation) and `minimize |S|`
(keep nothing — no context); the optimizer is the dial between them, subject
to C1–C9.

### 7.4 Structure and solvability

- C1–C4, C8 are 0-1 linear (pseudo-Boolean) once states are bit-vectors;
  the span part of C2/C6/C9 is GF(2)-linear (XOR) and solvable by Gaussian
  elimination inside a SAT/SMT solver. The offline model is the **exact
  oracle**.
- The cover/coverage subproblem is monotone submodular: the greedy selector
  is `(1 − 1/e)`-optimal, and the flag turns prefix policies into a
  one-variable dynamic program. The **online policy is greedy**; the
  prototype measures it against the oracle.

### 7.5 What the prototype shows

`notebooks/28_trajectory_optim.ipynb`: the greedy oracle over strata vs
originals with the `(1 − 1/e)` check on a small exhaustive instance; four
online policies (`keep-all`, `window-W`, `strata-window-K`,
`greedy-budget`) validated against C1/C4/C8; the cost table
`α·D + β·N + γ·rotation + δ·|S|` with the winning policy and the honest
trade-off (bounded context costs forgetting); and the C9 linking law
`spill ⊆ residual`.

---

## 8. The wishful forecast loop

The constraint model of §7 is a **policy, not a transition**. It never
mutates a node, never rewrites a commit, never changes a bit — IICA holds
throughout. It does two things: **reset priority** (the weights and budgets
choose which trajectory is preferred) and **make a choice** (keep / evict /
commit / order, among nodes that already exist — discovered, not created).
It sits beside the state machine, not inside it.

Treat the policy as a **wishful forecast** — a receding-horizon optimistic
controller with a surprise gate:

```text
Ŝ_{t+1..t+H} = plan(O_1..O_t, Ŝ_t, priorities αβγδ)      the wish
act as if Ŝ_{t+1} is achieved                              the optimist move
O_{t+1} arrives                                            reality
divergence = |Ŝ_{t+1} Δ O_{t+1}| = |D| + |N|
    D = Ŝ_{t+1} \ O_{t+1}    wishful surplus (kept, not confirmed)
    N = O_{t+1} \ Ŝ_{t+1}    reality beyond the wish (missed)
if divergence is statistically absurd:  re-anchor  Ŝ ← O   structural event
else:                                   recur (the wish becomes the baseline)
```

Mapping to existing machinery:

| wishful concept | ewm-sm machinery |
| - | - |
| wish `Ŝ` | the planned context state (e.g. the §7 greedy-budget cover) |
| `D`, `N` | the Noether decomposition, wish vs reality |
| divergence | `\|D\| + \|N\|` — Hamming distance wish↔reality |
| statistically absurd | the innovation gate: EWMA surprise (predictor portfolio), `2σ` residual spikes (BOOLRING), the soft-key jump detector (BASIS_FRAMES) |
| re-anchor | the structural commit / basis-change event — the wish is rejected, reality is re-admitted |
| fixed point | CID stability — when the wish is a good model, the context stops changing |

**The null model.** "Absurd" is only meaningful against the expected
divergence distribution under the wish; the gate is the existing EWMA + `kσ`
over a trailing window. The loop is: plan, act, measure, gate, re-anchor —
the CP plan is the prediction, the surprise machinery is the absurdity test.

**Optimist vs realist.** The realist recurs with reality as the baseline; the
optimist recurs with the wish. Optimism compounds bias — a feature while
reality agrees, a failure mode when it drifts. The gate is the guard (the
fire budget is the same kind of guard one level down).

`notebooks/29_wishful_forecast.ipynb` demonstrates the loop on the synthetic
clip: the optimist (greedy-budget context) vs the realist (yesterday's
frame), the divergence series with the `kσ` gate firing at scene cuts, the
re-anchor step that restores forecast quality — and the no-re-anchor control
that stays stuck in the old scene.

## See also

- `docs/BOOLRING.md` — the monotone basis, extension + rotation, scope limits.
- `docs/BASIS_FRAMES.md` — frames, generation stamps, time travel, the
  basis-change commit condition.
- `docs/EWM_RULES.md` — logical stratification and world-relative programs.
- `notebooks/27_basis_strata.ipynb` — the prototype: flag laws, context
  policies, membership/signature, spill-collapse time travel.
- `notebooks/28_trajectory_optim.ipynb` — the constraint-programming
  prototype: greedy oracle vs exhaustive, four online policies under
  C1/C4/C8, the cost table, and the spill ⊆ residual law.
- `notebooks/29_wishful_forecast.ipynb` — the wishful loop: optimist vs
  realist, divergence `= |D| + |N|`, the `kσ` gate at scene cuts, and the
  re-anchor step vs the stuck no-re-anchor control.
