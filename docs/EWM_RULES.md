# EWM rule layer: a Datalog-shaped DSL for the state machine

**Status:** design note. No code changes. It specifies a rule layer *for*
`ewm-sm`, on top of the HLLSet lattice and the existing `ewm-ops` graph, and it
is deliberately **Datalog-shaped rather than Prolog-shaped**. It builds on
`docs/FOL_HLLSET.md` (Boolean propositional + monadic quantification, local
universes) and `docs/BOOLRING.md` (the ring, D/R/N, the meet obstruction).

**Verdict in one paragraph.** A rule layer is a good fit, not a novelty, because
three pieces already exist: the lattice is complete and the positive rule core is
monotone (so the least fixpoint is available by Tarski; §2.3 for the
exceptions), semi-naive evaluation *is* the
machine's `N = S(t) \ H(t-1)` delta, and content addressing gives both the
fixpoint test and the derivation record. What must **not** be borrowed from
Prolog is SLD resolution over unification: the sketch loses tuple identity, so
joins are approximate unless routed through the LUT/`h:` fibre. The exact core
is *monadic* — unary predicates, quantifiers over a coordinate, stratified
negation, thresholds — with joins as a separate, clearly-marked identity path.

---

## 1. Why a fixpoint layer is native here

### 1.1 Tarski gives the semantics for free

Let `L` be a local universe (`docs/FOL_HLLSET.md` §4) and `B_L = [∅, L]` its
Boolean algebra. An **interpretation** assigns to each predicate symbol an
element of `B_L`; the space of interpretations is a product of complete Boolean
algebras, hence a complete lattice. For a **positive** program, the immediate
consequence operator

```text
T_P(I)(p) = ⋃ { ⟦body⟧_I : (p :- body) ∈ P }
```

is monotone (`∪`, `∩`, `∃_c`, `∀_c` are all monotone), so by
Knaster–Tarski it has a least fixpoint, reached by iterating from `⊥`. That
least fixpoint **is** the least model — the standard Datalog semantics. For a
monotone program the fixpoint is unique and independent of rule order.

### 1.2 Semi-naive iteration *is* the D/R/N pipeline

Write `S_k` for the state after round `k` and `Δ_k = T_P(S_{k-1}) \ S_k` for
what the round added. Then

```text
S_k      = S_{k-1} ∪ Δ_k            # the join — the EWM accumulation
Δ_{k+1}  = T_P(S_k) \ S_k           # the delta — the machine's N
```

This is the EWM turn verbatim: `S(t)` is the accumulating state, `N` is `Δ`,
and a commit happens when the delta is non-empty. Semi-naive evaluation — only
feed `Δ` to the rules that can consume it — is therefore an optimization the
representation already suggests, not an add-on. The ring's residual adds a
second, cheaper filter: a round whose delta lies **in the span** of the base
cannot change any committed direction, so `residual(Δ) = ∅` short-circuits the
commit decision (a structural no-op). Note this is a valid *commit filter*, not
monotone reasoning: span membership is GF(2)-linear — see §2.3.

### 1.3 Content addressing gives the stop test *and* the explanation

- **Stop test.** `Δ = ∅` is exactly `cid(S_k) = cid(S_{k-1})`. IICA makes the
  comparison exact, cheap, and safe under replay.
- **Explanation.** A derived node is a **view**: a recipe plus the SHA-1 CIDs
  of its sources. A derivation tree is a provenance chain, so an `explain`
  predicate is the existing view model read back — no separate proof object is
  needed. "Which rules fired, on which sets" is recorded by construction.

---

## 2. The language

### 2.1 Facts, predicates, universes

- A **fact** is a committed HLLSet (`h:` node) inside a local universe `L`.
- A **predicate** is a *unary* relation: an element of `B_L`. There is no
  separate ground-atom store — the extension of a predicate *is* its HLLSet.
- Every program is evaluated **relative to `L`**. There is no global negation;
  `not p` means `L \ ⟦p⟧`. A program is shared content (`p:` node) while a
  world supplies the `L` it is run in.

### 2.2 Rules

```text
head :- literal, ..., literal.
```

with literals of four kinds:

| literal | HLLSet meaning | monotone? |
| - | - | - |
| `p` (predicate) | `⟦p⟧` | yes |
| `some c (p)` / `every c (p)` | `∃_c ⟦p⟧` / `∀_c ⟦p⟧` (fibre over coordinate `c`) | yes |
| `not p` | `L \ ⟦p⟧` | **no** — stratify |
| `bss(p) > t` / `bss(p) < t` | degree comparison | `>` yes, `<` **no** — stratify |
| `residual(p) > 0`, `in_span(p)` | span membership — a **linear** (parity) condition | **no** — neither monotone nor anti-monotone |

### 2.3 Monotonicity is the design rule

`∪`, `∩`, `∃_c` and `∀_c` are monotone. Three families leave monotonicity, and
they are **not** the same kind of departure:

- the local complement `L \ ·` and **downward thresholds** (`bss < t`) are
  **anti-monotone** — more input can make them false;
- **span membership is linear, not ordered.** The reduction is GF(2)-linear
  (`residual(X Δ Y) = residual(X) Δ residual(Y)`), and monotonicity genuinely
  fails: `00001 ⊆ 00011` yet `residual(00001) = 00100 ⊄ residual(00011) = 00000`.
  So `residual(p) > 0` / `in_span(p)` is a **parity** condition; it belongs with
  `Δ` on the constraint side, not in a monotone body;
- `Δ` (XOR) itself is linear for the same reason.

A program is **stratified** when the dependency graph — edges for `not` and for
downward thresholds — has no cycle through an anti-monotone edge; strata are
then evaluated in order, each as a positive program. Linear literals do not fit
the monotone schema at all: they are either **constraints** handled beside the
fixpoint (M4) or rejected in the core. Non-stratified programs are **rejected at
compile time** rather than allowed to oscillate.

This is the single most important discipline in the design: *stratification
orders the negation-like literals (complement, downward thresholds), and keeps
the linear literals (span membership, XOR) separate.*

### 2.4 What the surface looks like

```text
% facts are value nodes; a "variable" is a predicate or a coordinate
frame(X).                                   % X ranges over committed sets

stable(X)  :- frame(X), bss(X, L) < 0.10.   % downward threshold -> neg stratum
novel(X)   :- frame(X), residual(X) > 0.
cut(X)     :- frame(X), not stable(X), novel(X).
quiet      :- every tz (frame).             % ∀ over one coordinate

explain(cut(@frame_11)).                     % the derivation chain, as a view
```

`cut` is a derived predicate; `quiet` is a **sentence** and therefore evaluates
to `L` or `∅` (two-valued *within* the world), while `cut` keeps a truth
**degree** `|⟦cut⟧|/|L|` — the BSS readout. A rule set compiles to an `ewm-ops`
program, so a rule's identity is the SHA-1 of its resolved canonical source.

---

## 3. Execution

```text
compile:  parse -> check stratification -> assign strata -> resolve to p: CIDs
run:      for each stratum, in order:
              S <- facts ∪ (already-computed earlier strata)
              repeat
                  D <- eval_stratum(S) \ S        # semi-naive: feed the delta
                  if cid(S ∪ D) == cid(S): break   # fixpoint by content address
                  S <- S ∪ D
          commit S as views with provenance (rule CID + source CIDs)
```

Properties:

- **Termination.** No function symbols → finite Herbrand base (the sites of
  `L`) → each stratum reaches its least fixpoint in finitely many rounds.
- **Determinism.** The least fixpoint is order-independent; the *fire order*
  affects only intermediate states, never the result (the operator is
  monotone).
- **Idempotence / replay.** Re-running a stratum over a fixpoint adds nothing;
  an interrupted run restarts from the last commit and reproduces the same
  `cid`. Recovery is a pop, as everywhere else in the machine.
- **Auditability.** Every derived node is a generation-stamped view; `explain`
  walks provenance.

---

## 4. What is exact, and what needs the identity layer

| fragment | status |
| - | - |
| unary predicates; `∪`, `∩`, local `¬`, `∃_c`, `∀_c` | **exact** on `B_L` |
| thresholds over `bss`/`residual` | exact as filters; degrees are exact |
| stratified negation | exact (strata are positive programs) |
| **join of two relations on a shared coordinate** | **approximate** on sketches — needs the LUT/`h:` fibre. This is *tuple identity*, a structural gap, distinct from the collision/alias gap, which is exact at the site level |
| equality / substitution (`x = y`) | not available (see `FOL_HLLSET.md` §8) |
| parity (XOR) constraints, span membership, `residual` | not Horn and not ordered (linear); the ring's affine fragment — keep as constraints (§2.3) |

The join row is the whole research question, and the design keeps it *out of
the core*: a rule set written without repeated variables is exact, and a rule
that needs a join must opt into the identity path (materialize the key through
the LUT, join at token level, re-ingest). The approximation rate of the sketch
path is measurable — it is the collision rate of the morphism — so the choice is
a cost/accuracy dial, not a hidden bug.

---

## 5. Semantics per world (why this composes with the companion)

Rules are evaluated relative to a local universe `L_w`, and derived sets are
elements of `B_{L_w}`. For `L' ⊆ L`, restriction is a Boolean homomorphism
(`FOL_HLLSET.md` §4.2), so a derivation restricts to a derivation; local truths
that agree on overlaps **glue**. The rule layer is therefore a **sheaf of least
fixpoint models** over the poset of local universes:

- a program is shared (`p:` CID), a run is world-relative;
- cross-world comparison is restriction to the overlap;
- there is no global `not`, and no global fixpoint — consistent with virtual
  isolation, and the logical face of the bridge.

---

## 6. Cost and optimization

- **Per round:** `O(Σ_rules |operands|)` bit operations; semi-naive bounds the
  work by `|Δ|` per round, not `|S|`.
- **Delta filters:** `residual(Δ) = ∅` → no structural commit; an in-span delta
  cannot create a new base direction.
- **View cache:** cache `(rule CID, input CID, basis generation) → result CID`;
  a read in a new generation re-derives only what is queried, never the whole
  history.
- **Threshold pruning:** downward thresholds cut the candidate set early;
  `bss` is computed from popcounts, so it is a few bit operations.

---

## 7. Risks, and how the design contains them

| risk | containment |
| - | - |
| joins lose tuple identity (structural) | keep out of the exact core; opt-in LUT path |
| token-identity queries pick up alias false positives from collisions | site semantics is exact; the alias gap equals the collision rate, so measure it |
| non-stratified negation oscillates | static stratification check; reject at compile time |
| downward thresholds are secretly negation | classified as anti-monotone; forced into a stratum |
| derived predicates grow the base without bound | watch `dimension` and `residual`; the ring's `RingStats` are the meters |
| semantics depend on the universe being right | the universe registry discipline: every run names its `L`; a wrong `L` is a wrong logic |
| run diverges from the intended fixpoint under a fire budget | the budget is a round cap; the fixpoint is unique, so a capped run is a prefix of it |

---

## 8. Milestones

1. **M1 — monadic core.** Stratified Datalog: unary predicates, `∃_c`/`∀_c`,
   thresholds, `explain`. No joins. Tested for convergence-by-CID, idempotent
   replay, and correct stratification.
2. **M2 — the identity path.** Joins through the LUT/`h:` fibre, with the
   sketch-path false-positive rate measured against it.
3. **M3 — world-relative programs.** Programs pinned by CID, run per `L_w`,
   with the gluing check on overlaps.
4. **M4 — parity constraints.** The ring's affine fragment as constraints
   alongside Horn rules (XOR is not Horn; it belongs on the constraint side).

## 9. A testable checklist

1. **Convergence.** `cid(S_k) = cid(S_{k-1})` at the reported fixpoint; the
   result is independent of rule order.
2. **Idempotence.** Re-running a stratum over its own fixpoint adds nothing.
3. **Stratification.** Programs with a negative cycle are rejected; the
   stratum order is a valid topological order of the anti-monotone edges.
4. **Threshold direction.** `bss > t` does not force a stratum; `bss < t` does.
5. **Explain.** The `explain` chain resolves to the actual source `h:` CIDs and
   rule `p:` CID.
6. **Restriction commutes with derivation.** A derivation in `L`, restricted to
   `L' ⊆ L`, equals the derivation of the restricted program in `L'`.
7. **Approximation.** With joins enabled on sketches, the false-positive rate
   tracks the collision rate; with the LUT path, it is zero.
8. **Linear literal.** A body using `residual(p) > 0` is rejected by the core
   (or routed to constraints); the reduction is confirmed GF(2)-linear, with the
   subset counterexample reproduced.

## See also

- `docs/FOL_HLLSET.md` — Boolean/monadic logic, local universes, transport.
- `docs/BOOLRING.md` — the ring, `D/R/N`, the meet obstruction, the tools.
- `docs/SEPARATION.md` — the fibre is global; why the LUT can supply identity.
- `crates/ewm-ops/src/dsl.rs` — the stack words: the connectives and the
  counted quantifiers (`exists_c` / `forall_c`) over `tz`/`reg`.
- `REFS/EWM/companion.tex` — worlds, virtual isolation, the bridge.
