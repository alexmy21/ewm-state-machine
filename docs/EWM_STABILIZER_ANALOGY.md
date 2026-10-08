# EWM state trajectory ↔ stabilizer formalism

**Status:** analysis — a precise, deliberately bounded analogy. Not a
foundation change and not a claim that the EWM state is a quantum state. The
load-bearing correspondences are exact GF(2) facts; the rest are labelled as
analogy or excluded. Illustrated in
[`notebooks/24_stabilizer_analogy.ipynb`](../notebooks/24_stabilizer_analogy.ipynb).

## The one-sentence version

The EWM ring is the **GF(2) / stabilizer (Clifford–Gaussian) shadow** of a
state trajectory: the same linear algebra that underlies stabilizer quantum
computing, used classically — rank is entropy, Gaussian elimination is the
simulation core, a re-basis is a Clifford frame change — while amplitudes,
phases, interference and the orthomodular projection lattice are absent.

## Dictionary

| EWM ring + lattice | Stabilizer / Gaussian formalism |
| - | - |
| HLLSet bit vector | Pauli operator as a binary symplectic vector in `GF(2)^{2n}` (X-block, Z-block) |
| `Δ` (XOR) | group multiplication mod 2 |
| span of originals | binary linear code / exponent of a stabilizer group |
| `dimension()` | code rank — the quantity that gives stabilizer-state entanglement entropy |
| row-echelon basis + pivots | generator matrix + Gaussian elimination |
| `change_of_basis`, pure re-basis | invertible GF(2) transform ⇒ **Clifford** frame change (Gottesman–Knill); symplectic once a form is fixed |
| span extension | adding a stabilizer generator / encoding a logical qubit (isometry, not invertible) |
| windowed eviction | discarding a generator — irreversible |
| monotone basis `S₁ ⊆ S₂ ⊆ …` | a **filtration (flag) of codes** — the shape of a measurement record |
| `cover` | classical support: the bit plane the code touches |
| `residual ≠ 0` | off-code component — analogue of non-stabilizerness ("magic") |
| meet table | subgroup-intersection data (a group closes under `·`, not under `∩`) |

## Exact correspondences (verified in notebook 24)

1. **Rank = entropy.** For a stabilizer state the entropy of a subsystem is the
   GF(2) rank of a submatrix of the symplectic/adjacency matrix; for a graph
   state `Γ`, `S(A) = rank(Γ[A, B])`. The notebook checks this against a
   numeric partial trace on a 5-qubit cycle: `0, 1, 2, 2, 1, 0` bits — exact
   agreement with the GF(2) ranks. `dimension()` is exactly such a rank.
2. **Change of basis = Clifford / Gaussian operation.** A pure re-basis (same
   span, different generators) is an invertible GF(2) transform; the notebook
   verifies the coordinate round-trip `a → a·M → a`. Extension raises the rank
   by one (an isometry); eviction drops it. `rotation_count` is the number of
   generators re-expressed — a gate/update-count analogue.
3. **Overlap is a subgroup meet.** With `ρ_G = (1/|G|) Σ_{g∈G} g` and
   `|G| = |H| = 2ⁿ`,

   ```text
   |⟨ψ_G|φ_H⟩|² = tr(ρ_G ρ_H) = (1/(|G||H|)) Σ_{g,h} tr(gh)
                = (|G∩H| − |G∩(−H)|) / 2ⁿ,
   ```

   because `tr(gh) ∈ {0, ±2ⁿ}`. The meet is a **subgroup/subspace
   intersection** (Gaussian elimination), *not* the bitwise `∩` of HLLSet
   supports. The closed form `2^{d−n}` (`d = dim(G_vec ∩ H_vec)`) holds only
   when the groups are phase-aligned; the notebook shows a graph-state pair
   where it fails (`d = 1` predicts `1/4`, the true overlap is `0`).
4. **The trajectory is a code filtration.** `S₁ ⊆ S₂ ⊆ …` is a flag of binary
   codes; adding a generator is a stabilizer measurement/encoding step. This is
   the order-theoretic shape of a stabilizer measurement record.

## Analogy only (do not overclaim)

- `residual ≠ 0` behaves *like* non-stabilizerness ("magic"): a set is
  expressible as an XOR of generators iff the residual is empty. But the
  residual is a **non-canonical coset representative** (see
  `docs/BOOLRING.md`), so it is not a magic monotone.
- `cover` is a classical support, not a reduced density matrix; `BSS` is a
  normalized set overlap, not a Hilbert–Schmidt inner product.

## Where it breaks

- **Boolean vs. orthomodular.** The HLLSet lattice is distributive; the
  lattice of Hilbert-space projections is orthomodular and non-distributive.
  In `C²`, three distinct rank-1 projections `P, Q, R` give
  `P ∧ (Q ∨ R) = P` but `(P ∧ Q) ∨ (P ∧ R) = 0`. So this is **not** quantum
  logic — it is the classical shadow of the GF(2) algebra.
- **No amplitudes, phases, interference, Born rule.** A plain GF(2) span is a
  binary **code**, not a stabilizer group: the Pauli central charge
  (`iᵏ`) and the symplectic commutation form are extra structure. "In span"
  means code membership, not "simultaneously stabilizes a state".
- **Gauge freedom.** Mapping bit positions to Paulis is a free choice; the
  analogy is only defined once that encoding is fixed.

## Making it literal (a concrete construction)

Fix a symplectic form on the 32768-bit plane and restrict the ring to
**isotropic/Lagrangian** subspaces (pairwise-commuting generators, each
generator with a chosen phase). Then the span is a genuine stabilizer code, and
the exact statements above — rank = entropy, change-of-basis = Clifford,
overlap = subgroup meet — become theorems about actual stabilizer states rather
than structural parallels. That is a testable experiment, not a metaphor.

## Scope

This document records a *reading* of the ring in the language of the
stabilizer formalism. It changes no behavior. The EWM state remains a
classical, content-addressed HLLSet trajectory; the value of the analogy is
that it imports the right toolkit — rank/entropy measures, Gaussian-elimination
frame changes, and the subgroup-meet view of overlap — without importing
claims the model cannot support.
