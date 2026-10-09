# First-order logic and HLLSet Algebra

**Status:** analysis note. Scope is deliberately limited to the
**propositional fragment** and **monadic quantification over one coordinate**
(FOL with one bound variable and the rest free). The relational layer — n-ary
relations, joins, equality/substitution — is *out of scope* and is flagged as
the boundary in §8.

**Headline.** `HLLSet` algebra is a concrete Boolean algebra, so by
Lindenbaum–Tarski it *is* propositional logic; monadic quantification is the
lattice operation of **cylindrification** (∃ = union over a fibre, ∀ = its
dual), which needs no new representation. The `ewm-ops` DSL already carries the
propositional connectives (`union`, `inter`, `diff`, `symdiff`), so the missing
piece for the monadic fragment is a single projection word. What is *not*
available without leaving the representation is relational composition, and the
ring's linear fragment is parity (XOR) logic, not conjunction.

---

## 1. Why the propositional identity is exact

The **HLLSet lattice** is `B = (2^Plane, ⊆, ∪, ∩, Δ)` with relative complement
`¬X = U \ X`. It is a Boolean algebra — in fact a complete, atomic one.

The **Lindenbaum–Tarski** construction says the formulas of propositional logic,
quotiented by provable equivalence, form a Boolean algebra; a model is a Boolean
homomorphism into the two-element algebra. Because `B` is itself a concrete
Boolean algebra of truth sets, the two coincide:

> a propositional formula over atom predicates `a ↦ H(a) ⊆ Plane` evaluates to
> the HLLSet `⟦φ⟧`, and `⟦·⟧` is a Boolean homomorphism.

So HLLSet algebra is not *like* propositional logic; it is an instance of it.
The design consequence is that the whole propositional layer is decidable and
canonical: evaluation is a handful of bit operations, and equivalence is a
single bitmap comparison.

---

## 2. Setup

| Symbol | Meaning | In the project |
| - | - | - |
| `Plane` | the bit plane, `{0 … M·b−1}` | `TOTAL_BITS = 32768` |
| `L ⊆ Plane` | **local universe** (the primitive notion) | a user's collected union, or a world's `image(bit_w)` |
| `U ⊆ Plane` | **global** universe, `U = image(bit)` | union of *all* ingested collections (`h:` nodes) |
| `H(a)` | the HLLSet interpreting predicate `a` | a value node |
| `⟦φ⟧ ⊆ L` | the truth set of `φ` in the local universe | the result node |
| `bit(t)` | token → site | `BitAddress::of_token` |
| `μ(A)` | token collection → HLLSet | `ingest` |

**Use a universe, not the plane, for logic.** A site is a *witness* of a token
only if some token maps to it; quantifier laws, negation and validity are all
relative to a universe. The global case is `U = image(bit)`; the primitive case
is a **local universe** `L` — the union of the HLLSets a reader actually holds
(§4). Negation is `¬_L X = L \ X`:

```text
neg   φ        ->  @universe diff ⟦φ⟧      # the reader's L, not the plane
```

When one global universe is intended, take `L = U` and the two coincide.

Two coordinate functions come free with the soldered geometry and give the
fibres for §5: `reg(p) = p div b` and `tz(p) = p mod b`, so
`Plane ≅ [0,M) × [0,b)` (`M = 1024`, `b = 32`). The `conv(n, dim)` channel model
supplies the same product structure for n-gram grids.

---

## 3. The propositional fragment *is* HLLSet algebra

### 3.1 Interpretation and connectives

Define `⟦a⟧ = H(a)` and extend structurally:

| Logic | HLLSet | `ewm-ops` word |
| - | - | - |
| `φ ∧ ψ` | `⟦φ⟧ ∩ ⟦ψ⟧` | `inter` |
| `φ ∨ ψ` | `⟦φ⟧ ∪ ⟦ψ⟧` | `union` |
| `¬φ` | `U \ ⟦φ⟧` | `@universe diff` |
| `φ → ψ` | `(U \ ⟦φ⟧) ∪ ⟦ψ⟧` | composition |
| `φ ⊕ ψ` (parity, *not* a Boolean primitive) | `⟦φ⟧ Δ ⟦ψ⟧` | `symdiff` |

**Homomorphism lemma.** `⟦·⟧` preserves `⊤ = U`, `⊥ = ∅`, and `∪, ∩, ¬`, hence
all Boolean laws; it is the unique homomorphism of the free Boolean algebra on
the atom predicates that extends `a ↦ H(a)`.

### 3.2 Truth, consequence, equivalence

```text
φ valid          ⟺  ⟦φ⟧ = U
φ unsatisfiable  ⟺  ⟦φ⟧ = ∅
φ ⊨ ψ            ⟺  ⟦φ⟧ ⊆ ⟦ψ⟧
φ ≡ ψ            ⟺  ⟦φ⟧ = ⟦ψ⟧
```

Each is a constant-time bitmap test after evaluation. This is a **decision
procedure**: the interpreted propositional theory is finite, so the bitmap *is*
a normal form.

**Canonical form, precisely.** The bitmap is canonical for the *interpreted*
formula. The *free* algebra on `k` atoms is the Boolean algebra of the `2^k`
cells of their Venn diagram, and it maps into `B` injectively exactly when the
atom family is in general position (every Venn cell non-empty); otherwise
distinct free-algebra elements can have the same truth set. Content addressing
picks this up automatically: equivalent programs share an `h:` key, and the
`p:` key is the recipe.

### 3.3 Three levels of semantics — the honest picture

The word "exact" depends on which question we ask. All three hold
simultaneously and do not contradict each other:

1. **Satisfaction (site level): exact.** For a token `t` and formula `φ`,
   `t ⊨ φ  ⟺  bit(t) ∈ ⟦φ⟧`. So the HLLSet expression is a faithful decision
   procedure for token satisfaction, with no approximation. (Induction on `φ`;
   the atom case is the definition of `ingest`.)
2. **The algebra's denotation is the *site* semantics, and it is exact.** Let
   `A^sat = bit⁻¹(μ(A))` be the **saturation** of `A` (all tokens sharing a site
   with a token of `A`). On saturated collections — equivalently, on the site
   sets that an HLLSet *is* — `μ` is a **Boolean isomorphism** onto `P(U)`:
   `μ(A ∩ B) = μ(A) ∩ μ(B)` and `μ(Tok \ A) = U \ μ(A)`. So `∧`, `∨`, `¬` are
   exact *at the level the algebra denotes*; there is no approximation there.
3. **Token identity is finer than the representation — that is the only gap.**
   For an unrestricted token collection `μ` still preserves `∪` exactly, but the
   meet can fail in one direction:
   `μ(A ∩ B) ⊆ μ(A) ∩ μ(B) = μ(A^sat ∩ B^sat)`. The meet answers the *site*
   question exactly and the *token-identity* question **soundly but
   incompletely**: there are **no false negatives** (a negative answer is always
   true), and false positives occur exactly when two different tokens collide on
   a site. Collisions identify tokens *by design* — the bit vector forgets
   provenance. We call this the **alias gap**; it is not an error of the algebra.
4. **Inverse (materialization): many-to-one.** Recovery is the inverse problem —
   exact on builders, probabilistic on extras. Token-level satisfiability is
   `⟦φ⟧ ∩ U ≠ ∅`; the site-level test `⟦φ⟧ ≠ ∅` is a relaxation.

**Remark (image and preimage).** For any `bit : Tok → Plane`, the *image* map on
subsets preserves `∅` and `∪` but not `∩` (that needs injectivity); the
*preimage* map preserves `∅, ∪, ∩, ¬` and is a Boolean homomorphism, right
adjoint to the image: `μ(A) ⊆ X ⟺ A ⊆ μ⁻¹(X)`. Writ large: **writes (ingest)
are the lossy left adjoint, reads (the fibre) the exact right adjoint**, and the
unit `A ⊆ A^sat` is the alias gap — the only place the two semantics differ.

The soundness that matters is unchanged: `⟦φ⟧ = ∅` implies no token satisfies
`φ`, and `⟦φ⟧ = U` implies every token does.

### 3.4 The block reading: a morphism is a partition

The same fact read from the address side. A morphism `bit : Tok → Plane` splits
the token space into its **fibres**

```text
Tok = ⊔_{s ∈ U} [s],        [s] = bit⁻¹(s),        U = image(bit).
```

The cells are mutually exclusive, and each **bit address points to exactly one
cell**. An HLLSet is a set of addresses, hence a **union of cells**. Write
`sat(A) = bit⁻¹(μ(A))` for the cell-closure of `A`: the unions of cells are
exactly the fixed points of `sat`, and they form a **Boolean subalgebra** of
`P(Tok)` on which `μ` restricts to a Boolean isomorphism

```text
{ A : sat(A) = A }  ≅  P(U),        μ(A ∩ B) = μ(A) ∩ μ(B).
```

That is *why* the meet is exact when the operands are HLLSets: `μ(A) ∩ μ(B)` and
`μ(A ∩ B)` name the **same set of blocks**, because both are read through the
same partition.

One caution, because it is the crux: the closure `sat` is a **join-homomorphism
only** — it is *not* a Boolean quotient. `sat(A ∩ B) ⊆ sat(A) ∩ sat(B)`, and the
inclusion can be strict: with `bit(t) = t mod 6`, `A = {0}` and `A' = {0,12}`
have the same closure, `B = {0,7}` and `B' = {6,7}` have the same closure, yet
`sat(A∩B) = [0]` while `sat(A'∩B') = ∅`. Block-unions are a *subalgebra of fixed
points*, not a quotient, and that one-directional gap is exactly the alias gap of
§3.3 — a distinction *inside* a cell.

**Corollary (what a bit vector determines).** `μ(A) = μ(A')  ⟺  sat(A) = sat(A')`:
an HLLSet determines the **saturation** of its collection and nothing finer. So
every expression of the algebra factors through the kernel of `μ`, and two token
collections that meet the same cells are indistinguishable by `∧`, `∨`, `\`,
`¬`, the quantifiers, or the ring's `Δ`. *Token identity — which token flipped a
bit — is exactly what the representation discards*, and no operation on bits can
recover it. Multiplicity goes with identity: the vector records which cells are
nonempty, not how many tokens met them — the same fact as the idempotence of
`μ`, since re-ingesting a token whose cell is already marked changes nothing.
Recovering identity requires the fibre (the LUT / `h:` layer); it is never in the
bits. That discarded distinction is the space in which the alias gap of §3.3
lives, and it is why joins (which need tuple identity) must leave the sketch and
go through the LUT.

Two consequences:

- **The universe is the block index set.** `U = image(bit)` is the set of
  non-empty cells, and `P(U)` — not `P(Plane)` — is the algebra a morphism can
  reach. Negation is `U \ X` for the same reason (§2, §4).
- **A morphism is a world.** Different hash configurations give different
  partitions of the same token space, hence different fixpoint algebras `P(U_w)`;
  the worlds of the companion note are exactly these partitions, and the local
  universes of §4 are their index sets.

So the two readings are dual and complementary: from the token side an HLLSet is
the **saturation** of a collection (a union of cells); from the address side it
is a **set of cells**. The morphism *is* the partition, and its kernel — the
part of token identity the representation does not keep — is the inside of a
cell.

### 3.5 Triangulation: recovering what the kernel discards

One morphism is one partition, and its kernel is what is lost. Several
*independent* representations cut the cells further:

```text
f_i : Tok → A_i                    a seed, or an n-gram position
F = (f_1, …, f_k) : Tok → ∏_i A_i  the combined measurement
ker F = ⋂_i ker f_i                the meet of the partitions = their common refinement
F⁻¹(x_1, …, x_k) = ⋂_i f_i⁻¹(x_i)  materialization = intersection of fibres
```

No single address names a token; the **tuple** of addresses does, whenever the
meet is the identity relation on the token universe. The expected ambiguity
decays as the product of the per-cell fractions — roughly `m^{-k}` for `m` cells
per measurement — so it falls off *exponentially in the number of measurements*.
The project uses two flavours, one carried by the data and one by the hash:

- **n-grams** (`conv(n, dim)`): a token is measured by the n-grams it fills, so
  context supplies several addresses for one token — a refinement for free;
- **n-seeds** (the configurable hash seeds, i.e. the worlds): independent
  partitions of the same token space, each cheap, intersected and escalated
  lazily on collision.

Both live in the fibre, not in the bits (the Corollary above): the LUT holds the
cells, and materialization is the intersection of candidate sets across
measurements. The same construction at the node level is the companion note's
multi-hash retrieval — one digest almost settles a lookup, and a second and
third are consulted only on a collision.

Caveat: triangulation makes the *combined* address injective with high
probability, but it inherits the hash assumptions (uniformity, no adversarial
keys) and it is exact only for registered builders; for unseen tokens the tuple
can be empty (sound) or shared — a residual collision rate that shrinks with
`k`.

---

## 4. Local universes: logic on top of a universe

The universe `U` of §2 is the **global** case. The primitive notion is the
*local universe* — the early **local-Universe** of the project: the union of the
HLLSets a reader actually holds. The plane is only the outer bound.

### 4.1 Relativization

For any `L ∈ B`, the interval

```text
B_L = [∅, L] = { X : X ⊆ L },   unit L, zero ∅, complement ¬_L X = L \ X
```

is itself a Boolean algebra: the **relativization** of `B` to `L`. Its atoms are
the singletons of `L`. Every logical notion of §3 is then relative to `L`:

```text
φ valid in L   ⟺  ⟦φ⟧ = L
φ unsat in L   ⟺  ⟦φ⟧ = ∅
¬φ             =  L \ ⟦φ⟧            # local negation, never absolute
```

Two natural local universes, which generalize each other:

- **collection universe** — the union of a user's collected HLLSets,
  `L = ⋃ collected` (the original local-Universe);
- **world universe** — the sites a hash configuration can reach,
  `L_w = image(bit_w)` (a "world" of the companion note).

Both are elements of `B`, so both generate their own `B_L`. Because different
users hash differently, their worlds carry different `L_w`: one bit plane holds
a **family** of relativizations, each with its own unit and its own negation.

### 4.2 Transport between universes

For `L' ⊆ L` the **restriction**

```text
ρ_{L→L'}(X) = X ∩ L',        ρ : B_L → B_{L'}
```

is a Boolean homomorphism — it preserves `∅, ∪, ∩` and complement,
`ρ(L \ X) = L' \ ρ(X)` — so *logical laws survive restriction*. The inclusion
`ι : B_{L'} ↪ B_L` preserves `∅, ∪, ∩` (not complement, not top), and
`ρ ∘ ι = id`: `B_{L'}` is a retract of `B_L` in the lattice sense.

The assignment `L ↦ B_L` with these restrictions is a **presheaf of Boolean
algebras over the poset of local universes**, and it satisfies the sheaf
condition for union-covers: a cover `⋃ L_i = L` with compatible pieces
`X_i ∩ (L_i ∩ L_j) = X_j ∩ (L_i ∩ L_j)` determines and is determined by
`X = ⋃ X_i`. So **local truth values that agree on overlaps glue to a global
one.** This is the formal content of "the algebra is built on top of local
universes": the plane supplies the carrier, the local universes supply the
units, and transport between them is restriction.

### 4.3 What this changes, logically

- **Locally classical, globally a sheaf.** Inside one `L` everything is
  ordinary Boolean logic. There is no global `¬` and no global `⊨`: `φ` can be
  valid in `L_{w1}` and refutable in `L_{w2}` without contradiction, because the
  readings live in different relativizations and are compared only on
  `L_{w1} ∩ L_{w2}`.
- **Cross-world statements are restrictions.** Comparing or moving a statement
  between universes is restriction to the overlap; because restriction is a
  homomorphism, no classical law is lost on the way. This is the logical face
  of the companion's bridge (`materialize → transcribe → ingest`) and of
  virtual isolation.
- **Truth is a degree, not a bit.** A formula with free coordinates has
  `⟦φ⟧ ∈ B_L` — a **Boolean-valued** truth value in the Scott–Solovay–Vopěnka
  sense. Its measure `|⟦φ⟧| / |L| = BSS(⟦φ⟧, L)` *is* the project's coverage
  readout: **BSS is the degree of truth of a Boolean-valued formula, relative
  to its local universe.** A sentence collapses to a *union of join-classes* of
  the `L`-partitions (§5.2): it is two-valued exactly when the coordinates
  separate `L`, and otherwise it keeps intermediate values — the local universe
  itself decides how classical its logic is.
- **Quantifiers are relative too.** `∃_c^L` and `∀_c^L` are the fibre
  cylindrifications of `B_L`; the laws of §5 hold inside each relativization,
  with `L` in place of `U`.

Relative to the companion note's Grothendieck target: the hash-configuration
site `Ω` and the universe poset are two sites over the same representation; the
world family is the richer one and the local-universe presheaf is its
order-theoretic skeleton. We claim only that
relativization-plus-restriction is what makes "logic per universe" precise,
not a sheaf-theoretic equivalence.

## 5. Monadic quantification: `∃`/`∀` over one coordinate

Fix a **coordinate** `c : Plane → I` (take `c = tz`, or `c = reg`, or a
`conv`-grid axis). Its **fibres** are

```text
F_i = { p ∈ Plane : c(p) = i },        Plane = ⊔_i F_i .
```

Read `F_i` as "the assignments that agree on the bound variable". The two
cylindrifications are

```text
∃_c X  =  ⋃ { F_i : X ∩ F_i ≠ ∅ }          # saturation: "some value of the variable"
∀_c X  =  ⋃ { F_i : F_i ⊆ X }     =  ¬_U ∃_c (U \ X)   # dual: "every value"
```

Both land in the **fibre subalgebra** `Fix = { X : X = ∃_c X } = {unions of
fibres}`, the predicates that do not distinguish points inside a fibre — i.e.
exactly the formulas whose bound-variable dependence has been eliminated.

### 5.1 Laws (all are bitmap identities, hence decidable)

| Law | Statement |
| - | - |
| extensive / contractive | `X ⊆ ∃_c X`, `∀_c X ⊆ X` |
| monotone | `X ⊆ Y ⇒ ∃_c X ⊆ ∃_c Y`, same for `∀_c` |
| idempotent | `∃_c ∃_c X = ∃_c X`, `∀_c ∀_c X = ∀_c X` |
| distributes over the join/meet it preserves | `∃_c (X ∪ Y) = ∃_c X ∪ ∃_c Y`, `∀_c (X ∩ Y) = ∀_c X ∩ ∀_c Y` |
| De Morgan | `¬_U ∃_c X = ∀_c ¬_U X`, `¬_U ∀_c X = ∃_c ¬_U X` |
| Galois connection | for saturated `S`: `∃_c X ⊆ S ⟺ X ⊆ S`, and `S ⊆ ∀_c X ⟺ S ⊆ X` |

`∃_c` is the least saturated superset of `X`; `∀_c` is the greatest saturated
subset. This is precisely [Halmos' monadic Boolean
algebra](https://en.wikipedia.org/wiki/Monadic_Boolean_algebra) with one
quantifier, and the one-variable fragment of [Tarski's cylindric
algebras](https://en.wikipedia.org/wiki/Cylindric_algebra).

### 5.2 Quantifier elimination and decidability

Because the fibre partitions of `U` are finite, `∃_c`/`∀_c` are computable and
satisfy quantifier elimination: a formula with the bound coordinate `c` is
equivalent to a quantifier-free one, obtained by saturating its truth set along
the fibres. Iterating over every coordinate leaves a set closed under each
partition, i.e. a union of **join-classes** — the classes of the equivalence
relation generated by "share a coordinate". So the fragment is decidable (each
decision is a bitmap comparison), but a sentence need not be two-valued:

- when the coordinate family **separates** `U` — one join-class, as for the full
  plane or a product `U_reg × U_tz` — every sentence is `∅` or `U`, and truth is
  two-valued;
- otherwise a sentence takes a value in the Boolean algebra of join-class unions
  (at most `2^{#classes}` values). Measured: a random 118-site local universe has
  **4** join-classes, and a fully quantified formula lands on a proper union of
  them, not on `∅` or `U`.

The number of join-classes is therefore a measure of how far the coordinates
fail to separate the universe — the residual non-two-valuedness of truth inside
a world.

### 5.3 Worked example (toy plane)

Take `Plane = {0 … 7}`, `c(p) = p mod 4`, so `F_0 = {0,4}`, `F_1 = {1,5}`,
`F_2 = {2,6}`, `F_3 = {3,7}`.

```text
X      = {0, 5}            # a relation on (bound, free) coordinates
∃_c X  = F_0 ∪ F_1 = {0,1,4,5}      # "some value of the bound variable"
∀_c X  = ∅                          # no full fibre lies in X
Fix    = { ∅, F_0∪F_1, …, {0…7} }   # 2^4 elements ≅ 2^I
```

`∃_c X` is saturated; applying `∃_c` again changes nothing, and `∀_c ∃_c X =
∃_c X` while `∃_c ∀_c X = ∅`.

**Implementation.** With the soldered plane, `∃_c` is "OR together the fibres
that meet `X`" and `∀_c` is "OR together the fibres contained in `X`" — a mask
per coordinate class plus an OR/AND of `M` or `b` words. In `ewm-ops` the DSL
words are `exists_c:tz`, `exists_c:reg`, `forall_c:tz`, `forall_c:reg`, each
with stack effect `( x universe -- result )`: the local universe is a
first-class value declared by a `universe <name> <ref>...` directive and
pushed with `@universe:<name>` (or bare `@universe` when exactly one is
declared). The quantifier is relative to that universe — the fibres are
restricted to `L`, never the raw plane. A `conv` axis is the same mask word
once a convolution coordinate exists; it is not yet wired. Nothing else about
the representation changes, and the result is a first-class node, so it can
be committed and re-read like any other.

---

## 6. Exactness summary

| Operation | Site level (`B`) | Token-collection level (via `μ`) |
| - | - | - |
| `∨`, cover, `∃_c` (union) | exact | exact (`μ` is a join-homomorphism) |
| `∧` (`inter`) | exact | exact on saturations (= site sets); token identity: sound, incomplete (`⊆`) |
| `\` (`diff`) | exact | exact on saturations; token identity: under-approximate (`⊆` of `μ(A\B)`) |
| `¬` | exact **relative to `U`** | only defined once `U` is fixed |
| `⊕` (`symdiff`) | exact | exact (XOR is a map on sites) |
| `∀_c` | exact | exact on saturations; inherits the token-identity gap |

Proof sketches: `μ(A∪B) = μ(A)∪μ(B)` is immediate. `μ(A∩B) ⊆ μ(A)∩μ(B)`, and
`μ(A)∩μ(B) = μ(A^sat∩B^sat)`, so the meet is exact on saturations; a *strict*
inclusion needs two different tokens on one site, and then the extra site is the
alias gap, measurable from the collision rate. `μ(A)\μ(B) ⊆ μ(A\B)` for the
same reason. So for token-identity queries conjunction errs toward the
**positive** and difference toward the **negative**; for site queries there is
no error at all.

---

## 7. The ring is a different fragment: parity, not conjunction

The GF(2) ring (`docs/BOOLRING.md`) closes the commit HLLSets under `Δ` but not
under `∩`, `∪`, or `\`. In logical terms:

- the ring span is the set of truth sets expressible by **XOR (parity)
  formulas** over the basis — call this the *affine fragment*;
- `A ∪ B = A Δ B Δ (A ∩ B)` shows the meet is the single obstruction, so a
  general Boolean formula has a *linear part* (its projection into the span) and
  a *meet part* (the residual);
- `coordinates(X) ≠ None` is therefore the exact test for "`X` is an affine
  formula in this basis", and `residual(X)` is its non-affine content.

So the three poles are: **lattice = Boolean logic**, **ring = affine/parity
logic**, **FOL = Boolean logic plus quantification and substitution**. The
propositional and monadic layers of this note live in the lattice; the ring is
the linear reading of the same objects.

---

## 8. Boundary: what is *not* first-order here (out of scope)

Monadic quantification is not full FOL. What is missing, and why it is a real
gap rather than a relabeling:

- **n-ary relations.** `conv(n, dim)` does give n-gram tuples, so an n-gram
  HLLSet can sketch an n-ary relation — but as a set of *sites*, with the tuple
  structure compressed.
- **Composition / join.** `R(x,y) ⋈ S(y,z)` needs to know *which* `y` in `R`
  matched which in `S`. A bit sketch cannot: the fibre is shared, but the
  pairing is lost. Joins are recoverable only through the LUT / `h:` fibre,
  which is the identity-carrying layer (`docs/SEPARATION.md`).
- **Equality, substitution, diagonals.** The Tarski machinery for variables
  (`x = y`, renaming) is what turns a Boolean algebra into a cylindric algebra;
  the HLLSet plane has coordinates but no *variable identity*, so `x = y` is not
  a lattice operation.

That is the next level (cylindric / relation algebra / Datalog), and the
concrete question it poses is: *what has to be kept, alongside the bits, to make
the join exact?* The LUT/fibre is the candidate answer.

---

## 9. A testable checklist

Everything above is a claim that can be checked on real HLLSets:

1. **Homomorphism laws.** For random atom families and formulas: `⟦¬¬φ⟧ = ⟦φ⟧`,
   `⟦φ∧φ⟧ = ⟦φ⟧`, `⟦φ∨¬φ⟧ = U`, `⟦φ∧¬φ⟧ = ∅`.
2. **Closure laws.** `∃_c` extensive, monotone, idempotent; fixpoints = unions
   of fibres; the Galois connection with saturated sets.
3. **Quantifier laws.** `∃_c(X∪Y) = ∃_cX ∪ ∃_cY`, `∀_c(X∩Y) = ∀_cX ∩ ∀_cY`,
   `¬∃_c = ∀_c¬`, `¬∀_c = ∃_c¬`.
4. **Quantifier elimination.** Random sentences over `{tz, reg}` collapse to
   `⊤`/`⊥`; a formula with a free coordinate keeps exactly that coordinate's
   dependence.
5. **Approximation.** Measure the `∧` false-positive rate and the `\`
   false-negative rate as a function of the observed collision rate; confirm
   `∅`/`U` answers are reliable (sound refutation / sound validity).
6. **Affine fragment.** Confirm `⟦φ⟧` is in the ring span exactly for
   `XOR`-only formulas, and that `residual` is zero precisely then.
7. **Relativization and transport.** For `L' ⊆ L`, restriction `X ↦ X ∩ L'` is
   a Boolean homomorphism; local truth values that agree on overlaps glue.
8. **World-relative logic.** The same formula gets different truth sets under
   two local universes; there is no global negation; `BSS(⟦φ⟧, L)` is the
   degree of truth of `φ` in `L`.
9. **Separation.** Count the join-classes of the coordinate partitions on `L`;
   a fully quantified formula is two-valued iff that count is `1` (the full
   plane is separated).

## See also

- `docs/BOOLRING.md` — the lattice/ring, the meet obstruction, the ring tools.
- `docs/SEPARATION.md` — the fibre is global: the identity layer joins need.
- `crates/ewm-ops/src/dsl.rs` — the words: the connectives, and now the counted
  quantifiers plus the `universe` directive.
- `REFS/EWM/companion.tex` — worlds, virtual isolation and the bridge; the
  multi-universe reading of §4.
- Background: Lindenbaum–Tarski algebras; Halmos, *Monadic Boolean Algebras*
  (1956); Henkin–Monk–Tarski, *Cylindric Algebras* (1971/1985); Tarski, *On the
  Calculus of Relations* (1941); Codd (1970) for the relational reading.
