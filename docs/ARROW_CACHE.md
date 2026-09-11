# Arrow Cache — design note for the cache layer of ewm-state-machine

Status: **design note, no code yet** (decision record 2026-09-10).

## 1. Context

The state machine is layered as:

```text
ewm-git            durable commit stack (tip = head)        — Roaring wire format
     ▲ │  load / commit
     │ ▼
StateCache         S(t) + H(t-1)                            — shareable, in-memory
     ▲ │  read / propose
     │ ▼
ewm-app ([UM])     stateless driver                         — owns only the store handle
```

The cache today is a plain Rust value built from `BTreeMap`s, `Vec`s, and
Roaring bitmaps. Most cache objects are *tables*:

- `HllsetLut` — `(key, th)` pairs;
- `TfTable` — `(token, count)` pairs;
- `LutIndex` — `(bit, token)` reverse index;
- `TFVec` — 32,768 × `f64`;
- `ContextTree` — leaf and level hash strings;
- global G1/G2/G3 slices — three bitsets over 32,768 positions.

Apache Arrow is a natural substrate for exactly these objects, and Arrow IPC
files give the cache an **embedded extended cache**: spillable, restorable,
zero-copy, and readable from Python/Rust without going through the commit log.

## 2. Decisions

| # | Decision | Rationale |
| - | -------- | --------- |
| D1 | Arrow is the **cache-layer substrate**, not the state representation. | Keep IICA content keys stable (see §3). |
| D2 | HLLSet/Roaring and the two morphisms stay native. | Bitwise lattice ops + soldered serialization are Roaring's job. |
| D3 | Arrow IPC files form the **extended cache** (spill/restore). | Cache is derived data; IPC makes it shareable and quickly rebuildable. |
| D4 | A pinned Arrow schema version joins the soldered constants. | Cache files must be interpretable across versions and languages. |

## 3. The boundary (what Arrow may and may not touch)

**Arrow backs (future `ewm-cache` crate):**

- `HllsetLut` — `<key, th>`;
- `TfTable` — `<token, count>`;
- `LutIndex` — `<bit, token>` per channel;
- `TFVec` — the 32,768-entry bit-TF vector;
- `ContextTree` — leaves and Merkle levels;
- global G1/G2/G3 slices — the cumulative per-channel union (head states);
- turn records — the presentation log.

**Arrow does NOT touch:**

- the HLLSet bit-plane itself (Roaring bitmap stays canonical);
- ingest/materialize bitwise operations;
- the content-key derivation (`h:<sha1>` over Roaring serialization);
- the commit objects and the wire format of `ewm-git`.

Consequence: switching the cache to Arrow changes **no** `h:`/`t:` keys and
breaks no crosscheck with `ewm-fpga-bridge` or `hllset-fpga-simulator`.

## 4. Cache objects and their Arrow schemas (v1)

The cache is a set of named RecordBatches. Every batch is sorted in a
canonical order so two equal caches produce byte-identical IPC payloads
(IICA-friendly determinism).

Cache objects fall into three classes:

| Class | Objects | Semantics |
| ----- | ------- | --------- |
| **content-addressed immutable** | G1/G2/G3, `tf_vec` | named by their SHA1; the name IS the content ID |
| **append-only** | `lut_g1/g2/g3`, `hllset_lut` | grow monotonically; entries are immutable (hash-determined) |
| **context-local** | `tf_table` | built for the current context only; ephemeral |

### 4.0 Bootstrap schemes — same Gn, separate LUTs

n-grams and n-seeds are two ways of **bootstrapping token presentation** in
an HLLSet. The HLLSet itself is bootstrap-scheme agnostic, and there is
**one G1, one G2, one G3** — the same channels serve both schemes. The token
LUTs are kept **separate per scheme**, and the scheme prefix lives on the
**LUT name**, not on the Gn HLLSet key:

```text
G1/G2/G3         h:<sha1>    scheme-agnostic channel HLLSets (shared)
ng:G1 … ng:G3    n-gram LUTs — order can be restored (window chain)
ns:G1 … ns:G3    n-seed LUTs — plain set only (seeded hashes are orderless)
```

Given a recovered Gx, materialization picks the LUT whose name prefix
matches the requested bootstrap scheme. The prefix also answers the order
question for recovered tokens: `ng` means the n-gram LUTs know the original
order; `ns` means only the set survives.

**Gn are gates.** Each channel is the union of its two scheme components:

```text
G1 = G1_ng ∪ G1_ns      G2 = G2_ng ∪ G2_ns      G3 = G3_ng ∪ G3_ns
```

`gate(Gx, H) = Gx ∩ H` extracts H's channel component — it extracts bits,
nothing more. **Bits are anonymous**: a bit does not remember its parent —
the same bit may have been set by a 1-gram, a seed-0 hash, or PAD
(collisions are fine; resolving them is the materializer's problem, solved
by bootstrapping + disambiguation). Interpreting the extracted bits
(1-gram vs seed-0 atoms) happens only in materialization, in the context
of a specific HLLSet and a chosen LUT.

### 4.1 `hllset_lut` — named HLLSets, append-only, context-scoped

```text
schema: (name: Utf8, key: Utf8, th: UInt64)
sorted by: (name, key)
```

Named HLLSets are registered **under their name**: G1, G2, G3 are the names
of the channel HLLSets created by ingest. A named HLLSet is immutable —
updating any Gx creates a **new** Gx HLLSet (new SHA1); the old entry stays
registered and addressable. Append-only and held **in memory**; contains
only the HLLSets relevant to the current context (not a global registry).
Registration is idempotent, TH is monotonic.

### 4.2 `tf_table` — current context only

```text
schema: (token: Binary, count: UInt64)
sorted by: token
```

Built for the **current context** (the working set of the active turns), not
a global table. It is the token-level TF that materialize uses for
tie-breaking; it may be rebuilt from the LUTs and does not need to survive
context switches.

### 4.3 token LUTs (`ng:G1` … `ng:G3`, `ns:G1` … `ns:G3`) — append-only

Six batches — one LUT per scheme per channel (the Gn HLLSets themselves are
shared and scheme-agnostic):

```text
schema: (bit: UInt32, token: Binary)
sorted by: (bit, token)
```

`bit` is the flat bit address `reg * 32 + tz` (the `BitAddress` wire form).
The pointed-to token is stored, not the n-gram bytes — same convention as
`hllset-morphisms::api`.

**Append-only immutability due to hash**: a token's fiber is determined by
its hash, so entries are never updated or removed — the LUT only grows.

### 4.4 `tf_vec` — content-addressed by SHA1

```text
schema: (index: UInt32, value: Float64)
row count: 32768 (index = bit position, 0..32767)
```

The TF vector is an immutable, content-addressed value: its name is its
SHA1. Dense alternative: a single `FixedSizeList<Float64, 32768>`. The
two-column form is chosen because it is easier to diff and to read from
Python without knowing the fixed size in advance.

### 4.5 `context_tree`

Two batches:

```text
tree_leaves:  (h: Utf8, lut: Utf8, view: Utf8)   sorted by: (h, lut, view)
tree_levels:  (level: UInt32, index: UInt32, hash: Utf8)  sorted by: (level, index)
```

`tree_leaves` is the flattened leaf/views relation; the tree itself is
rebuilt canonically from it (`ContextTree::build`).

### 4.6 global slices — G1, G2, G3 are HLLSets

G1, G2, and G3 are **HLLSets**, and their content-addressable IDs are the
names `G1`, `G2`, `G3`. In the cache they are stored as their Roaring
serialized bytes under `h:<sha1>`; the manifest binds the logical names to
those keys.

**Gx versioning per commit.** Each commit records the G1/G2/G3 SHA1s it
committed (`ewm-git::Repository::state_keys`, `CommitView::state_key`).
Updating any Gx creates a new immutable Gx HLLSet — the old version remains
addressable. Because the token LUTs are append-only, the pair
`(Gx version of commit t, current token LUTs)` restores the tokenLUT state
at any past commit point: materialize that commit's Gx over the LUTs.

For columnar interchange a sparse projection is available:

```text
schema: (bit: UInt32)
sorted by: bit
```

The sparse form is a view; the HLLSet bytes remain the canonical object. A
dense `Boolean[32768]` view can be derived when a consumer needs bitwise
work.

### 4.7 turns

```text
schema: (turn: UInt64, id: UInt32)
sorted by: (turn, id)
```

Long form of the turn records (one row per token occurrence).

## 5. Extended cache: Arrow IPC file layout

```text
<cache_dir>/
├── MANIFEST.arrow        # one batch: schema_version, tip, created_at, name → sha1
├── objects/
│   ├── <sha1>.hllset     # Roaring bytes of G1 / G2 / G3, named by their content ID
│   └── <sha1>.tfvec      # TFVec bytes, named by its content ID
└── tables/
    ├── hllset_lut.arrow  # append-only, context-scoped
    ├── ng:G1.arrow       # n-gram LUTs (order-preserving), append-only
    ├── ng:G2.arrow
    ├── ng:G3.arrow
    ├── ns:G1.arrow       # n-seed LUTs (orderless), append-only
    ├── ns:G2.arrow
    ├── ns:G3.arrow
    ├── tree_leaves.arrow
    ├── tree_levels.arrow
    └── turns.arrow
```

`tf_table` is **not spilled**: it is context-local and rebuilt with the
current context.

Rules:

1. **One IPC file per batch.** Partial updates rewrite only the changed
   batch; no whole-cache rewrite.
2. **Content-addressed objects live under their SHA1** (`objects/`): G1, G2,
   G3, and `tf_vec` are immutable values; their file name is their identity.
3. **`MANIFEST.arrow` is the atomic commit point** of a cache snapshot: it
   binds the logical names (`G1`, `G2`, `G3`, `tf_vec`) to their SHA1s and
   names the tip + the SHA1 of every table batch. A snapshot is valid iff all
   referenced objects and batches match their SHA1.
4. **Schema version** is a soldered constant, `ARROW_CACHE_SCHEMA_VERSION = 1`
   (to live in `hllset-contracts` when the cache crate is implemented).
5. **Restore order**: read `MANIFEST` → verify SHA1s → rebuild the native
   `StateCache` (the cache is derived data; any corrupt batch can be
   recomputed from the `ewm-git` store).
6. The cache directory may carry a content key (`c:<sha1>` over the
   manifest) so snapshots are addressable like everything else.

## 6. `StateCache` integration sketch

The `StateCache` API the [UM] touches does not change shape; its backing
store becomes Arrow RecordBatches behind a small backend trait:

```rust
// sketch only — no code yet
trait CacheBackend {
    fn get(&self, name: &str) -> Option<RecordBatch>;
    fn put(&mut self, name: &str, batch: RecordBatch);
    fn flush(&mut self) -> Result<()>;   // write dirty batches to IPC files
}

struct StateCache {
    backend: Box<dyn CacheBackend>,      // memory or arrow-ipc
    // ... same public accessors as today (tree(), turns(), working_g1(), ...)
}
```

- In-memory backend: RecordBatches held in RAM (today's behavior, columnar).
- Arrow-IPC backend: the same RecordBatches spilled to §5's directory.
- `StateCache::restore(repo)` stays the source of truth for recovery; the
  Arrow IPC cache is an **acceleration path** that may skip the commit-log
  walk when the manifest is valid and newer than the tip.

## 7. Costs, trade-offs, non-goals

**Costs**

- `arrow-rs` is a large dependency; feature-gate it and keep it out of the
  core crates (`hllset-core`, `hllset-morphisms`, `ewm-git`).
- Schema versioning becomes a contract we must maintain.
- The cache is derived data — Arrow gives zero-copy and spill, not new
  durability guarantees.

**Trade-offs to settle at implementation time**

- `arrow-rs` (official) vs `arrow2` (lighter): start with `arrow-rs`,
  feature-gated, unless the dependency cost proves too high.
- Dictionary-encode `token` columns (`DictionaryArray`) for LUT-heavy caches.
- Dense vs sparse slices: keep both views available; sparse is canonical.

**Non-goals (explicitly out)**

- Replacing Roaring/HLLSet with Arrow bitmaps.
- A query engine (DataFusion) or Arrow Flight RPC.
- Changing any `h:`/`t:` content key.
- Making `ewm-git` Arrow-based.

## 8. Open questions for the implementation review

1. Checkpoint policy: spill after every commit, on an explicit `checkpoint()`,
   or when the in-memory cache exceeds a byte budget?
2. Should the token LUT batches be merged into one `(channel, bit, token)`
   batch, or stay as three named batches?
3. Does the scene-analytics notebook become the first Python consumer of the
   IPC cache, or do we keep the CLI bridge for now?

## 9. Acceptance criteria (for the future spike)

- IPC round-trip of every batch in §4 is byte-identical after re-sorting.
- Restoring a cache from IPC equals a cache rebuilt from the store.
- No `h:`/`t:` key changes before vs after the Arrow-backed cache.
- Core crates build without `arrow-rs` in their dependency tree.
