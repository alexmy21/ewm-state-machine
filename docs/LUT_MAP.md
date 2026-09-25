# LUT Map — run-time and Arrow cache

Where every "LUT" lives, what it maps, and what changed in the router/gate
work (notebooks 21–25 + the Rust gate implementation). The key confusion to
kill first: **there are two unrelated families both called "LUT".**

---

## 1. The two LUT families

```text
   ┌─────────────────────────────────────────────────────────────────────┐
   │                              LUTs                                   │
   └───────────────┬───────────────────────────────┬─────────────────────┘
                   │                               │
     ┌─────────────▼─────────────┐     ┌───────────▼────────────────────┐
     │  TOKEN LUT (reverse idx)  │     │  HLLSET LUT (touch registry)   │
     │  bit ──► {tokens}         │     │  sha1 ──► TH (touch count)     │
     └─────────────┬─────────────┘     └───────────┬────────────────────┘
                   │                               │
     run-time: LutIndex                run-time: HllsetLut (×2 flavours)
     Arrow:    tables/ng:G1.arrow …    Arrow:    tables/hllset_lut.arrow
               tables/ns:G1.arrow …
```

- **Token LUT** = the reverse index that *materialization* walks: a bit's
  fiber holds every token that hashed into that bit. This is what restores
  tokens from bits.
- **HLLSet LUT** = the preservation side effect: which HLLSets exist, under
  which name, and how often they were touched. It ranks HLLSets, never
  restores tokens.

They share the word "LUT", nothing else.

---

## 2. The scheme invariant (before the structures)

One G1, one G2, one G3 — shared by both bootstrap schemes. The scheme lives
on the **LUT name** and now also on the **HLLSet key mark**:

```text
                      ┌────────────────────────────────────────┐
                      │        shared channel HLLSets          │
                      │   G1 = 1-gram = seed-0   (identical)   │
                      │   G2 = 2-gram / seed-1                 │
                      │   G3 = 3-gram / seed-2                 │
                      └──────────────┬─────────────────────────┘
                                     │
        ┌────────────────────────────┴────────────────────────────┐
        │                                                         │
┌───────▼───────────────────────┐             ┌───────────────────▼────────────┐
│  n-gram  (ordered stream)     │             │  n-seed  (unordered catalog)   │
│  LUTs: ng:G1 ng:G2 ng:G3      │             │  LUTs: ns:G1 ns:G2 ns:G3       │
│  keys: h:<sha1>               │             │  keys: c:<sha1>                │
│  order: restorable (window    │             │  order: set only (orderless)   │
│         chain, De Bruijn)     │             │                                │
└───────────────────────────────┘             └────────────────────────────────┘
```

Rules:

- An HLLSet is **one scheme only** — never a mix of n-grams and n-seeds.
- The materializer must run against the LUTs of the scheme that built the
  HLLSet (`ng:` LUTs for streams, `ns:` LUTs for catalogs).
- `G1` is the bridge: comparing across schemes, only `G1` matches verbatim
  (see `bss_g1` / G1-scoped gating).

---

## 2b. n-gram vs n-slide — one slide algorithm, two applications

```text
                    THE SLIDE
   window of n consecutive tokens/cells, hashed with the channel seed,
   atom set in the channel HLLSet
        │
        ├── n-gram  (dim=1, ordered token sequence)
        │     window POINTS AT its first component
        │     LUT insert: fiber ← pointed-to token
        │     order restorable (De Bruijn chain)
        │     channels: G1/G2/G3, LUTs ng:G1 … ng:G3
        │
        └── n-slide (window projection, NO LUT insert)
              window atom set only — the window is the unit of projection
              used for:
              ├── conv windows       G2_2d, G3_2d … (LUT = window anchor,
              │                      where materialization needs one)
              ├── order side channels S4_1d (slide4), S4_2d, S4_nd
              │                      (joint edge checks only, seed n+…)
              └── sliding windows    the same slide over a series —
                                     projection/average of the window,
                                     never a token vocabulary
```

This is exactly why `slide4` carries the NO-LUT flag: `slide4 = conv(4, 1) = S4_1d`
is an **n-slide** window reused from the n-gram algorithm — the slide
mechanics are identical, but there is no pointed-to token, so nothing is
inserted into a LUT and the channel can only *verify* edges, not restore
tokens.

---

## 3. Run-time map (memory)

```text
   ewm-app (StateCache)                     ewm-git (Repository)
   ┌─────────────────────────────┐          ┌──────────────────────────────┐
   │ working[3]  cumulative G1/G2/G3        │ Ingestor (streaming)         │
   │ ring        basis history   │          │   per pass: fresh Ingest     │
   │ tip, turns                  │          │   cumulative: token_tf,      │
   └─────────────┬───────────────┘          │               bit_tf (TFVec) │
                 │                          │ HllsetLut   ObjectId → TH    │
                 │ commits LatticeState     │ BitTf       bit → TF         │
                 ▼                          │ gates/      user → catalog   │
        ┌───────────────────┐               └──────────────┬───────────────┘
        │ hllset-morphisms  │                              │
        │  (the scheme layer)                              │
        └───────────────────┘                              │
                                                           ▼
                                        ┌──────────────────────────────────┐
                                        │  hllset-lut                      │
                                        │  LutIndex  bit → {tokens}        │
                                        └──────────────────────────────────┘
```

### 3.1 `hllset-lut::LutIndex` — the only true reverse index

```rust
pub struct LutIndex {
    by_bit: BTreeMap<u32, BTreeSet<Token>>,
}
// insert_token_seeded(token, seed)   hash → bit, insert into fiber
// insert_token_at(token, bit)        insert into an explicit fiber
// fiber(bit) -> BTreeSet<Token>
// materialize(hllset) -> BTreeSet<Token>   ⋃ fibers over active bits
```

One `LutIndex` per channel per ingest. It does not know the scheme; the
scheme is in the name attached to it (`ng:` / `ns:`).

### 3.2 The two ingest holders

```text
   Ingested  (n-gram, ordered)                Ingest  (n-seed, unordered)
   ┌───────────────────────────────┐         ┌───────────────────────────────┐
   │ sketches[3]  1/2/3-gram bits  │         │ hllsets[3]  seed-0/1/2 bits   │
   │ luts[3]      ng:G1 ng:G2 ng:G3          │ luts[3]     ns:G1 ns:G2 ns:G3 │
   │ tf           TfTable (token→count)      │ tf          TfTable           │
   │ slide4       S4_1d = conv(4,1)       │                               │
   │              n-slide (seed 3,        │                               │
   │              order-only, NO LUT)     │                               │
   │ projection   G1 ∪ G2 ∪ G3     │         │ projection() G1 ∪ G2 ∪ G3     │
   │ key/keys     h:<sha1>         │         │ key()/keys() c:<sha1>  (NEW)  │
   └───────────────────────────────┘         └───────────────────────────────┘
        ordered round-trip                       set restoration only
```

`slide4` is real but invisible in notebook outputs: it is populated during
ingest (`conv(4,1)` n-slide windows, seed 3, no LUT insert) and consulted
only by the ordered walk (`slide4_contains`) to make De Bruijn transitions
collision-free. The content channels the notebooks see are exactly
`G1 ∪ G2 ∪ G3` = 1-/2-/3-gram.

### 3.2b Fixed vs parameterized

```text
   default api::ingest (n-gram)     FIXED  n = 3   (CHANNELS = 3 const,
                                    seeds [0,1,2], for n in 1..=CHANNELS)
   morphisms::Ingest (n-seed)       FIXED  3 seeds (SEEDS = [0,1,2],
                                    N_SEEDS = 3)
   conv path  ConvSpec { n, dim }   PARAMETER  n and dim (dim=1 = n-gram
                                    regime; seed formula covers n > 3)
   ewm_git::Ingestor (streaming)    PARAMETER  seeds: &[u64]; ewm-app
                                    constructs it with &[0, 1, 2]
```

### 3.3 The two HLLSet LUTs (touch registries)

```text
   hllset_morphisms::HllsetLut              ewm_git::HllsetLut
   ┌──────────────────────────────┐          ┌──────────────────────────────┐
   │ (name, key) → TH             │          │ ObjectId → TH                │
   │   ("G1", "h:…") → 3          │          │   repo-level registry,       │
   │ append-only, CRDT merge      │          │   ranked() in `store` output │
   └──────────────────────────────┘          └──────────────────────────────┘
   ingest side effect                        repository side effect
```

### 3.4 TF tables (adjacent, not LUTs)

```text
   TfTable      token → count        (morphisms, per ingest)
   TFVec        bit   → TF           (hllset-core)
   BitTf        bit   → TF over lattice top   (ewm-git, commit-linked)
```

---

## 4. Materialization data flow (ungated)

```text
   HLLSet (bits) ──────────────┐
                               ▼
   for each active bit ──► LutIndex.fiber(bit) ──► candidate tokens (set)
                               ▲
   LUTs picked by scheme ──────┘   ng: for ordered, ns: for unordered

   ordered path (n-gram only):
   candidates ──► order_tokens ──► De Bruijn walk over G2/G3 edges,
                                    count-constrained by TfTable
```

## 5. Gated materialization (this session)

```text
   gate-HLLSet G_u  (catalog: c:<sha1>, built by n-seed Ingest)
                         │
                         │  G1 scope when crossing schemes:
                         │  gated = G_u.G1 ∩ frame.G1
                         ▼
   materialize against the gate's OWN LUTs (its ingest) ──► codebook filter
                         │
                         ▼
   restored = user's tokens present, exact up to hash collisions

   functions: unordered_tokens_gated  (n-gram sketches + n-gram gate)
              ingest_gated_set        (n-seed Ingest + gate)
              bss_g1                  (G1-scoped similarity)
```

---

## 6. Arrow cache (`ewm-cache`)

```text
   <cache_dir>/
   ├── MANIFEST.arrow              # name → sha1, tip, schema_version
   │                               # key: c:<sha1 of manifest bytes>
   ├── objects/
   │   ├── <sha1>.hllset           # Roaring bytes of G1/G2/G3
   │   └── <sha1>.tfvec            # TFVec bytes
   └── tables/
       ├── ng:G1.arrow … ng:G3.arrow   ┐
       ├── ns:G1.arrow … ns:G3.arrow   │ token LUTs, one schema:
       ├── G2_2d.arrow …               │   (bit UInt32, token Binary)
       │                               │   sorted by (bit, token),
       │                               │   append-only, one IPC/batch
       ├── hllset_lut.arrow        # (name, key, th)
       ├── tree_leaves.arrow / tree_levels.arrow
       └── turns.arrow
```

- The cache holds **derived data only** — any corrupt batch is recomputable
  from `ewm-git`.
- HLLSets stay Roaring-native; Arrow is the spill/restore substrate.
- `MANIFEST.arrow` is the atomic commit point; the cache directory carries
  the `c:` content key.

---

## 7. What changed in the router/gate work (delta)

The LUT **structures did not change**. `LutIndex`, `Ingest`/`Ingested`
fields, `TfTable`, both `HllsetLut`s, and all Arrow schemas are as before.

```text
   CHANGED
   ├── HLLSet::content_key_c()           c:<sha1> catalog mark (new)
   ├── Ingest::key()/keys()              now emit c: (was h:)
   ├── FrameSet.g1s / Dimension.g1       per-frame / per-dimension G1 sketches
   ├── Repository.put_gate/gate/…        gates/<user>.catalog (token list,
   │                                     first line c:<sha1> — NOT a LUT)
   └── new LUT consumers                 unordered_tokens_gated,
                                         ingest_gated_set, bss_g1

   UNCHANGED
   ├── LutIndex structure
   ├── Ingest / Ingested fields
   ├── TfTable / TFVec / BitTf
   ├── HllsetLut (both)
   ├── ng:/ns: LUT names
   └── Arrow lut_schema (bit, token)
```

---

## 8. The shared reverse LUT — implemented

The Arrow schema already existed (`lut_schema`: one file per batch), so the
shared reverse LUT was pure plumbing, now in place:

```text
   run-time ingest                          Arrow cache (shared)
   Ingested.luts[ch] / Ingest.luts[ch]      tables/ng:G1.arrow … ns:G3.arrow
        │  LutIndex::rows()                        ▲
        └──────────── ExtendedCache::merge_lut() ──┘   (append-only union,
                                                       dedup by (bit, token))
   HLLSet (bits)  ──►  ExtendedCache::materialize_lut(hllset, table)
                       = ⋃ fibers over active bits — no live ingest needed
```

- `LutIndex::rows()` exports fibers as sorted `(bit, token)` rows.
- `ExtendedCache::merge_lut(name, &index)` merges one ingest into the
  shared table (monotone union).
- `ExtendedCache::materialize_lut(&hllset, table)` is the shared-LUT
  materializer: probabilistic restoration (every candidate of an active
  bit is kept), unfiltered, with no `Ingest`/`Ingested` alive.
- **Auto-merge is wired**: `ewm-git::Ingestor::ingest_stream` now leaves
  the per-pass `luts` in `IngestOutput`, and `ewm-app::StateMachine` with an
  attached cache (`attach_shared_lut`) merges them into `ns:G1…ns:G3` on
  every turn.

What remains for full production: a CLI surface that materializes
arbitrary commit states through the shared tables.
