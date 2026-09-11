# Separation of concerns — the bit is the fiber

The `conv(n, dim)` generalization does not touch the foundation; it only
moves the boundary between **encodings** and **LUTs**. The contract:

1. **No changes on the HLLSet side.** HLLSet is a bitmap — union,
   intersection, difference, popcount, SHA1. Nothing in the conv
   generalization touches `hllset-core`.

2. **Changes live only between encodings ↔ LUT.** `dim` is a property of
   the encoding (how tokens compose into windows) and of the LUT (what a
   fiber points at) — never of the HLLSet.

3. **If a token is a tuple, the tuple size is the dim.** `dim=1`: scalar
   tokens. `dim=2`: pair tokens (a grid cell read as a `(row, col)` tuple).
   `dim=3`: triple tokens (`(r, c, ch)` — RGB, depth, time). `dim=N`:
   N-tuples.

4. **The tuple is recorded into the LUT as a token.** The LUT fiber stores
   the pointed-to tuple exactly as it was ingested. The morphism serializes
   tuples to bytes (NUL-joined); the LUT sees a token, not its structure.

5. **The order of dimensions is private between the token collection and
   the LUT.** Ingest and materialize share a lexicographic convention
   (last-axis fastest), but nothing outside that pair depends on it — the
   HLLSet keys, the gates, and the store are order-agnostic. The order does
   not need to be frozen anywhere.

   *The reason:* the hash function is **deterministic** (idempotent),
   **immutable**, and **content-addressed**. If one presentation has
   `dim1 = width, dim2 = height` and another has `dim1 = height,
   dim2 = width` with the same values, the two layouts serialize to
   different window bytes — so they hash to different bits and are recorded
   as **two different presentations for the same size, measured for
   different objects**, in different HLLSets. The shared `1×…×1` channel
   (G1) is identical in both (the token set is orientation-free); the
   window channels and the projection keys differ. Orientation is part of
   the content — nothing global needs to know about it.

6. **The sync is between a given HLLSet and the corresponding Gn** — `n`
   is the window size. Each channel is a gate: `Gn ∩ H` extracts the
   n-window component of any HLLSet H. The gate is dimension-agnostic; the
   channel's seed is `seed(n, dim) = (dim − 1)·3 + (n − 1)`, with
   `seed(1, ·) = 0` (the shared token channel).

7. **The bit is the fiber.** One bit address links:

   ```text
   HLLSet membership  ↔  Gn membership  ↔  LUT position  ↔  real tokens
                                                        (a tuple of tokens,
                                                         or a tuple of
                                                         tuples … of tokens)
   ```

   Bits stay anonymous; the LUT restores their origin.

## The layers

```text
real tokens (tuples of tuples)            the application's world
        │  conv windows, dim-aware
        ▼
encodings (ngram / grid / tensor)        seeds per (n, dim); LUT fibers
        │  hash to a bit
        ▼
the bit — the fiber                       HLLSet ∩ Gn ∩ LUT position
        │
        ▼
HLLSet (bitmap) + Gn (gates)             unchanged foundation
```

Everything above the bit is the morphism's business; everything at and
below the bit is the foundation's business.
