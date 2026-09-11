# Explorer — the read-only projection of the ewm-state-machine

Status: **implemented, first increment** (`ewm-sm-explore` + `StateSnapshot`).

## 1. What is explored

The state machine is one data structure spread over three locations:

```text
1. S(t) run-time  (ewm-app::StateCache)   working set, tree, turns, tip
2. cache          (designed, Arrow)        LUTs, TF, hllsetLUT, Gx versions
3. persistent     (ewm-git::Repository)    commit DAG, blobs, HEAD, hllsetLUT
```

The explorer is a **read-only projection**: it never mutates the harness,
the cache, or the store. It is the third consumer of the three-layer
contract, after the [UM] and the tests.

## 2. Commands

```text
ewm-sm-explore store <path> [--json]      persistent layer over ewm-git
ewm-sm-explore snapshot <file> [--json]   S(t) run-time area (StateSnapshot)
ewm-sm-explore help
```

The S(t) snapshot is exported by the app:

```bash
cargo run -p ewm-app -- --stub "1,2,3;2,3,4" --repo /tmp/demo \
    --snapshot /tmp/demo/snapshot.json
cargo run -p ewm-sm-explore -- snapshot /tmp/demo/snapshot.json
```

## 3. What each projection shows

### 3.1 `store` — the persistent layer

- `tip` — the HEAD pointer (the stack top);
- `context bits` — the G1 lattice top popcount;
- `lattice tops` — G1/G2/G3 popcounts (the union of all committed states);
- `hllsetLUT (repo registry)` — the repo's `<SHA1, TH>` touch registry;
- `commits (tip-first)` — per commit: id, parents, message, **G1/G2/G3 keys**
  (`Repository::state_keys` — the named Gx versions), and the G1 D/R/N view.

### 3.2 `snapshot` — the S(t) run-time area

- `tip` — the H(t-1) link into the persistent layer;
- `turn_count`, `working` G1/G2/G3 bits, `tree` root + leaf count;
- `turns` — ids, G1 leaf key, produced commit;
- `tf_base` — the TF baseline entries restored from the tip;
- `cache` — the designed Arrow layer, shown as schema stubs until
  implemented (batch names pinned in `ARROW_CACHE.md`).

### 3.3 The mapping between layers

The same identifiers appear in both projections:

| Link | S(t) snapshot | Persistent store |
| ---- | ------------- | ---------------- |
| the tip | `snap.tip` | `store.tip` |
| a turn leaf | `turn.g1_key` | commit G1 key (`state_keys`) |
| a commit | `turn.commit` | commit id in the DAG |
| D/R/N | (last outcome) | `G1 view` per commit |

This is the monitoring contract: a future `--follow` mode re-renders when
the snapshot file (or the store) changes, using the same projections.

## 4. Non-goals

- The explorer never mutates the store (read-only by construction: it only
  opens `Repository::open` and calls `read_*`/`state_*`/`view`/`log`).
- It is not a query engine; textual + JSON projections only.
- The Arrow cache layer is shown as stubs until `ewm-cache` exists.

## 5. Next increments

1. `--follow` monitor: watch the snapshot file or store directory and
   re-render on change.
2. Cache-layer projection: when `ewm-cache` lands, add a `cache <dir>`
   command rendering the Arrow batches (§4 of `ARROW_CACHE.md`).
3. Web UI: serve the same JSON projections over HTTP.
