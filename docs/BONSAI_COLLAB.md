# Bonsai + ewm-sm: the collaborative model

## 1. Why Bonsai can be the base model

Notebooks 19–20 showed the property that makes Bonsai different from the
other front-ends:

| Capability | Small LLMs (09–16) | Laya | Bonsai 2 27B |
| --- | --- | --- | --- |
| Generate text | yes | no | yes (reasoning + answer) |
| Expose reasoning trace | no | n/a | yes (`reasoning_content`) |
| Bidirectional token access | no (only out) | no tokens | **yes — `/tokenize` and `/detokenize`** |
| Large context plane | 2k–32k | 512 | **262K, RAM-tiered KV** |
| Prompt-cache / context reuse | no | no | **yes** (checkpoints, prefix cache) |
| Runs locally, open weights | yes | yes (Apache-2.0) | **yes (5.9 GB ternary)** |

The combination of *bidirectional token access* + *a huge context plane* is
what "collaborative" means here: the context is not a fixed transcript — it
is a **proposal** that ewm-sm recomputes every turn and Bonsai reasons over.

## 2. The collaboration loop

```text
                       ┌─────────────────────────────────────────────┐
                       │              ewm-bonsai (controller)        │
   user query ────────►│                                             │
                       │  1. state S(t-1)  = HLLSet lattice           │
                       │  2. context proposal P(t)                    │
                       │       full  = materialize(S) → cap           │
                       │       compact = D-part only (new tids)       │
                       │       surprise = D + Noether R/N tids        │
                       │  3. P(t) --detokenize--> text prefix         │
                       │                                             │
                       │  4. Bonsai.chat(prefix + query)              │
                       │     → answer + reasoning + usage             │
                       │                                             │
                       │  5. tokenize(answer) → tid stream            │
                       │  6. ingest → S(t), D/R/N, materialize        │
                       └──────────────┬──────────────────────────────┘
                                      │
        ┌─────────────────────────────┼───────────────────────────────┐
        │                             ▼                               │
   ewm-sm (unchanged)          PrismML llama.cpp (unchanged)          │
   ingest / materialize /       Bonsai 2 27B + KV cache +             │
   noether / project           prompt cache + 262K context            │
        └─────────────────────────────────────────────────────────────┘
```

The user never sees the raw transcript: they see the answer and the
**context proposal** that produced it, and they can change the proposal
policy at any time (`/context`). The loop is:

> query → lattice state → proposed context → Bonsai → new lattice state → …

## 3. Components

| Component | Location | Responsibility |
| --- | --- | --- |
| `BonsaiAdapter` | `trainer/adapters.py` | chat / chat_full / tokenize / detokenize over the llama.cpp server |
| `EwmScene` | `trainer/ewm.py` | the only Rust-facing module (ingest, materialize, noether, project) |
| `BonsaiSession` | `trainer/bonsai_cli.py` | the collaborative controller: state, context proposal, policy |
| `displacement_tokens` | `trainer/adapters.py` | D-part filter (tids new to the lattice) |
| CLI front-end | `trainer/bonsai_cli.py` `main()` | the REPL the user talks to |
| ewm-sm crates | `crates/` | unchanged |
| PrismML llama.cpp | `/home/alexmy/tools/Bonsai-demo` | unchanged |

Separation of concerns stays exactly as in `docs/TRAINER.md`: the controller
is outside ewm-sm, Bonsai is outside ewm-sm, and both are connected by the
two JSON surfaces (ewm-scene CLI; llama.cpp HTTP).

## 4. The context proposal (the part that is "collaborative")

A proposal is a triple `(mode, cap, detokenize)`:

| Mode | tids that enter the prefix | Use |
| --- | --- | --- |
| `full` | `memory_tokens(materialize(S))`, last `cap` | baseline transcript memory |
| `compact` | `displacement_tokens` history, last `cap` | novelty-only; repeats collapse |
| `surprise` (next) | D-part + tids from frames whose Noether displacement/indicators exceed a threshold | keeps only structurally surprising content |

The controller records, per turn:

- `prompt_tokens` (Bonsai's own cost metric),
- the proposal mode and cap,
- union popcount and content key,
- Noether indicators (`dp`, `ind1`, `ind2`, `ind3`).

This makes the context a measurable, switchable artifact — not an implicit
transcript. Notebook 20 is the first A/B of this artifact.

## 5. CLI surface (v1)

```text
> what is the capital of France?        # ask Bonsai through the lattice
/context full|compact                   # switch the proposal policy
/cap 24                                 # set the memory cap
/state                                  # S(t): pop, key, D/R/N, memory preview
/history                                # per-turn record (prompt_tokens, proposal, answer)
/policy                                 # current proposal policy
/reset                                  # clear the session
/quit                                   # exit
```

Example session:

```text
ewm-bonsai> /context compact
  proposal: compact, cap=24
ewm-bonsai> What is the capital of France?
  [compact] prompt_tokens=75  answer="The capital of France is Paris."
  state: pop 7 → 7, memory preview="Paris is the capital of France."
```

## 6. What this buys (the roadmap)

1. **A user-driven context lab.** Switch `full`/`compact`/`surprise` mid-
   conversation and watch Bonsai's `prompt_tokens` and answer quality change
   live — the A/B from notebook 20 becomes an interactive tool.
2. **Persistent, addressable sessions.** S(t) content keys + `ewm-git` allow
   saving/restoring a conversation as a content-addressed state, and diffing
   two sessions by their HLLSets.
3. **Decision-model gates.** Laya (or Jev, when the key arrives) decides the
   proposal mode and cap per turn from the trajectory features — the
   controller learns to manage Bonsai's context.
4. **Multi-model futures.** The same `BonsaiSession` interface can route to
   the three small LLMs or the heterogeneous front-ends under one router.

## 7. Non-goals (v1)

- No changes to ewm-sm crates.
- No fine-tuning of Bonsai.
- No server-side changes to the PrismML llama.cpp fork.
