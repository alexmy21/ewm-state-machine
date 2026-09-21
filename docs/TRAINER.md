# Trainer layer — dev environment, interfaces, and the DSL roadmap

**Status:** design note, written after notebooks 08–11 (2026-09-18).
The trainer/controller is a notebook-level layer outside ewm-sm; this
document freezes its interfaces and records the DSL decision.

## 1. Separation of concerns

```text
Layer                    Owns                                        Interface
─────────────────────────────────────────────────────────────────────────────────
apparatus (ewm-sm,       S(t) HLLSets, D/R/N, Boolean ring,         in:  tid{n} streams
UNCHANGED)               basis frames, materialize / project        out: trajectory JSON

trainer / controller     predictors, selectors, DFT, surprise,      in:  trajectory JSON
(new, outside ewm-sm)    memory, curiosity                          out: probe config

adapters / probes        the only model-specific code               in:  probe config
(one per LLM)                                                        out: tid{n} streams
```

The loop closes at the top: probes → ewm-sm → trajectory → trainer → probe
config → probes. The apparatus never changes; the trainer is the integrating
system that feeds itself with its own predictions.

**Verified across notebooks 08–14** (synthetic probes → real LLMs → closed
loop → predictor portfolio → learned selector → Jev router and its patterns):
every extension lives in the notebooks and the `trainer/` package, and zero
Rust crates of ewm-sm were modified. The two ewm-sm surfaces used are the
`ewm-scene` JSON protocol and the `ewm-ops` boot DSL — nothing else.

## 2. Runtime environment (already well defined)

1. **EWM state** — HLLSets over the bit plane, content-addressed
   (`h:<sha1>`), with the Boolean lattice (∪, ∩, Δ) and the GF(2) Boolean
   ring (basis, coordinates, residual, generation).
2. **Soft state** — the fixed-size BSS vector space
   `w_i = |S ∩ B_i| / |B_i|` over a named frame (the per-LLM ring). This is
   the working space of the predictors.
3. **Protocol** — `ewm-scene` JSONL in / JSON out; `ewm-ops` boot DSL for
   operational graphs.
4. **Existing DSL** — the `ewm-ops` boot DSL (Forth-like postfix, compiled,
   content-addressed vocabulary `v:<sha1>`). The Lua→Forth front-end is on
   hold, not discarded.

## 3. Interfaces (the D-in-progress, made explicit)

Three JSON shapes. The future DSL compiles down to these, plus `ewm-ops`
boot files.

### 3.1 Probe config — controller → adapter

```jsonc
{
  "step": 12,
  "query": "What is the capital of France?",   // or a query index
  "memory": ["tid576", "tid6722", "..."],     // materialized union state,
                                              // restored order, capped
  "gate": null,                               // optional allowed-vocab HLLSet key
  "temperature": null,                        // optional decoding control
  "max_new_tokens": 48
}
```

### 3.2 Trajectory record — ewm-sm → trainer (one per step)

```jsonc
{
  "step": 12,
  "union_key": "h:...",
  "union_pop": 428,
  "bss": [0.54, 0.33, 0.21],                  // m-dim soft state in the frame
  "drn": { "dp": 60, "rp": 340, "np": 80 },
  "ring": {
    "dim": 16, "residual": 0,
    "basis_generation": 16, "basis_change": false,
    "soft": [0.9, 0.1, "..."], "soft_first": ["..."], "spill_first": 0
  },
  "per_llm": {
    "llm_a": { "pop": 54, "drn": { "..." : 0 }, "ring": { "..." : 0 } },
    "llm_b": { "..." : 0 },
    "llm_c": { "..." : 0 }
  },
  "memory_ordered": ["tid576", "tid6722", "..."]   // materialized, restored order
}
```

### 3.3 Predictor / selector interface

```jsonc
// predictor: history → prediction
{ "name": "dft-periodic",
  "history": [[0.54, 0.33, 0.21], ["..."]],
  "prediction": [0.55, 0.32, 0.22] }

// selector: performance → choice
{ "weights": { "persistence": 0.12, "linear": 0.21,
               "dft-periodic": 0.19, "ridge": 0.11 },
  "choice": "ridge", "epsilon": 0.1 }
```

The predictor is a pure function of the trajectory history; the selector is
a policy over predictor performance. Both are swappable behind these shapes.

### 3.4 Decision record — decision model → controller

A non-traditional decision model (e.g. TypeSafe AI's Jev / System One)
returns typed decisions instead of text. Its output is the fourth shape:

```jsonc
{ "decision": "llm_b",
  "confidence": 0.87,
  "probabilities": { "llm_a": 0.05, "llm_b": 0.87, "llm_c": 0.08 },
  "decision_type": "route_query",
  "aux": { "needs_memory": 0.42 },
  "model": "jev-latest", "input_tokens": 410, "output_tokens": 8,
  "mock": false }
```

In the router scenario, the decision chooses which LLM answers; the record
is logged into the trajectory. Probabilities are a fixed-size vector, so
they can also feed the selector or the BSS soft state directly.

## 4. Reference implementation — `trainer/` package (implemented)

The notebook logic is extracted into a `trainer/` Python package at the repo
root (numpy-only for the core; torch only inside `LlmAdapter`):

| Module | Contents |
| --- | --- |
| `trainer/protocol.py` | `ProbeConfig`, `TrajectoryRecord`, `Prediction`, `Selection`, `DecisionRecord` — the §3 shapes as dataclasses |
| `trainer/predictors.py` | `dft_period`, `Predictor` base, persistence / linear / dft-periodic / ridge, `default_portfolio()` |
| `trainer/selector.py` | `EwmaSelector` + `LearnedSelector` (linear scoring policy over trajectory features, online ridge fit to reward = −surprise) + `selector_features` |
| `trainer/adapters.py` | `Adapter`, `SyntheticAdapter`, `LlmAdapter`, `DecisionRouter` ABC + `JevAdapter` (TypeSafe System One router with mock fallback), `decision_tokens`, `memory_tokens`, `build_prompt` |
| `trainer/ewm.py` | `EwmScene` client (the only place that talks to the Rust apparatus) + JSONL writers |
| `trainer/loop.py` | `run_open_loop`, `run_open_loop_from_streams` (capture-first front-ends), `run_closed_loop`, `run_jev_loop` (Jev router loop with confidence gate and decision-in-state) |
| `trainer/smoke_test.py` | full pipeline against real `ewm-scene` with a deterministic synthetic adapter (no torch) |

Run the self-check with:

```bash
python3 trainer/smoke_test.py
```

The package is the spec: a future DSL must reproduce this behavior from the
three JSON shapes, nothing more.

## 5. DSL roadmap and the Terra decision

- The D in DSL is emerging through §3; do not build the DSL toolchain before
  these shapes are stable.
- The future high-level DSL is a **front-end that compiles down to** the
  `ewm-ops` boot DSL + this JSON protocol — it does not replace the runtime.
- **Terra** is conceptually the right shape (Lua metaprogramming + LLVM
  native codegen) but is effectively dormant (old LLVM, little maintenance)
  and unnecessary now: predictors operate on fixed-size BSS vectors that
  numpy already handles in microseconds. Adopt Terra later only if all three
  hold: (1) D is stable, (2) runtime codegen of numeric kernels is actually
  needed, (3) the team is willing to own the toolchain.
- Lua-family preference: **LuaJIT + FFI** first (mature, alive, C ABI into
  Rust). Terra only if LuaJIT's metaprogramming proves insufficient.
- Most natural end state: Rust apparatus + `ewm-ops` boot DSL + JSON
  protocol + a thin Lua scripting surface; Python remains the exploration
  environment.
- Immediate de-risking experiment (optional): a minimal Lua-subset parser
  that emits `ewm-ops` boot DSL — the Lua→Forth idea without Terra.

## 6. Status and next

1. ✅ Three JSON shapes frozen (this document).
2. ✅ `trainer/` reference package extracted and self-checked.
3. ✅ Learned selector implemented (`LearnedSelector`, notebook 12): on the
   16-step real-LLM loop the EWMA baseline is still the stronger selector
   (meta/best 0.95 vs 1.11) — the learned selector needs a longer horizon.
4. ✅ Jev integration surface (notebooks 13–14): `DecisionRecord` shape,
   `JevAdapter` (real SDK / deterministic mock), `DecisionRouter` ABC,
   confidence-gated routing, and decision-in-state (`jev_*` tokens ingested
   into `S(t)`). The live TypeSafe API is waitlisted; the mock keeps the
   loop executable and the real key is a drop-in env var.
5. ✅ Heterogeneous front-ends (notebook 15): capture-first DeepSeek-OCR +
   NVIDIA VLA + V-JEPA streams (`captures/capture_*.py`) run through the
   unchanged ewm-sm — encoding-agnostic union `S(t)` verified across three
   modalities.
6. Next: live-Jev benchmark when the key arrives; `Score`/`Noul`
   per-action gates; a closed loop across the heterogeneous front-ends
   (memory + curiosity + front-end selection); longer closed loops and
   surprise-driven updates to the predictors themselves.
