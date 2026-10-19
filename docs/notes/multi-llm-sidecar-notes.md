# Multi-LLM side-car notes

## Proposal (session end-state)

Seven notebooks now prove the same point: **ewm-state-machine serves many AI
models with almost no changes** — the only model-specific part is a tiny
adapter that turns an LLM's output into a `tid{n}` token stream. Done for
Qwen-Drive (03), Flux shim (05), and V-JEPA (07).

Next: a **notebook-based environment** running the same ewm-sm as a side-car
against **multiple LLMs at once**, with the system state as their lattice
join:

```text
S(t) = S_1(t) ∪ S_2(t) ∪ … ∪ S_m(t)        S_i(t) = HLLSet state of the i-th LLM
```

The m-dimensional BSS similarity vector associates with S(t):

```text
bss(t) = [w_1(t), …, w_m(t)],   w_i(t) = |S(t) ∩ B_i| / |B_i|
```

where the basis B = (B_1, …, B_m) is the ring built over the per-LLM
HLLSets. The sequence bss(1), bss(2), … is the **system trajectory**.

## The hierarchy — one operator, three levels

The proposal introduces a family of patterns; all of them are the same
operator applied at different levels of a recursive structure:

| Pattern | Leaves | System state | Decomposition |
| - | - | - | - |
| `(LLM, Ring)` | m physical LLMs | union of per-LLM HLLSets | m-dim BSS vector over the per-LLM ring |
| `(LLM, Ring, DRN)` | m physical LLMs | union stream | DRN of the union stream AND of each S_i(t) |
| `(Ring, Ring, DRN)` | rings instead of physical LLMs | union of per-ring states | DRN over ring states (recursion) |

`(Ring, Ring, DRN)` is the recursion step: a ring is just another HLLSet
stream, so a ring can be a "virtual LLM" — the whole hierarchy is a
training ground for the system.

## Why this is low-risk

Everything maps onto existing machinery:

- **`S(t) = ∪ S_i(t)` already exists.** Notebook 04's `ewm-scene pyramid`
  ingests m perceptrons per frame and computes the union top-perceptron.
  Replacing the perceptrons with LLM outputs changes only the probe.
- **The m-dim BSS vector already exists.** Basis frames
  (`docs/BASIS_FRAMES.md`) project any set into a basis as
  `soft = [w_1, …, w_k]`; per-LLM dimensions are the special case where the
  basis vectors are the per-LLM HLLSets (or their ring bases).
- **DRN is already stream-agnostic.** `ewm-scene` computes D/R/N for any
  HLLSet stream; applying it per-LLM and to the union is the same call in a
  loop.
- **The adapter is the only model-specific code.** A new LLM = a `probe()`
  that emits `tid{n}` strings into `frames.jsonl`. Nothing in `ewm-app`,
  `ewm-scene`, or `ewm-ops` changes.

## Milestone plan (proposed for the new session)

1. **Scaffold** — notebook 08 `multi_llm_sidecar.ipynb`: define 3 synthetic
   "LLM probes" (seeded token streams, optionally real small models later),
   run `ewm-scene pyramid` with m perceptrons per frame → union state S(t).
2. **`(LLM, Ring)`** — per-LLM rings/bases; m-dim BSS vector per step;
   plot the system trajectory; measure basis changes per LLM.
3. **`(LLM, Ring, DRN)`** — DRN of the union stream and of each S_i(t);
   the Noether balance table per LLM and for the system.
4. **`(Ring, Ring, DRN)`** — feed the per-LLM ring states back in as the
   "leaves" of a second-level pyramid; recursive decomposition.
5. **Training ground** — define the trajectory dataset (BSS vectors, DRN
   balances, basis-change events) and a first training/eval target for the
   system (e.g., predict the next BSS vector / next commit signal).

## Session

Start a new session — the repo work continues in
ewm-state-machine, but the conversation context should be fresh.
Kickoff for the new session:

- Project: /home/alexmy/SGS/SGS_lib/fractal_manifold_gen2/ewm-state-machine
- Task: implement the multi-LLM side-car environment as notebook 08.
  Start by reading docs/notes/multi-llm-sidecar-notes.md (this file),
  notebook 04 (`perceptron_pyramid` — the m-perceptron union already exists),
  docs/BASIS_FRAMES.md (m-dim soft projection), and notebook 07
  (`jepa_ewm_state_machine_cooperation` — the tiny-adapter pattern).
- First milestone: notebook 08 with 3 synthetic LLM probes, union state
  S(t) = ∪ S_i(t), and the m-dim BSS trajectory — then grow the pattern
  matrix (LLM, Ring) → (LLM, Ring, DRN) → (Ring, Ring, DRN).

## Session 2026-09-18 — notebook 08 implemented

`notebooks/08_multi_llm_sidecar.ipynb` is executed and checked in:

- **Scaffold** — three seeded LLM probes (`llm_a` chatty drift, `llm_b`
  bursty, `llm_c` cyclic) feed `ewm-scene pyramid`; union `S(t)` discovered
  by two paths agrees on all 64 content keys (IICA).
- **(LLM, Ring)** — per-LLM `sidecar` rings (basis generations 44/11/9),
  m-dim BSS trajectory over the per-LLM cumulative frame, time-travel spill
  through each LLM's first basis.
- **(LLM, Ring, DRN)** — Noether balance table per LLM and for the system
  (`Σ(|N|−|D|)` per stream).
- **(Ring, Ring, DRN)** — materialized ring states as virtual LLMs; level-2
  union keys, DRN and ring dims/residuals all match level 1.
- **Training ground** — fixed-shape trajectory dataset; persistence is the
  next-BSS baseline (MSE 0.0026), and a NumPy logistic probe recovers the
  next-commit signal on held-out steps (1.00 test vs 0.75 majority).

## Session 2026-09-18 (cont.) — notebook 09, real small LLMs

`notebooks/09_multi_llm_sidecar_real.ipynb` (kernel: `ewm-nanolm`) replaces
the synthetic probes with three cached real LLMs:

- `llm_a` DeepSeek-R1-Distill-Qwen-1.5B, `llm_b` Qwen2.5-1.5B-Instruct,
  `llm_c` gpt2 (GPT-Neo-1.3B crashed under transformers 5.16.1 generate).
- 16 queries × 2 cycles (T=32), greedy decoding, 48 tokens/frame; the
  side-car discovers the period: `bss(t+16) == bss(t)` exactly and every
  ring plateaus at dim 16.
- Pattern matrix reproduced with real streams: union keys 32/32, level-2
  `(Ring, Ring, DRN)` recursion all True, Noether table, time-travel spills.
- Training ground: persistence MSE 0.0022 / linear 0.0059 vs cycle replay
  and 1-NN cycle model **0.0** on the second cycle; commit rate 1.0 (live
  streams always have new bits), 15 basis-change events all in cycle 1.
- Next: sampled (non-greedy) decoding for non-periodic streams, then a
  sequence model over the trajectory dataset.

## Session 2026-09-18 (cont.) — notebook 10, closed loop (DFT + memory + curiosity)

`notebooks/10_multi_llm_sidecar_loop.ipynb` (kernel: `ewm-nanolm`) closes the
loop **around** the unchanged apparatus:

- **Pavlov phase** — 8-query bank × 3 cycles (T=24), no feedback. The DFT
  observer detects period 8 from both the BSS trajectory and the frozen-basis
  soft keys (`sidecar --freeze 8`).
- **Closed loop** — T=16 steps: memory = materialized union state in restored
  order fed back as `memory: tid… tid…`; curiosity = DFT-periodic predictor
  chooses the query whose Pavlov prototype is farthest from the prediction;
  surprise = L2 prediction error (0.077 → 0.023 across halves).
- The trajectory leaves the Pavlov prototypes (mean distance 0.18), 4/8
  queries chosen, apparatus calls unchanged (pyramid / ingest / sidecar
  --freeze / project / materialize).
- Separation of concerns: ewm-sm = state apparatus; trainer/controller =
  notebook-level integrator that reads trajectory JSON and writes probe
  prompts; the surprise series is the training signal for the next layer.
- Next: replace the DFT-periodic predictor with a learned sequence model and
  let surprise update the controller policy.

## Session 2026-09-18 (cont.) — notebook 11, predictor portfolio + meta-selection

`notebooks/11_multi_llm_sidecar_predictor.ipynb` (kernel: `ewm-nanolm`)
isolates the predictor as a swappable portfolio and adds an online
meta-controller:

- **Portfolio** — persistence / linear / DFT-periodic / online ridge (L2 fit
  over `[x(t), Δx(t), sin, cos]` phase features) behind one interface.
- **Offline benchmark on the Pavlov cycle** — DFT-periodic best (0.081),
  ridge 0.109, persistence 0.119, linear 0.210.
- **Closed loop with meta-selection** — EWMA surprise per predictor +
  ε-greedy choice (warm-up random). Persistence dominated early (11 picks),
  ridge gained trust late (3 picks, lowest final EWMA 0.113); DFT-periodic —
  best on the Pavlov cycle — was the worst in the loop (0.143), showing the
  regime shift.
- Meta mean surprise 0.093 vs best single 0.088 (1.06×) — the exploration
  cost of learning to choose; LeCun encoder→context→predictor separation
  holds with HLLSets as contexts, no decoder.
- Next: a learned selector conditioned on trajectory features instead of
  EWMA, and surprise-driven updates to both predictors and selector.

## Session 2026-09-18 (cont.) — trainer dev environment documented

`docs/TRAINER.md` written: separation-of-concerns map, the three protocol
interfaces (probe config / trajectory record / predictor+selector), the
`trainer/` reference-package plan, and the DSL roadmap. Decision recorded:
do not build the DSL toolchain around Terra now — the future DSL is a
front-end compiling down to the `ewm-ops` boot DSL + the JSON protocol;
LuaJIT+FFI before Terra; Terra only if runtime numeric codegen becomes a
real need and the toolchain cost is accepted. README layout updated with
TRAINER.md and PODMAN.md.

## Session 2026-09-18 (cont.) — trainer package extracted

`trainer/` reference package implemented at repo root (numpy-only core;
torch only inside `LlmAdapter`):

- `protocol.py` — ProbeConfig / TrajectoryRecord / Prediction / Selection.
- `predictors.py` — dft_period + persistence / linear / dft-periodic / ridge
  portfolio behind one `predict(history)` interface.
- `selector.py` — EwmaSelector (EWMA surprise + ε-greedy).
- `adapters.py` — Adapter / SyntheticAdapter / LlmAdapter, memory_tokens,
  build_prompt (memory in token materialized restored order).
- `ewm.py` — EwmScene client, the only Rust-facing module.
- `loop.py` — run_open_loop / run_closed_loop.
- `smoke_test.py` — full pipeline against real ewm-scene with the synthetic
  adapter (no torch): passes, period 8 detected, closed loop runs, meta mean
  surprise 0.0250 vs ridge 0.0214 (best single).
- README layout + docs/TRAINER.md §4 updated.

## Session 2026-09-18 (cont.) — learned selector (notebook 12)

- `trainer/selector.py` gained `LearnedSelector`: per-predictor linear
  scoring over features `[bss, Δbss, DFT period, EWMA surprises, phase,
  bias]`, ridge-fit online to reward = −surprise; `selector_features` added;
  `run_closed_loop` now passes features to any selector (EWMA ignores them).
- `dft_period` hardened: harmonics can dominate the fundamental, so the
  period is now the gcd of the significant bins (≥5% of total power), not
  the strongest bin. Real-LLM open loop with the `Query:` prefix now
  correctly detects period 8 (was mis-detecting 4).
- `notebooks/12_multi_llm_sidecar_learned_selector.ipynb` runs the real LLMs
  through the `trainer/` package: same open loop, then EWMA vs learned
  selectors each driving a 16-step closed loop. EWMA meta/best 0.952,
  learned 1.109 — the learned selector needs a longer horizon. Smoke test
  updated and passes.
- Next: longer closed loops; surprise-driven updates to the predictors
  themselves.

## Session 2026-09-20 — Jev as router (notebook 13)

TypeSafe AI System One (Jev) integrated as the first non-LLM actor:

- `typesafe-sdk 0.7.0` installed into `ewm-nanolm`; API shape confirmed:
  `TypeSafeClient.system_one(state, questions)` with `Choice` → `.choice`
  / `.probabilities` / `.confidence`, `Score` → `.score`, `Noul` → `.noul`.
- `trainer/protocol.py` gained `DecisionRecord` (typed decision shape) and
  `ProbeConfig.route` (routing mode: only the chosen LLM answers).
- `trainer/adapters.py` gained `JevAdapter` — real SDK when
  `TYPESAFE_API_KEY` is set, deterministic mock otherwise; `LlmAdapter` and
  `SyntheticAdapter` respect `cfg.route`.
- `trainer/loop.py` gained `run_jev_loop`: each step sends query + memory +
  BSS to Jev, routes to the chosen LLM, logs the decision; smoke test
  extended and passes (mock router).
- `notebooks/13_multi_llm_sidecar_jev_router.ipynb` executed: open loop
  (period 8, IICA true), then 16-step Jev router (mock; decisions 6/6/4,
  mean confidence 0.423). Set `TYPESAFE_API_KEY` to run the real model.
- Next: ingest Jev's decision record as a first-class probe so the decision
  is part of S(t), and use `Score`/`Noul` for confidence-gated actions.

## Session 2026-09-20 (cont.) — Jev waitlisted; patterns built with the mock (notebook 14)

Live TypeSafe key is waitlisted, so the integration continues with the mock:

- `trainer/adapters.py` gained `DecisionRouter` ABC (any decision provider)
  and `decision_tokens` (lowers a DecisionRecord into `jev_*` tokens).
- `trainer/loop.py` `run_jev_loop` gained `confidence_floor`, `fallback`,
  `ingest_decision`, and `JevLoopResult.gated_log`; parent dirs are created.
- `notebooks/14_multi_llm_sidecar_jev_patterns.ipynb` executed:
  loop A (routing only, 11/16 gated) vs loop B (+ decision-in-state, 13/16
  gated); loop B's union frames carry `jev_choice_*` / `jev_conf_*` /
  `jev_p_*` tokens, so the decision is part of S(t) and memory.
- Usability verdict: Jev's typed interface is the best external fit for the
  trainer layer; postpone live calls until the key arrives, keep building
  with the mock. Next: live-Jev benchmark, `Score`/`Noul` per-action gates.

## Session 2026-09-20 (cont.) — invariant verified

`git status -- crates/ Cargo.toml Cargo.lock` is empty: notebooks 08–14 and
the whole `trainer/` package were built without modifying any ewm-sm Rust
crate. The only ewm-sm surfaces used are the `ewm-scene` JSON protocol and
the `ewm-ops` boot DSL. Recorded in docs/TRAINER.md §1.

## Session 2026-09-20 (cont.) — heterogeneous side-car (notebook 15)

Realistic three-front-end scenario, capture-first:

- `captures/capture_ocr.py` (deepseek-ocr env): real DeepSeek-OCR vision
  encoder on `test_ocr.png` via `eval_mode=True` → 15 token ids, 24
  cumulative frames.
- `captures/capture_vla.py` (ewm-vla venv): real Cosmos-Policy action chunk
  (16×7, task "pick up the black bowl …") quantized by `ActionQuantizer`
  → 16 frames, padded to 24.
- `captures/capture_jepa.py` (ewm-jepa env): V-JEPA ViT-L/16 target
  encodings (1568×1024) quantized by `Quantizer` (4096 anchors) → 24
  cumulative frames.
- `trainer/loop.py` gained `run_open_loop_from_streams` (capture-first);
  `run_open_loop` refactored over it; smoke test still passes.
- `notebooks/15_hetero_sidecar_ocr_vla_jepa.ipynb` (python3 kernel, no GPU):
  IICA true, union pop 80 → 672, `(Frontend, Ring)` pattern matrix, Noether
  D/R/N, materialized memory 254 tokens restored.
- Encoding-agnostic union `S(t) = S_ocr ∪ S_vla ∪ S_jepa` verified across
  three modalities with zero ewm-sm changes. Next: close the loop across
  the heterogeneous front-ends (memory + curiosity + front-end selection).

## Session 2026-09-21 — Rust Laya as the router (notebook 16)

Article *"Introducing System One Models & Laya (and Jev)"* → open-source
alternative to the waitlisted TypeSafe API:

- **Laya**: Apache-2.0, ungated, 421M, non-autoregressive typed decisions
  (`choice`/`score`/`noul`, calibrated probabilities, one forward pass).
- **Rust implementation**: `laya` crate (pure Rust on candle). Cloned
  `/home/alexmy/tools/laya-rust`, built `laya` CLI + added a new
  `laya-jsonl` daemon bin (persistent stdin/stdout, model loaded once).
- Checkpoint `convaiinnovations/laya/typed-decisions` downloaded to
  `/home/alexmy/.cache/laya/typed-decisions` (808 MB).
- `trainer/adapters.py` gained `LayaAdapter` behind the same
  `DecisionRouter` seam as `JevAdapter`; `run_jev_loop` now takes
  `DecisionRouter`; smoke test extended (real Laya, CPU, ~600 ms/decision
  steady state) and passes.
- `notebooks/16_multi_llm_sidecar_rust_laya.ipynb` executed: real LLMs on
  GPU + Rust Laya on CPU; confidence gating + decision-in-state with real
  calibrated probabilities. On the synthetic routing queries Laya's
  confidences are near-uniform (~0.008), so the gate routed all 16 steps to
  the fallback — the gate working as designed; next step is better routing
  prompts (prose + phrase options) and CUDA inference.

## Session 2026-09-21 (cont.) — Laya prompt fix, CUDA diagnosis, heterogeneous router (notebooks 17–18)

- Prompt A/B on the eight queries: article-style phrasing (answer-type
  options, query-first prose) roughly doubles mean confidence (0.004 →
  0.010) and rebalances choices (old: 7×llm_c; new: 8×llm_a) — but
  confidences stay low because the generic queries genuinely don't
  discriminate. The gate is the correct answer.
- `LayaAdapter._prose_state` + `run_jev_loop` defaults updated to the
  article style; `run_jev_loop_from_streams` added (router over
  pre-captured streams) and exported.
- CUDA build: `laya-jsonl --features cuda` builds with
  `CUDA_COMPUTE_CAP=86` (WMMA kernels needed it), but nvcc 12.8 emits PTX
  ISA 8.7 while driver 565.77 supports ≤ 12.7 → `CUDA_ERROR_UNSUPPORTED_PTX_VERSION`.
  Diagnosed precisely; CPU daemon (~600 ms) remains the working path until
  a driver update or nvcc ≤ 12.7.
- `notebooks/17_..._laya_prompts.ipynb` executed: A/B + improved router
  loop on real LLMs (choices 7/6/3, all gated to fallback).
- `notebooks/18_hetero_sidecar_rust_laya_router.ipynb` executed (python3
  kernel, CPU): Laya routes the notebook-15 front-ends — ocr 9 / vla 2 /
  jepa 13, mean confidence 0.103, sensible query→front-end matches (camera
  → jepa, document → ocr), gating + decision-in-state intact.

## Session 2026-09-22 — Bonsai (Ternary 27B) behind the lattice (notebook 19)

PrismML `Bonsai-demo` cloned to `/home/alexmy/tools/Bonsai-demo`;
`BONSAI_OPENWEBUI=0 BONSAI_CODE_INTERPRETER=0 ./setup.sh` fetched the
PrismML llama.cpp CUDA binaries + `Ternary-Bonsai-2-27B-PQ2_0.gguf`
(7.2 GB) + vision projector.

- GPU quirk: this llama fork lists the **RTX 3060 as CUDA0** and the Quadro
  M1200 as CUDA1 (opposite of nvidia-smi), so `CUDA_VISIBLE_DEVICES=1`
  selects the M1200 and OOMs. Run the server **without** the override; the
  3060 holds the model (5.7 GB, ~8 tok/s).
- `trainer/adapters.py` gained `BonsaiAdapter` (chat + /tokenize +
  /detokenize over the llama.cpp server) and it is exported.
- `notebooks/19_bonsai_sidecar_context.ipynb` executed (python3 kernel):
  open loop captures answer+reasoning tid streams (IICA true, 6×2 BSS),
  then a closed loop where the **lattice-materialized memory** (restored
  order, deduplicated) is detokenized and prefixed to the next query —
  Bonsai never sees raw chat history, only the lattice's memory. This is
  the context-management fit: Bonsai owns token-level context (262K, KV
  cache, prompt cache); ewm-sm decides what that context is.
- Also: `~/.bashrc` now exports `CUDA_VISIBLE_DEVICES=0` — the RTX 3060 is
  the only CUDA-visible GPU for torch and llama.cpp alike (the M1200 stays
  bound for display).

## Session 2026-09-22 (cont.) — compressed Bonsai context (notebook 20)

- `trainer/adapters.py` gained `displacement_tokens(seen, tids)` — the
  D-part of the Noether decomposition at token granularity (only tids new
  to the lattice); `BonsaiAdapter` gained `chat_full` (returns usage, so we
  can read Bonsai's own `prompt_tokens`).
- `notebooks/20_bonsai_compressed_context.ipynb` executed (python3 kernel):
  baseline full-materialized-memory loop vs displacement-filtered loop over
  the same 6 queries. Total prompt tokens 487 → 478 (~1.8% fewer on this
  short, low-redundancy conversation); the win scales with redundancy.
  Answers stay on-topic where the model answers (a few turns are consumed
  by reasoning on both sides).
- Next: context-aware filter (D + Noether R/N-surprising tids) with a
  Laya-decided cap per turn; test on a genuinely long repetitive
  conversation.

## Session 2026-09-22 (cont.) — the collaborative model: Bonsai as the base model

Bonsai is the first model where we can read *and* write its token space
(`/tokenize` + `/detokenize`), so the context it sees is a **proposal** that
ewm-sm recomputes every turn.

- Architecture written: `docs/BONSAI_COLLAB.md` — the collaboration loop
  (query → lattice state → context proposal → Bonsai → new lattice state),
  proposal modes (`full` / `compact` / `surprise`), component table, CLI
  surface, roadmap. ewm-sm and the PrismML llama.cpp fork stay unchanged.
- Implemented `trainer/bonsai_cli.py`: `BonsaiSession` (state tracker +
  context proposer + Bonsai) and the `ewm-bonsai` REPL
  (`/context full|compact`, `/cap N`, `/state`, `/history`, `/policy`,
  `/reset`, `/quit`).
- Smoke-tested end-to-end against the running server: `/context compact`,
  two queries, `/state` (S(t) pop 21→78, content key, memory counts) and
  `/history` all work; per-turn `prompt_tokens` reported.
- Next: `surprise` proposal mode (D + Noether R/N tids); persistent
  addressable sessions via ewm-git; Laya decides the proposal mode/cap per
  turn.

## Session 2026-09-22 (cont.) — bonsai-ewm isolated + live TypeSafe Jev

- `bonsai-ewm` production project scaffolded next to ewm-state-machine
  (stdlib-only package, REPL, podman Containerfile, ARCHITECTURE +
  SESSION_HANDOFF docs).
- **The TypeSafe API key has arrived** (`bonsai-ewm/docs/notes/api-keys.txt`,
  gitignored). `~/.bashrc` now exports `TYPESAFE_API_KEY` from that file.
  Live verification: model `jev-1.13.0`, ~0.5–0.7 s/decision, calibrated
  probabilities. ewm-state-machine notebooks 13/14 now use real Jev
  automatically when run.
- `bonsai-ewm` gained a stdlib-only `JevAdapter` (REST `/v1/systemone`)
  and `/context auto`: Jev chooses full/compact + cap per turn from
  trajectory features (verified: compact/conf 0.27, keep_short 0.84 → cap
  8). Mock fallback when the key is missing.
