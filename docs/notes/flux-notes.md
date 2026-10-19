# Flux notes

## Repo assessment

aarambh-vision-studio is a real Rust workspace — 24 crates on Candle 0.11, MMDiT + rectified flow matching, built from
scratch (tokenizer, text encoder, VAE, transformer blocks, training, inference, serve). Dual-licensed MIT OR Apache-2.0,
source-only (no checkpoints), and self-labeled pre-implementation with a Phases 0–27 roadmap.

Key fact for us: our side-car needs a host that emits tensors, not trained weights — so the pre-implementation state is
fine. The hook point is obvious and clean: the inference runtime's sampler loop (aarambh-vision-inference), where S(t) =
latent state at denoising step t.

## Recommendation: `ewm-flux-host` here, port + PR later

Your lean is the right one, for three reasons:

1. Tight R&D loop. Our tested crates (ewm-boolring, ewm-scene) live here; the integration loop (cargo test, notebooks
05+) stays fast, with no coupling to their review cycle.
2. Their API is pre-implementation and will churn. Keep the host-specific surface minimal and stable behind an adapter;
port when aarambh-vision-inference stabilizes. Contributing early PRs into a moving target would be wasted effort.
3. License alignment holds both ways. Ours is MIT; theirs is MIT OR Apache-2.0. An adapter crate contributed to their
workspace under MIT (or dual) drops in cleanly.

## How `ewm-flux-host` consumes the host in the meantime

The cleanest sequence:

- First milestone — no dependency on their code at all. Define a small SidecarProbe trait (per-step state → our HLLSet),
implement ewm-flux-host against a synthetic/random-weight MMDiT shim in our workspace, and close the whole loop:
per-step S(t), D/R/N, ring, warnings, two-score eval. The side-car integration is host-agnostic by design; the only
host-specific part is reading a tensor at each step.
- Then wire to the real host. Either vendoring a shallow copy (path deps) or consuming their crates once buildable; at
that point the port is mostly moving the adapter crate into their workspace.
- PR path. Contribute aarambh-vision-sidecar (adapter + probe) plus a hook in their sampler loop, with our crates as git
dependencies or published. The PR will be small and reviewable.

The Phase-1 scope mirrors Qwen: close the loop on synthetic latents first (random weights are fine — the side-car
monitors the trajectory, not the image quality), define the two-score evaluation for a generative model (loop accuracy =
latent token restoration; decode quality = latent reconstruction / image metrics once a VAE exists), then port.

## Session

Start a new session — the repo work continues in ewm-state-machine, but the conversation context should be fresh.
Kickoff for the new session:

- Project: /home/alexmy/SGS/SGS_lib/fractal_manifold_gen2/ewm-state-machine. Host: <https://github.com/AarambhDevHub/aarambh-vision-studio> (MIT OR Apache-2.0, Candle 0.11, pre-implementation).
- Decision: develop ewm-flux-host in our workspace first (synthetic MMDiT shim, in-process side-car loop, two-score eval), then port
the adapter crate into aarambh-vision-studio and open a PR. Start by reading docs/BOOLRING.md, docs/SEPARATION.md, docs/ASSIGNMENT_QWENDRIVE.md and notebook 03, then scaffold the crate and close the loop on a synthetic latent trajectory.

## Session 2 — basis frames as interpretations; time travel; commit conditions

Formal write-up: **docs/BASIS_FRAMES.md** (Part 1 — mathematical ground;
Part 2 — implementation aspects).

The ring basis changes as the run grows — unavoidable, since the basis is
the span of the context so far. Design settled this session:

- **Do not store BSS vectors per HLLSet.** A BSS projection is
  sub-millisecond (k roaring intersections, measured ~0.4 ms for k=64 in
  release), so soft keys are recomputed on demand. What is stored is the
  **basis frames themselves** — one `BasisSnapshot { generation, basis,
  cover }` per basis-content change (`ewm-flux-host::sidecar`).
- **The basis frame is the interpretation of the HLLSet.** Projecting any
  HLLSet into any historical frame is time travel: re-reading a state
  through the lens of a past context. The soft key `w_i = |X ∩ B_i| / |B_i|`
  stays well-defined for any X (the basis elements are fixed lattice
  nodes); only the hard key (GF(2) coordinates) is span-membership-only.
- **The spill is the measurable error of time travel.** For a frame with
  cover `∪ B_i`, `spill = |X \ cover|` is the part of X the old
  interpretation cannot see — one difference + popcount once the cover is
  precomputed. `spill ≤ residual`: bits outside the union are necessarily
  outside the span.
- **Basis change is a commit condition.** Each step now carries
  `basis_change` (the push changed the ring basis). A basis change is a
  **structural event** — the context re-indexed itself — and is the second
  commit condition next to `new bits > 0`; the generation stamp + rotation
  count/mass measure its magnitude. (Wiring the [UM] commit rule in
  `ewm-app::run_turn` to also commit on `basis_change` is the natural next
  step.)

Notebook 05 demonstrates both: `soft_final`/`step_final` in one coordinate
system, `spill_final` per step, and the oldest-frame spill of the newest
step (429 bits on the default run) as the concrete time-travel error.

## Phase 2 — real-host assessment & adapter (this session)

Repo cloned to `/home/alexmy/SGS/aarambh-vision-studio`, branch
`sidecar-adapter` (commit `d034373`). Assessment:

- **The workspace builds.** 25 crates; `aarambh-vision-core` carries the
  real typed surface — `ModelConfig` (d_model, layers, heads,
  `latent_channels = 16`, `patch_size = 2`), `DrishtiRequest` (`steps`,
  `solver`, `guidance_scale`, `seed`, `resolution`, per-engine controls),
  `SolverKind` (Euler/Heun/Distilled).
- **The sampler loop is a spec, not code.** `aarambh-vision-inference`,
  `-model`, `-nn` are empty stubs. The architecture doc (Part 2 §13) pins
  the loop: Euler `x_{t-Δt} = x_t − Δt·v_θ(x_t, t)`, Heun optional, CFG
  `v_cfg = v_uncond + g·(v_cond − v_uncond)`, 20–50 steps. Hook point is
  therefore fixed: **once per denoising step, S(t) = latent state x_t**.

Adapter delivered (in their workspace, ready to PR when the sampler lands):

- `crates/aarambh-vision-sidecar` — thin Candle adapter:
  - `FluxSidecar::observe(step, &Tensor)` — the only host-specific code
    ([seq_len, dim] f32 → per-token vectors).
  - `FluxSidecar::finish() -> FluxReport` — calls the new
    `ewm_flux_host::run_states` (real-host path: any latent trajectory).
  - `SamplerHook` trait — the one-method contract the inference crate can
    take without depending on EWM types.
- `ewm-flux-host` gained `run_states(&[LatentState], &mut CodebookProbe)`
  and re-exports `CodebookEncoder`; `run()` now delegates to it.
- Dependency for the PR: `ewm-flux-host = { git =
  "https://github.com/alexmy21/ewm-state-machine.git", branch = "main" }`
  (currently a local path while developing; switch once the ewm crates are
  pushed/published).

Next: when `aarambh-vision-inference` implements §13, add the one-line hook
`sidecar.observe(step, &x_t)` to the sampler loop, switch the path dep to
git, and open the PR.
