# adapters.py — the only model-specific code: probe config -> tid streams.
#
# The `Adapter` is the boundary between the trainer and the LLMs. `run(cfg)`
# returns one `tid{n}` token list per LLM for one step. Two implementations:
#
#   SyntheticAdapter — deterministic, no torch; for tests and fast loops.
#   LlmAdapter       — the three real small LLMs (needs the ewm-nanolm env).
#
# Memory is fed back in token materialized format with restored order, exactly
# as in notebooks 10/11.

from __future__ import annotations

import json
import os
import subprocess
import zlib
from abc import ABC, abstractmethod
from typing import Optional

import numpy as np

from .protocol import DecisionRecord, ProbeConfig

# The three real small LLMs (cached on this machine). GPT-Neo-1.3B is omitted:
# its generate path crashes under transformers 5.16.1.
DEFAULT_MODELS = {
    "llm_a": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    "llm_b": "Qwen/Qwen2.5-1.5B-Instruct",
    "llm_c": "gpt2",
}


def memory_tokens(ordered: list[str], cap: int = 24) -> list[str]:
    """Materialized memory: restored order, consecutive duplicates removed,
    capped to the last `cap` tokens."""
    toks: list[str] = []
    for t in ordered:
        if not toks or toks[-1] != t:
            toks.append(t)
    return toks[-cap:]


def displacement_tokens(seen: set, tids: list[str]) -> list[str]:
    """The D-part of the Noether decomposition at token granularity: only the
    tids that are new to the lattice (`t not in seen`) are returned, and they
    are added to `seen`. This is the compressed, information-bearing context —
    repeated content collapses to nothing, exactly like the HLLSet union."""
    out: list[str] = []
    for t in tids:
        if t not in seen:
            out.append(t)
            seen.add(t)
    return out


def build_prompt(cfg: ProbeConfig) -> str:
    """Probe prompt: memory prefix (token format) + the query."""
    if cfg.memory:
        return "memory: " + " ".join(cfg.memory) + "\nQuery: " + cfg.query
    return "Query: " + cfg.query


class Adapter:
    """Base adapter: `run(cfg) -> {llm_name: [tid...]}`."""

    names: tuple[str, ...] = ()

    def run(self, cfg: ProbeConfig) -> dict[str, list[str]]:
        raise NotImplementedError


class SyntheticAdapter(Adapter):
    """Deterministic synthetic probes: enough to exercise the whole loop.

    The streams are period-8 in the query index and echo a few memory tokens,
    so the loop's memory feedback changes the trajectory — a fast stand-in
    for the real LLMs.
    """

    def __init__(self, names=("llm_a", "llm_b", "llm_c"), seed: int = 0, tokens_per_step: int = 24):
        self.names = tuple(names)
        self.seed = seed
        self.tokens_per_step = tokens_per_step

    def run(self, cfg: ProbeConfig) -> dict[str, list[str]]:
        out: dict[str, list[str]] = {}
        half = max(1, self.tokens_per_step // 2)
        quarter = max(0, self.tokens_per_step // 4)
        phase = cfg.step % 8
        # Deterministic across runs (str.hash() is salted per process).
        qhash = zlib.crc32(cfg.query.encode("utf-8")) % 64
        for i, n in enumerate(self.names):
            if cfg.route is not None and n != cfg.route:
                out[n] = []                       # routing mode: only one answers
                continue
            # Phase-only seed: with empty memory the stream is exactly
            # period-8; memory feedback (echo) breaks the periodicity.
            rng = np.random.default_rng(self.seed + phase * 101 + i * 7)
            base = [f"tid{phase * 50 + j}" for j in range(half)]
            q_tokens = [f"tid{2000 + qhash + j}" for j in range(quarter)]
            noise = [f"tid{1000 + int(rng.integers(0, 64))}" for _ in range(quarter)]
            echo = list(cfg.memory[:4])           # memory feedback
            out[n] = base + q_tokens + echo + noise
        return out


class LlmAdapter(Adapter):
    """The three real small LLMs, loaded once on the pinned GPU."""

    def __init__(self, models: Optional[dict[str, str]] = None, max_new_tokens: int = 48):
        # Pin the GPU before torch is imported anywhere else. With
        # CUDA_DEVICE_ORDER=PCI_BUS_ID on this machine the Quadro M1200 is
        # device 0 and the RTX 3060 is device 1. Respect an explicit
        # CUDA_VISIBLE_DEVICES (the notebooks set it in the first cell);
        # default to device 1 (the RTX).
        os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")
        os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.max_new_tokens = max_new_tokens
        model_ids = dict(models or DEFAULT_MODELS)
        self.names = tuple(model_ids)
        self.loaded: dict[str, tuple[object, object]] = {}
        for name, model_id in model_ids.items():
            tok = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
            if tok.pad_token is None:
                tok.pad_token = tok.eos_token
            model = AutoModelForCausalLM.from_pretrained(
                model_id, local_files_only=True, dtype=torch.float16, device_map="auto"
            )
            model.eval()
            self.loaded[name] = (tok, model)

    def run(self, cfg: ProbeConfig) -> dict[str, list[str]]:
        torch = self.torch
        prompt = build_prompt(cfg)
        max_new = cfg.max_new_tokens or self.max_new_tokens
        out: dict[str, list[str]] = {}
        for n in self.names:
            if cfg.route is not None and n != cfg.route:
                out[n] = []                       # routing mode: only one answers
                continue
            tok, model = self.loaded[n]
            inputs = tok(prompt, return_tensors="pt").to(model.device)
            with torch.no_grad():
                gen = model.generate(
                    **inputs,
                    max_new_tokens=max_new,
                    do_sample=False,
                    pad_token_id=tok.eos_token_id,
                )
            new_ids = gen[0, inputs["input_ids"].shape[1]:].cpu().tolist()
            out[n] = [f"tid{i}" for i in new_ids]
        return out


def decision_tokens(record: DecisionRecord) -> list[str]:
    """Lower a typed decision into tid-style tokens, so the decision itself
    can be ingested into S(t) as a first-class probe."""
    toks = [f"jev_choice_{record.decision}"]
    toks.append(f"jev_conf_{int(round(record.confidence * 10))}")
    for name, p in record.probabilities.items():
        toks.append(f"jev_p_{name}_{int(round(p * 100))}")
    for key, val in record.aux.items():
        toks.append(f"jev_{key}_{int(round(val * 10))}")
    return toks


class DecisionRouter(ABC):
    """The generic decision-model boundary: any provider that returns a
    typed DecisionRecord (Jev, a local classifier, a rule engine) implements
    this interface."""

    @abstractmethod
    def route(
        self,
        state: dict,
        options: dict[str, str],
        instructions: str,
        extra_noul: Optional[dict[str, str]] = None,
    ) -> DecisionRecord:
        raise NotImplementedError


class JevAdapter(DecisionRouter):
    """TypeSafe AI System One (Jev) as the router.

    Jev returns typed, probabilistic decisions instead of text. This adapter
    wraps the `typesafe-sdk` and exposes one call:

        route(state, options, instructions, extra_noul=None) -> DecisionRecord

    When `TYPESAFE_API_KEY` is missing, it falls back to a deterministic mock
    with the same DecisionRecord shape, so the loop runs end-to-end without
    the API (clearly flagged with `mock=True`).
    """

    def __init__(self, mock: bool = False):
        self.mock = mock or not os.environ.get("TYPESAFE_API_KEY")
        self._client = None
        if not self.mock:
            try:
                from typesafe_sdk import TypeSafeClient

                self._client = TypeSafeClient()
            except Exception as exc:      # SDK missing or key invalid
                self.mock = True
                self._mock_reason = str(exc)
            else:
                self._mock_reason = ""
        else:
            self._mock_reason = "TYPESAFE_API_KEY not set"

    def route(
        self,
        state: dict,
        options: dict[str, str],
        instructions: str,
        extra_noul: Optional[dict[str, str]] = None,
    ) -> DecisionRecord:
        if self.mock:
            return self._mock_route(state, options, instructions, extra_noul)

        from typesafe_sdk import Choice, Noul

        questions: dict = {
            "route": Choice(instructions=instructions, criteria=dict(options))
        }
        if extra_noul:
            for name, noul_instructions in extra_noul.items():
                questions[name] = Noul(instructions=noul_instructions)

        resp = self._client.system_one(state=state, questions=questions)
        ans = resp.answers["route"]
        aux = {name: float(resp.answers[name].noul) for name in (extra_noul or {})}
        usage = resp.usage
        return DecisionRecord(
            decision=ans.choice,
            confidence=float(ans.confidence),
            probabilities={str(k): float(v) for k, v in ans.probabilities.items()},
            decision_type="route_query",
            aux=aux,
            model=resp.model,
            input_tokens=usage.input_tokens or 0,
            output_tokens=usage.output_tokens or 0,
            mock=False,
        )

    def _mock_route(
        self,
        state: dict,
        options: dict[str, str],
        instructions: str,
        extra_noul: Optional[dict[str, str]] = None,
    ) -> DecisionRecord:
        # Deterministic, context-shaped mock: prefer the option whose name is
        # closest to a crc32 hash of the query + memory, with a softmax-ish
        # distribution so confidence and probabilities are realistic.
        query = str(state.get("query", ""))
        memory = str(state.get("memory", ""))
        seed = zlib.crc32((query + memory).encode("utf-8"))
        names = list(options)
        rng = np.random.default_rng(seed)
        logits = rng.uniform(0.0, 1.0, size=len(names))
        probs = np.exp(logits) / np.exp(logits).sum()
        probs = {n: float(p) for n, p in zip(names, probs)}
        choice = max(probs, key=probs.get)
        aux = {name: 0.5 for name in (extra_noul or {})}
        return DecisionRecord(
            decision=choice,
            confidence=float(probs[choice]),
            probabilities=probs,
            decision_type="route_query",
            aux=aux,
            model="mock-jev",
            input_tokens=0,
            output_tokens=0,
            mock=True,
        )


class LayaAdapter(DecisionRouter):
    """Rust Laya (the open-source System One decision model) as the router.

    Laya is the Apache-2.0, ungated sibling of TypeSafe's Jev: a
    non-autoregressive ModernBERT-large encoder + RL decision head that
    answers typed questions (choice / score / noul) with calibrated
    probabilities in one forward pass. This adapter talks to a persistent
    `laya-jsonl` daemon (pure Rust on candle — no Python, no torch), so the
    checkpoint is loaded once and every `route()` is a single batched pass.

    Falls back to the deterministic mock when the binary or checkpoint is
    missing.
    """

    def __init__(
        self,
        bin_path: Optional[str] = None,
        model_dir: Optional[str] = None,
        device: str = "cpu",
    ):
        self.bin = bin_path or os.environ.get(
            "LAYA_BIN", "/home/alexmy/tools/laya-rust/target/release/laya-jsonl"
        )
        self.model_dir = model_dir or os.environ.get(
            "LAYA_MODEL", "/home/alexmy/.cache/laya/typed-decisions"
        )
        self.device = device
        self.mock = not (
            os.path.exists(self.bin)
            and os.path.exists(os.path.join(self.model_dir, "model.safetensors"))
        )
        self._mock_reason = (
            ""
            if not self.mock
            else f"laya binary or checkpoint missing ({self.bin}, {self.model_dir})"
        )
        self._proc = None
        self._req_id = 0

    def _ensure_proc(self):
        if self._proc is None or self._proc.poll() is not None:
            self._proc = subprocess.Popen(
                [self.bin, "--model", self.model_dir, "--device", self.device],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )

    def route(
        self,
        state: dict,
        options: dict[str, str],
        instructions: str,
        extra_noul: Optional[dict[str, str]] = None,
    ) -> DecisionRecord:
        if self.mock:
            return self._mock_route(state, options, instructions, extra_noul)

        questions = {
            "route": {
                "type": "choice",
                "instructions": instructions,
                "criteria": dict(options),
            }
        }
        for name, instr in (extra_noul or {}).items():
            questions[name] = {"type": "noul", "instructions": instr}

        self._ensure_proc()
        self._req_id += 1
        payload = {
            "id": self._req_id,
            "state": self._prose_state(state),
            "questions": questions,
        }
        self._proc.stdin.write(json.dumps(payload) + "\n")
        self._proc.stdin.flush()
        line = self._proc.stdout.readline()
        resp = json.loads(line)["response"]
        ans = resp["answers"]["route"]

        probabilities = {str(k): float(v) for k, v in ans.get("probabilities", {}).items()}
        aux = {
            name: float(resp["answers"][name].get("noul", 0.0))
            for name in (extra_noul or {})
        }
        usage = resp.get("usage", {})
        return DecisionRecord(
            decision=str(ans.get("choice", "")),
            confidence=float(ans.get("confidence", 0.0)),
            probabilities=probabilities,
            decision_type="route_query",
            aux=aux,
            model=str(resp.get("model", "laya")),
            input_tokens=int(usage.get("input_tokens", 0)),
            output_tokens=int(usage.get("output_tokens", 0)),
            mock=False,
        )

    @staticmethod
    def _prose_state(state: dict) -> str:
        """Laya rewards prose, not raw structs (see the model card): name the
        situation the way a person would, put the query first, keep the
        opaque tid memory short and framed."""
        parts = []
        query = str(state.get("query", "")).strip()
        memory = str(state.get("memory", "")).strip()
        bss = state.get("bss", {})
        if query:
            parts.append(f"The user asked: \"{query}\".")
        if memory:
            toks = memory.split()
            shown = " ".join(toks[:8])
            parts.append(
                f"The system's memory holds {len(toks)} content-addressed token ids "
                f"from earlier answers; the most recent are: {shown}."
            )
        if bss:
            parts.append(
                "The three models' current coverage of the conversation is: "
                + ", ".join(f"{k} {float(v):.3f}" for k, v in bss.items())
                + "."
            )
        return " ".join(parts)

    def _mock_route(
        self,
        state: dict,
        options: dict[str, str],
        instructions: str,
        extra_noul: Optional[dict[str, str]] = None,
    ) -> DecisionRecord:
        query = str(state.get("query", ""))
        memory = str(state.get("memory", ""))
        seed = zlib.crc32((query + memory).encode("utf-8"))
        names = list(options)
        rng = np.random.default_rng(seed)
        logits = rng.uniform(0.0, 1.0, size=len(names))
        probs = np.exp(logits) / np.exp(logits).sum()
        probs = {n: float(p) for n, p in zip(names, probs)}
        choice = max(probs, key=probs.get)
        aux = {name: 0.5 for name in (extra_noul or {})}
        return DecisionRecord(
            decision=choice,
            confidence=float(probs[choice]),
            probabilities=probs,
            decision_type="route_query",
            aux=aux,
            model="mock-laya",
            input_tokens=0,
            output_tokens=0,
            mock=True,
        )

    def close(self):
        if self._proc is not None and self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=10)
            except Exception:
                self._proc.kill()
        self._proc = None


# ---------------------------------------------------------------------------
# Structural local-LLM router — the prompted Jev/Laya replacement.
# ---------------------------------------------------------------------------

STRUCTURAL_ROUTER_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

STRUCTURAL_ROUTER_SYSTEM = (
    "You are the routing controller of a multi-LLM memory system. "
    "You answer with exactly one JSON object and nothing else."
)


def structural_decision_prompt(
    state: dict,
    options: dict[str, str],
    instructions: str,
    extra_noul: Optional[dict[str, str]] = None,
) -> str:
    """Serialize the ewm-sm structural state into the decision prompt (the
    *user* message; the router wraps it with a system message / chat template).

    The state may carry, beyond query/memory/step/bss:
      - drn  {dp, rp, np, ind1, ind3}          latest Noether transition
      - ring {residual, in_span, dim, rotation_count, spill_first}
    Missing fields are omitted, so the prompt degrades gracefully to the
    query/memory/bss surface the Laya adapter already uses.
    """
    names = list(options)
    lines = []
    query = str(state.get("query", "")).strip()
    lines.append(f'Query: "{query}"')
    memory = str(state.get("memory", "")).strip()
    if memory:
        toks = memory.split()
        lines.append(
            f"Memory: {len(toks)} content-addressed token ids; most recent: "
            + " ".join(toks[:8])
        )
    bss = state.get("bss") or {}
    if bss:
        lines.append(
            "BSS coverage (fraction of the conversation each model's frame covers): "
            + ", ".join(f"{k} {float(v):.3f}" for k, v in bss.items())
        )
    drn = state.get("drn") or {}
    if drn:
        lines.append(
            "Noether D/R/N of the last transition (departed/retained/new bits): "
            + f"dp={drn.get('dp', 0)} rp={drn.get('rp', 0)} np={drn.get('np', 0)}"
        )
        if "ind1" in drn and "ind3" in drn:
            lines.append(f"indicators ind1={drn['ind1']:.3f} ind3={drn['ind3']:.3f}")
    ring = state.get("ring") or {}
    if ring:
        lines.append(
            "Boolean-ring reading of the last frame: "
            + f"residual={ring.get('residual', 0)} "
            + f"in_span={bool(ring.get('in_span', False))} "
            + f"dim={ring.get('dim', 0)} "
            + f"rotation_count={ring.get('rotation_count', 0)} "
            + f"spill_first={ring.get('spill_first', 0)}"
        )
    windows = state.get("windows") or {}
    if windows:
        # compact per-feature multi-window view: name ma2/ma3/ma4/ma5
        by_feat: dict[str, list[str]] = {}
        for key, val in windows.items():
            if key.endswith("_ma2") or key.endswith("_ma3") or key.endswith("_ma4") or key.endswith("_ma5"):
                feat, _, k = key.rpartition("_ma")
                by_feat.setdefault(feat, []).append((int(k), float(val)))
        if by_feat:
            parts = []
            for feat in sorted(by_feat):
                seq = " ".join(f"ma{k}={v:.3f}" for k, v in sorted(by_feat[feat]))
                parts.append(f"{feat} {seq}")
            lines.append(
                "Moving averages of the structural series over the last 2/3/4/5 "
                "steps: " + "; ".join(parts)
            )
    crossings = state.get("crossings") or {}
    if crossings:
        events = sorted(k for k, v in crossings.items() if v)
        lines.append(
            "Moving-average crossovers (short vs long window, from below = up / "
            "from above = down): " + (", ".join(events) if events else "none")
        )
    lines.append("Options:")
    for n in names:
        lines.append(f"- {n}: {options[n]}")
    lines.append(f"Instructions: {instructions}")
    if extra_noul:
        aux_names = ", ".join(f'"{n}"' for n in extra_noul)
        lines.append(
            f"Also answer the numeric questions {aux_names} in the aux object "
            "(each 0.0 to 1.0)."
        )
    probs_schema = ", ".join(f'"{n}": 0.0' for n in names)
    aux_schema = ", ".join(f'"{n}": 0.0' for n in (extra_noul or {}))
    lines.append(
        'JSON schema: {"decision": "<option name>", "confidence": 0.0, '
        f'"probabilities": {{{probs_schema}}}'
        + (f', "aux": {{{aux_schema}}}' if extra_noul else "")
        + "}"
    )
    return "\n".join(lines)


def _extract_json_object(text: str) -> Optional[dict]:
    """Pull the first {...} span out of a model reply (fences, prose, etc.)."""
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        obj = json.loads(text[start : end + 1])
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def _structural_fallback(
    state: dict, options: dict[str, str], extra_noul: Optional[dict[str, str]]
) -> DecisionRecord:
    """Parse-proof fallback: trust the state, not the parser.

    Chooses the option with the highest BSS coverage (the model whose frame
    covers the conversation best), with zero confidence so a configured
    confidence gate can fall back further. Used when the local LM is missing
    or its JSON reply cannot be parsed.
    """
    names = list(options)
    if not names:
        decision = ""
    else:
        bss = state.get("bss") or {}
        try:
            decision = max(names, key=lambda n: float(bss.get(n, 0.0)))
        except Exception:
            decision = names[0]
    probs = {n: (1.0 if n == decision else 0.0) for n in names}
    return DecisionRecord(
        decision=decision,
        confidence=0.0,
        probabilities=probs,
        decision_type="route_query",
        aux={name: 0.5 for name in (extra_noul or {})},
        model="structural-llm-fallback",
        mock=True,
    )


class StructuralLlmRouter(DecisionRouter):
    """A local small LLM prompted with the ewm-sm structural state (BSS,
    D/R/N, Boolean ring) that returns a typed routing decision as JSON.

    Drop-in replacement for Jev/Laya behind `DecisionRouter`::

        router = StructuralLlmRouter()
        result = run_jev_loop(..., jev=router, ..., structural=True)

    When torch/transformers or the checkpoint are unavailable — or
    `mock=True` is passed — it degrades to the deterministic structural
    fallback (`mock=True`), so the loop always runs.
    """

    def __init__(
        self,
        model_id: Optional[str] = None,
        max_new_tokens: int = 192,
        temperature: float = 0.0,
        mock: bool = False,
    ):
        self.model_id = model_id or STRUCTURAL_ROUTER_MODEL
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.mock = mock
        self._mock_reason = ""
        if mock:
            self._mock_reason = "mock=True"
            return
        # Pin the GPU before torch is imported anywhere else. With
        # CUDA_DEVICE_ORDER=PCI_BUS_ID on this machine the Quadro M1200 is
        # device 0 and the RTX 3060 is device 1. Respect an explicit
        # CUDA_VISIBLE_DEVICES; default to device 1 (the RTX).
        os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")
        os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except Exception as exc:
            self.mock = True
            self._mock_reason = f"torch/transformers unavailable: {exc}"
            return
        try:
            self.tok = AutoTokenizer.from_pretrained(self.model_id, local_files_only=True)
            # Force the router onto the selected CUDA device. `device_map="auto"`
            # may otherwise offload it to CPU when the probe LLMs already hold
            # the GPU, which turns every decision into a ~60 s CPU generation.
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id, local_files_only=True, dtype=torch.float16
            ).to("cuda:0")
            self.model.eval()
        except Exception as exc:
            self.mock = True
            self._mock_reason = f"checkpoint unavailable for {self.model_id}: {exc}"
            return
        self._torch = torch
        self._device = torch.device("cuda:0")

    def _generate_json(self, user_prompt: str) -> tuple[str, int, int]:
        tok = self.tok
        messages = [
            {"role": "system", "content": STRUCTURAL_ROUTER_SYSTEM},
            {"role": "user", "content": user_prompt},
        ]
        try:
            text = tok.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            text = STRUCTURAL_ROUTER_SYSTEM + "\n\n" + user_prompt
        inputs = tok(text, return_tensors="pt").to(self._device)
        prompt_tokens = int(inputs["input_ids"].shape[1])
        with self._torch.no_grad():
            gen = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=self.temperature > 0,
                temperature=self.temperature if self.temperature > 0 else 1.0,
                pad_token_id=tok.eos_token_id,
            )
        new_ids = gen[0, inputs["input_ids"].shape[1] :].cpu().tolist()
        reply = tok.decode(new_ids, skip_special_tokens=True)
        return reply, prompt_tokens, len(new_ids)

    def route(
        self,
        state: dict,
        options: dict[str, str],
        instructions: str,
        extra_noul: Optional[dict[str, str]] = None,
    ) -> DecisionRecord:
        names = list(options)
        if self.mock:
            return _structural_fallback(state, options, extra_noul)

        user_prompt = structural_decision_prompt(state, options, instructions, extra_noul)
        reply, prompt_tokens, new_tokens = self._generate_json(user_prompt)
        obj = _extract_json_object(reply)
        if obj is None:
            rec = _structural_fallback(state, options, extra_noul)
            rec.model = "structural-llm-parse-fallback"
            return rec

        probs: dict[str, float] = {}
        raw_probs = obj.get("probabilities") or {}
        if isinstance(raw_probs, dict):
            for n in names:
                try:
                    probs[n] = float(raw_probs.get(n, 0.0))
                except Exception:
                    probs[n] = 0.0
        if not any(v > 0 for v in probs.values()):
            probs = {n: 1.0 / len(names) for n in names} if names else {}
        total = sum(probs.values())
        if total > 0:
            probs = {n: v / total for n, v in probs.items()}
        elif names:
            probs = {n: 1.0 / len(names) for n in names}

        decision = str(obj.get("decision", "")).strip()
        if decision not in names:
            decision = max(probs, key=probs.get) if probs else (names[0] if names else "")
        try:
            confidence = float(obj.get("confidence", probs.get(decision, 0.0)))
        except Exception:
            confidence = probs.get(decision, 0.0)
        confidence = min(1.0, max(0.0, confidence))

        aux = {}
        for name in (extra_noul or {}):
            try:
                v = float((obj.get("aux") or {}).get(name, 0.5))
            except Exception:
                v = 0.5
            aux[name] = min(1.0, max(0.0, v))

        return DecisionRecord(
            decision=decision,
            confidence=confidence,
            probabilities=probs,
            decision_type="route_query",
            aux=aux,
            model=self.model_id,
            input_tokens=prompt_tokens,
            output_tokens=new_tokens,
            mock=False,
        )


class BonsaiAdapter:
    """PrismML Bonsai (ternary 27B) served by the PrismML llama.cpp fork.

    Bonsai is a reasoning model; the server exposes the OpenAI-compatible
    chat API plus llama.cpp's /tokenize and /detokenize endpoints. The
    adapter turns Bonsai's answers into the same `tid{n}` streams the rest
    of the trainer understands — and turns materialized tid memory back into
    text for the next prompt, so ewm-sm can act as Bonsai's context manager.
    """

    def __init__(self, base_url: str = "http://127.0.0.1:8081", timeout: int = 600):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        import urllib.request

        self._urllib = urllib.request

    def _post(self, path: str, payload: dict) -> dict:
        import json as _json

        req = self._urllib.Request(
            self.base_url + path,
            data=_json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        with self._urllib.urlopen(req, timeout=self.timeout) as resp:
            return _json.loads(resp.read().decode("utf-8"))

    def health(self) -> bool:
        try:
            with self._urllib.urlopen(self.base_url + "/health", timeout=10) as resp:
                import json as _json

                return _json.loads(resp.read().decode("utf-8")).get("status") == "ok"
        except Exception:
            return False

    def chat(self, prompt: str, max_tokens: int = 256, temperature: float = 0.0):
        """-> (answer_text, reasoning_text)."""
        full = self.chat_full(prompt, max_tokens=max_tokens, temperature=temperature)
        return full["content"], full["reasoning"]

    def chat_full(self, prompt: str, max_tokens: int = 256, temperature: float = 0.0) -> dict:
        """Full response: content, reasoning and usage (incl. prompt_tokens)."""
        r = self._post(
            "/v1/chat/completions",
            {
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "temperature": temperature,
                "stream": False,
            },
        )
        msg = r["choices"][0]["message"]
        return {
            "content": str(msg.get("content") or ""),
            "reasoning": str(msg.get("reasoning_content") or ""),
            "usage": dict(r.get("usage", {})),
        }

    def tokenize(self, text: str) -> list[int]:
        return list(self._post("/tokenize", {"content": text}).get("tokens", []))

    def detokenize(self, tokens: list[int]) -> str:
        return str(self._post("/detokenize", {"tokens": tokens}).get("content", ""))
