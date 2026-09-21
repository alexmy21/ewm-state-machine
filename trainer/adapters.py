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
        # Pin the GPU before torch is imported anywhere else.
        os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
        os.environ["CUDA_VISIBLE_DEVICES"] = "1"
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
