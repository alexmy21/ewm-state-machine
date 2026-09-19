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

import os
import zlib
from typing import Optional

import numpy as np

from .protocol import ProbeConfig

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
