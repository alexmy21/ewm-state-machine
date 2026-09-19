# loop.py — the closed loop: open-loop recording + memory/curiosity feedback.
#
# `run_open_loop` records the Pavlov cycle (fixed queries, no feedback) and
# returns the BSS trajectory + the per-LLM frame. `run_closed_loop` then
# closes the loop with the predictor portfolio and the meta-selector.

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .adapters import Adapter, memory_tokens
from .ewm import EwmScene, write_pyramid, write_union
from .predictors import dft_period
from .protocol import ProbeConfig
from .selector import EwmaSelector, LearnedSelector, selector_features


@dataclass
class OpenLoopResult:
    M_ol: np.ndarray
    M_proto: np.ndarray
    frame_path: str
    union_path: str
    pyramid_path: str
    names: list[str]
    same_union: bool
    period: Optional[float]


@dataclass
class LoopResult:
    M_cl: np.ndarray
    query_log: list[int]
    choice_log: list[Optional[str]]
    surprise_meta: list[float]
    surprise_all: dict[str, list[float]]
    ewma_log: dict[str, list[float]]
    counts: dict[str, int]


def run_open_loop(
    adapter: Adapter,
    ewm: EwmScene,
    queries: list[str],
    work_dir: str,
    cycles: int = 3,
    max_new_tokens: int = 48,
) -> OpenLoopResult:
    """Play the query bank `cycles` times with no feedback; record the
    trajectory and the per-LLM frame used as the fixed BSS coordinate system.
    """
    names = list(adapter.names)
    T = cycles * len(queries)
    os.makedirs(work_dir, exist_ok=True)

    streams: dict[str, list[list[str]]] = {n: [] for n in names}
    for t in range(T):
        cfg = ProbeConfig(step=t, query=queries[t % len(queries)], max_new_tokens=max_new_tokens)
        per = adapter.run(cfg)
        for n in names:
            streams[n].append(per[n])

    pyramid_path = f"{work_dir}/open_pyramid.jsonl"
    write_pyramid(
        pyramid_path,
        [{n: streams[n][t] for n in names} for t in range(T)],
    )
    union_path = f"{work_dir}/open_union.jsonl"
    write_union(
        union_path,
        [[tok for n in names for tok in streams[n][t]] for t in range(T)],
    )

    pyr = ewm.pyramid(pyramid_path)
    uni = ewm.ingest(union_path)
    same_union = all(
        uni["frames"][i]["key"] == pyr["frames"][i]["union_key"] for i in range(T)
    )

    # Fixed frame: the ring over the per-LLM HLLSets (cumulative vocabulary).
    cum = {n: sorted({tok for t in range(T) for tok in streams[n][t]}) for n in names}
    frame_path = f"{work_dir}/frame_llms.json"
    with open(frame_path, "w") as fh:
        json.dump({"dimensions": [{"name": n, "tokens": cum[n]} for n in names]}, fh)

    proj = ewm.project(union_path, frame_path)
    M_ol = np.array([f["bss"] for f in proj["frames"]])
    M_proto = M_ol[:len(queries)]

    return OpenLoopResult(
        M_ol=M_ol,
        M_proto=M_proto,
        frame_path=frame_path,
        union_path=union_path,
        pyramid_path=pyramid_path,
        names=names,
        same_union=same_union,
        period=dft_period(M_ol),
    )


def run_closed_loop(
    adapter: Adapter,
    ewm: EwmScene,
    open_result: OpenLoopResult,
    portfolio,
    selector: EwmaSelector | LearnedSelector,
    queries: list[str],
    work_dir: str,
    T: int = 16,
    max_new_tokens: int = 48,
) -> LoopResult:
    """Close the loop: memory (materialized union state, restored order) +
    curiosity (selected prediction -> farthest Pavlov prototype) for T steps.
    """
    names = open_result.names
    union_path = f"{work_dir}/closed_union.jsonl"
    open(union_path, "w").close()

    hist = open_result.M_ol.copy()
    M_cl_rows: list[np.ndarray] = []
    query_log: list[int] = []
    choice_log: list[Optional[str]] = []
    surprise_meta: list[float] = []
    surprise_all: dict[str, list[float]] = {name: [] for name in portfolio}
    ewma_log: dict[str, list[float]] = {name: [] for name in portfolio}

    mat_ol = ewm.materialize(open_result.union_path)
    mem = memory_tokens(mat_ol["frames"][-1]["ordered"])

    for t in range(T):
        if t == 0:
            chosen: Optional[str] = None
            preds = None
            q = 0
        else:
            preds = {name: p.predict(hist) for name, p in portfolio.items()}
            feat = selector_features(hist, selector.ewma, period=dft_period(hist))
            chosen = selector.choose(feat)
            pred_sel = preds[chosen]
            candidates = [j for j in range(len(queries)) if j != query_log[-1]]
            q = max(
                candidates,
                key=lambda j: float(np.linalg.norm(pred_sel - open_result.M_proto[j])),
            )
        query_log.append(q)
        choice_log.append(chosen)

        cfg = ProbeConfig(
            step=t, query=queries[q], memory=list(mem), max_new_tokens=max_new_tokens
        )
        per = adapter.run(cfg)
        toks = [x for n in names for x in per[n]]
        with open(union_path, "a") as fh:
            fh.write(json.dumps({"id": t + 1, "tokens": toks}) + "\n")

        proj = ewm.project(union_path, open_result.frame_path)
        m_t = np.array(proj["frames"][-1]["bss"])
        M_cl_rows.append(m_t)

        if t > 0:
            surprises: dict[str, float] = {}
            for name in portfolio:
                s = float(np.linalg.norm(preds[name] - m_t))
                surprises[name] = s
                surprise_all[name].append(s)
            feat = selector_features(hist, selector.ewma, period=dft_period(hist))
            selector.update(surprises, feat)
            surprise_meta.append(float(np.linalg.norm(preds[chosen] - m_t)))
        else:
            for name in portfolio:
                surprise_all[name].append(0.0)
            surprise_meta.append(0.0)

        for name in portfolio:
            ewma_log[name].append(selector.ewma[name])

        hist = np.vstack([hist, m_t])

        mat_cl = ewm.materialize(union_path)
        mem = memory_tokens(mat_cl["frames"][-1]["ordered"])

    return LoopResult(
        M_cl=np.array(M_cl_rows),
        query_log=query_log,
        choice_log=choice_log,
        surprise_meta=surprise_meta,
        surprise_all=surprise_all,
        ewma_log=ewma_log,
        counts=dict(selector.counts),
    )
