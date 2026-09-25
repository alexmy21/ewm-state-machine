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

from .adapters import Adapter, DecisionRouter, decision_tokens, memory_tokens
from .ewm import EwmScene, write_pyramid, write_union
from .predictors import dft_period
from .protocol import DecisionRecord, ProbeConfig
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

    streams: dict[str, list[list[str]]] = {n: [] for n in names}
    for t in range(T):
        cfg = ProbeConfig(step=t, query=queries[t % len(queries)], max_new_tokens=max_new_tokens)
        per = adapter.run(cfg)
        for n in names:
            streams[n].append(per[n])

    return run_open_loop_from_streams(ewm, streams, work_dir, proto_len=len(queries))


def run_open_loop_from_streams(
    ewm: EwmScene,
    streams: dict[str, list[list[str]]],
    work_dir: str,
    proto_len: Optional[int] = None,
) -> OpenLoopResult:
    """Open loop over pre-captured streams (capture-first front-ends): write
    the pyramid + union, run the ewm-sm pipeline, and project the union onto
    the per-front-end cumulative frame."""
    names = list(streams)
    T = min(len(streams[n]) for n in names)
    os.makedirs(work_dir, exist_ok=True)

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

    # Fixed frame: the ring over the per-front-end HLLSets (cumulative vocabulary).
    cum = {n: sorted({tok for t in range(T) for tok in streams[n][t]}) for n in names}
    frame_path = f"{work_dir}/frame_frontends.json"
    with open(frame_path, "w") as fh:
        json.dump({"dimensions": [{"name": n, "tokens": cum[n]} for n in names]}, fh)

    proj = ewm.project(union_path, frame_path)
    M_ol = np.array([f["bss"] for f in proj["frames"]])

    return OpenLoopResult(
        M_ol=M_ol,
        M_proto=M_ol[:proto_len] if proto_len else M_ol,
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


@dataclass
class JevLoopResult:
    M_jev: np.ndarray
    query_log: list[int]
    decision_log: list[DecisionRecord]
    union_path: str
    gated_log: list[bool] = field(default_factory=list)
    # (state, options, instructions, decision) per step — the distillation
    # dataset for the typed-head router (behavioral cloning target).
    routing_log: list[dict] = field(default_factory=list)


def _structural_state(
    ewm: EwmScene,
    union_path: str,
    bss_history: Optional[list] = None,
    bss_names: Optional[list] = None,
) -> dict:
    """Latest Noether D/R/N + Boolean-ring reading from the current union file
    (all frames up to the decision point), plus a **multi-window moving-average
    view** of the structural series and **crossover events** (short window
    crossing long window from above/below) — the "last 2-3-4-5 messages"
    temporal context for the router. Best-effort: an apparatus failure must
    never break the loop."""
    out: dict = {}
    n_out = None
    s_out = None
    try:
        n_out = ewm.noether(union_path)
        if n_out.get("dp"):
            out["drn"] = {
                "dp": n_out["dp"][-1],
                "rp": n_out["rp"][-1],
                "np": n_out["np"][-1],
            }
            if n_out.get("ind1") and n_out.get("ind3"):
                out["drn"]["ind1"] = n_out["ind1"][-1]
                out["drn"]["ind3"] = n_out["ind3"][-1]
    except Exception:
        pass
    try:
        s_out = ewm.sidecar(union_path)
        frames = s_out.get("frames") or []
        if frames:
            f = frames[-1]
            out["ring"] = {
                "residual": f.get("residual", 0),
                "in_span": bool(f.get("in_span", False)),
                "dim": f.get("dim", 0),
                "rotation_count": f.get("rotation_count", 0),
                "spill_first": f.get("spill_first", 0),
            }
    except Exception:
        pass

    # ---- multi-window moving averages + crossings over the full series ----
    try:
        series: dict[str, list[float]] = {}
        if n_out:
            for key in ("np", "rp", "dp", "ind1", "ind3"):
                if n_out.get(key):
                    series[key] = [float(v) for v in n_out[key]]
        if s_out:
            frames = s_out.get("frames") or []
            for key in ("residual", "dim", "rotation_count", "spill_first"):
                series[key] = [float(f.get(key, 0)) for f in frames]
        if bss_history is not None:
            bss_arr = np.asarray(bss_history, dtype=float)
            if bss_arr.ndim == 2 and bss_arr.shape[0] > 0:
                names = list(bss_names or [f"d{i}" for i in range(bss_arr.shape[1])])
                for i, name in enumerate(names[: bss_arr.shape[1]]):
                    series[f"bss_{name}"] = bss_arr[:, i].tolist()

        def _ma(xs: list[float], k: int) -> float:
            win = xs[-k:] if len(xs) >= k else xs
            return float(np.mean(win)) if win else 0.0

        def _ma_prev(xs: list[float], k: int) -> float:
            win = xs[-k - 1 : -1] if len(xs) >= k + 1 else xs[:-1]
            return float(np.mean(win)) if win else 0.0

        windows: dict[str, float] = {}
        for name, xs in series.items():
            if not xs:
                continue
            for k in (2, 3, 4, 5):
                windows[f"{name}_ma{k}"] = round(_ma(xs, k), 4)
        out["windows"] = windows

        crossings: dict[str, bool] = {}
        for name, xs in series.items():
            if not xs:
                continue
            for ks, kl in ((2, 4), (3, 5)):
                cur = _ma(xs, ks) - _ma(xs, kl)
                prev = _ma_prev(xs, ks) - _ma_prev(xs, kl)
                crossings[f"{name}_x{ks}x{kl}_up"] = bool(prev <= 0 < cur)
                crossings[f"{name}_x{ks}x{kl}_down"] = bool(prev >= 0 > cur)
        out["crossings"] = crossings
    except Exception:
        pass
    return out


def run_jev_loop(
    adapter: Adapter,
    jev: DecisionRouter,
    ewm: EwmScene,
    open_result: OpenLoopResult,
    queries: list[str],
    work_dir: str,
    T: int = 16,
    max_new_tokens: int = 48,
    instructions: Optional[str] = None,
    options: Optional[dict[str, str]] = None,
    confidence_floor: Optional[float] = None,
    fallback: Optional[str] = None,
    ingest_decision: bool = True,
    structural: bool = False,
) -> JevLoopResult:
    """Jev as the router: each step, Jev returns a typed decision about which
    LLM should answer the query; only the chosen LLM generates.

    - `confidence_floor` gates the decision: below the floor, the query is
      routed to `fallback` (Pattern 2 from the Jev guide).
    - `ingest_decision` lowers the DecisionRecord into tokens and appends
      them to the union frame, so the decision itself becomes part of S(t)
      and of the materialized memory.
    """
    names = open_result.names
    options = options or {
        "llm_a": "the query needs a longer, structured, reasoned explanation",
        "llm_b": "the query needs a general knowledge answer or a short factual reply",
        "llm_c": "the query is a trivial lookup needing only a very short completion",
    }
    instructions = instructions or "What kind of answer does this query need?"
    union_path = f"{work_dir}/jev_union.jsonl"
    os.makedirs(os.path.dirname(union_path), exist_ok=True)
    open(union_path, "w").close()

    mat_ol = ewm.materialize(open_result.union_path)
    mem = memory_tokens(mat_ol["frames"][-1]["ordered"])

    M_rows: list[np.ndarray] = []
    query_log: list[int] = []
    decision_log: list[DecisionRecord] = []
    gated_log: list[bool] = []
    routing_log: list[dict] = []

    for t in range(T):
        q = t % len(queries)
        query_log.append(q)
        state = {
            "query": queries[q],
            "memory": " ".join(mem),
            "step": t,
            "bss": ({n: round(float(M_rows[-1][i]), 4) for i, n in enumerate(names)}
                    if M_rows else {}),
        }
        if structural and t > 0:
            state.update(_structural_state(ewm, union_path, bss_history=M_rows, bss_names=names))
        decision = jev.route(
            state, options, instructions,
            extra_noul={"needs_memory": "The query requires context from previous answers"},
        )
        decision_log.append(decision)
        routing_log.append({
            "step": t,
            "state": {k: v for k, v in state.items()},
            "options": dict(options),
            "instructions": instructions,
            "decision": decision.to_dict(),
        })

        gated = bool(confidence_floor is not None and decision.confidence < confidence_floor)
        gated_log.append(gated)
        route_name = decision.decision
        if gated and fallback is not None:
            route_name = fallback

        cfg = ProbeConfig(
            step=t,
            query=queries[q],
            memory=list(mem),
            max_new_tokens=max_new_tokens,
            route=route_name,
        )
        per = adapter.run(cfg)
        toks = [x for n in names for x in per[n]]
        if ingest_decision:
            toks = toks + decision_tokens(decision)
        with open(union_path, "a") as fh:
            fh.write(json.dumps({"id": t + 1, "tokens": toks}) + "\n")

        proj = ewm.project(union_path, open_result.frame_path)
        m_t = np.array(proj["frames"][-1]["bss"])
        M_rows.append(m_t)

        mat = ewm.materialize(union_path)
        mem = memory_tokens(mat["frames"][-1]["ordered"])

        if t % 4 == 0 or t == T - 1:
            print(f"t={t:2d} q={q}  route={route_name:6s}  "
                  f"confidence={decision.confidence:.3f}  gated={gated}  mock={decision.mock}")

    return JevLoopResult(
        M_jev=np.array(M_rows),
        query_log=query_log,
        decision_log=decision_log,
        union_path=union_path,
        gated_log=gated_log,
        routing_log=routing_log,
    )


def run_jev_loop_from_streams(
    ewm: EwmScene,
    streams: dict[str, list[list[str]]],
    router: DecisionRouter,
    frame_path: str,
    queries: list[str],
    work_dir: str,
    T: int = 24,
    instructions: Optional[str] = None,
    options: Optional[dict[str, str]] = None,
    confidence_floor: Optional[float] = None,
    fallback: Optional[str] = None,
    ingest_decision: bool = True,
    structural: bool = False,
) -> JevLoopResult:
    """Router loop over pre-captured front-end streams (notebook 15 style):
    each step, the router decides which front-end should handle the query;
    that front-end's next captured frame joins the union S(t)."""
    names = list(streams)
    options = options or {
        "ocr": "the query asks to read or verify text from a page or document",
        "vla": "the query asks which physical action the robot should take next",
        "jepa": "the query asks what happens next in a video or scene",
    }
    instructions = instructions or "Which front-end should handle this query?"
    union_path = f"{work_dir}/router_union.jsonl"
    os.makedirs(os.path.dirname(union_path), exist_ok=True)
    open(union_path, "w").close()

    mem: list[str] = []
    M_rows: list[np.ndarray] = []
    query_log: list[int] = []
    decision_log: list[DecisionRecord] = []
    gated_log: list[bool] = []
    routing_log: list[dict] = []

    for t in range(T):
        q = t % len(queries)
        query_log.append(q)
        state = {
            "query": queries[q],
            "memory": " ".join(mem),
            "step": t,
            "bss": ({n: round(float(M_rows[-1][i]), 4) for i, n in enumerate(names)}
                    if M_rows else {}),
        }
        if structural and t > 0:
            state.update(_structural_state(ewm, union_path, bss_history=M_rows, bss_names=names))
        decision = router.route(
            state, options, instructions,
            extra_noul={"needs_memory": "The query requires context from previous steps"},
        )
        decision_log.append(decision)
        routing_log.append({
            "step": t,
            "state": {k: v for k, v in state.items()},
            "options": dict(options),
            "instructions": instructions,
            "decision": decision.to_dict(),
        })

        gated = bool(confidence_floor is not None and decision.confidence < confidence_floor)
        gated_log.append(gated)
        route_name = decision.decision
        if gated and fallback is not None:
            route_name = fallback

        toks = list(streams[route_name][t])
        if ingest_decision:
            toks = toks + decision_tokens(decision)
        with open(union_path, "a") as fh:
            fh.write(json.dumps({"id": t + 1, "tokens": toks}) + "\n")

        proj = ewm.project(union_path, frame_path)
        m_t = np.array(proj["frames"][-1]["bss"])
        M_rows.append(m_t)

        mat = ewm.materialize(union_path)
        mem = memory_tokens(mat["frames"][-1]["ordered"])

        if t % 4 == 0 or t == T - 1:
            print(f"t={t:2d} q={q}  route={route_name:6s}  "
                  f"confidence={decision.confidence:.3f}  gated={gated}  mock={decision.mock}")

    return JevLoopResult(
        M_jev=np.array(M_rows),
        query_log=query_log,
        decision_log=decision_log,
        union_path=union_path,
        gated_log=gated_log,
        routing_log=routing_log,
    )
