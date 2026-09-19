#!/usr/bin/env python3
"""Smoke test for the trainer package.

Runs the full open-loop + closed-loop pipeline against the real `ewm-scene`
binary with a deterministic SyntheticAdapter (no torch needed):

    python3 trainer/smoke_test.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trainer import (
    EwmScene,
    EwmaSelector,
    LearnedSelector,
    ProbeConfig,
    SyntheticAdapter,
    default_portfolio,
    dft_period,
    run_closed_loop,
    run_open_loop,
)

QUERIES = [
    "What is the capital of France?",
    "Explain gravity in one sentence.",
    "Write a haiku about rain.",
    "What is the meaning of life?",
    "Name three planets in our solar system.",
    "What is the speed of light?",
    "Define entropy in simple words.",
    "Who wrote Romeo and Juliet?",
]


def main() -> int:
    work_dir = "/tmp/ewm-trainer-smoke"
    os.makedirs(work_dir, exist_ok=True)

    # ── protocol round-trip ───────────────────────────────────────────────
    cfg = ProbeConfig(step=3, query=QUERIES[3], memory=["tid1", "tid2"])
    assert ProbeConfig.from_dict(cfg.to_dict()) == cfg
    print("protocol round-trip: ok")

    # ── predictors on a synthetic period-8 series ─────────────────────────
    t = np.arange(24)
    series = np.column_stack([
        np.sin(2 * np.pi * t / 8.0),
        np.cos(2 * np.pi * t / 8.0),
        0.5 * np.sin(2 * np.pi * t / 8.0 + 1.0),
    ])
    assert abs(dft_period(series) - 8.0) < 1e-6
    portfolio = default_portfolio()
    for name, p in portfolio.items():
        pred = p.predict(series)
        assert pred.shape == (3,), name
    print("predictors: ok (period 8 detected, all shapes correct)")

    # ── full loop with the synthetic adapter + real ewm-scene ─────────────
    adapter = SyntheticAdapter(names=("llm_a", "llm_b", "llm_c"), seed=0)
    ewm = EwmScene()

    open_result = run_open_loop(adapter, ewm, QUERIES, work_dir, cycles=3)
    assert open_result.same_union is True
    assert open_result.M_ol.shape == (24, 3)
    assert open_result.M_proto.shape == (8, 3)
    assert open_result.period is not None and abs(open_result.period - 8.0) < 1e-6
    assert np.allclose(open_result.M_ol[8], open_result.M_ol[0])
    print(f"open loop: ok (IICA {open_result.same_union}, period {open_result.period})")

    selector = EwmaSelector(names=list(portfolio), eps=0.1, decay=0.8, warmup=2, seed=0)
    loop_result = run_closed_loop(
        adapter, ewm, open_result, portfolio, selector, QUERIES, work_dir, T=16
    )
    assert loop_result.M_cl.shape == (16, 3)
    assert len(loop_result.query_log) == 16
    assert loop_result.choice_log[0] is None
    assert sum(loop_result.counts.values()) == 15          # one choice per t > 0
    for name in portfolio:
        assert len(loop_result.surprise_all[name]) == 16
        assert len(loop_result.ewma_log[name]) == 16
    print("closed loop (EWMA selector): ok")
    print("  choices:", loop_result.counts)
    for name in portfolio:
        print(f"    {name:12s} {np.mean(loop_result.surprise_all[name][1:]):.6f}")
    print(f"    {'meta':12s} {np.mean(loop_result.surprise_meta[1:]):.6f}")

    # ── learned selector on the same open loop ─────────────────────────────
    learned = LearnedSelector(names=list(portfolio), eps=0.1, warmup=4, seed=0)
    loop_learned = run_closed_loop(
        adapter, ewm, open_result, portfolio, learned, QUERIES, work_dir, T=16
    )
    assert loop_learned.M_cl.shape == (16, 3)
    assert loop_learned.choice_log[0] is None
    assert sum(loop_learned.counts.values()) == 15
    print("closed loop (learned selector): ok")
    print("  choices:", loop_learned.counts)
    for name in portfolio:
        print(f"    {name:12s} {np.mean(loop_learned.surprise_all[name][1:]):.6f}")
    print(f"    {'meta':12s} {np.mean(loop_learned.surprise_meta[1:]):.6f}")
    print("smoke test passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
