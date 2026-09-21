#!/usr/bin/env python3
"""Capture NVIDIA VLA (Cosmos-Policy) actions as a standard tid{n} JSONL.

Run in the `ewm-vla` venv:

    /home/alexmy/SGS/SGS_lib/fractal_manifold/ewm-vla/repo/.venv/bin/python3 \
        captures/capture_vla.py

Produces /home/alexmy/.cache/ewm-hetero/capture_vla.jsonl with T=16
cumulative frames over the real captured action chunk (quantized with the
same frozen action codebook as ewm-vla notebook 01).
"""

import json
import os
import sys

ROOT = "/home/alexmy/SGS/SGS_lib/fractal_manifold/ewm-vla"
sys.path.insert(0, ROOT)

OUT_DIR = "/home/alexmy/.cache/ewm-hetero"
OUT_PATH = os.path.join(OUT_DIR, "capture_vla.jsonl")
T = 16

os.makedirs(OUT_DIR, exist_ok=True)

import numpy as np  # noqa: E402

from ewm_vla import ActionQuantizer  # noqa: E402
from ewm_vla.pipeline import load_capture  # noqa: E402

cap = load_capture()
print("capture:", "REAL VLA" if not cap["synthetic"] else "SYNTHETIC")
print("task    :", cap["task"])
print("actions :", cap["actions"].shape, "| value:", cap["value_prediction"])
print("proprio :", cap["proprio"].shape)

q_a = ActionQuantizer(dim=7, codebook_size=512, seed=0)
a_ids = q_a.encode(cap["actions"]).tolist()
print("action tids:", len(a_ids), "unique:", len(set(a_ids)))

q_p = ActionQuantizer(dim=9, codebook_size=128, seed=0)
p_id = int(q_p.encode(np.asarray(cap["proprio"], dtype=np.float32).reshape(1, -1))[0])
q_v = ActionQuantizer(dim=1, codebook_size=16, seed=0)
v_id = int(q_v.encode(np.asarray([cap["value_prediction"]], dtype=np.float32).reshape(1, 1))[0])

frames = []
for t in range(T):
    ids = a_ids[: t + 1] + [p_id, v_id]
    frames.append({"id": t + 1, "tokens": [f"tid{i}" for i in ids]})

with open(OUT_PATH, "w", encoding="utf-8") as fh:
    for fr in frames:
        fh.write(json.dumps(fr) + "\n")

with open(os.path.join(OUT_DIR, "capture_vla.meta.json"), "w") as fh:
    json.dump({
        "frontend": "nvidia-vla-cosmos-policy",
        "task": str(cap["task"]),
        "T": T,
        "action_tids": len(a_ids),
        "action_codebook": 512,
        "proprio_codebook": 128,
        "value_codebook": 16,
    }, fh, indent=2)

print("wrote", OUT_PATH, "frames:", len(frames),
      "tokens/frame:", [len(f["tokens"]) for f in frames[:3]], "...")
