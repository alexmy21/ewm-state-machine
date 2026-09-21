#!/usr/bin/env python3
"""Capture V-JEPA latent predictions as a standard tid{n} JSONL stream.

Run in the `ewm-jepa` conda env:

    CUDA_VISIBLE_DEVICES=1 /home/alexmy/.conda/envs/ewm-jepa/bin/python \
        captures/capture_jepa.py

Produces /home/alexmy/.cache/ewm-hetero/capture_jepa.jsonl with T=24
cumulative frames of quantized target encodings (the same frozen cosine
codebook as ewm-jepa notebook 01).
"""

import json
import os
import sys

PROJECT_ROOT = "/home/alexmy/SGS/SGS_lib/fractal_manifold/ewm-jepa"
JEPA_ROOT = os.path.join(PROJECT_ROOT, "jepa")
CHECKPOINT = os.path.join(PROJECT_ROOT, "checkpoints", "vitl16.pth.tar")

sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, JEPA_ROOT)

OUT_DIR = "/home/alexmy/.cache/ewm-hetero"
OUT_PATH = os.path.join(OUT_DIR, "capture_jepa.jsonl")
T = 24
CODEBOOK = 4096

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

os.makedirs(OUT_DIR, exist_ok=True)

import torch  # noqa: E402

from app.vjepa.utils import init_video_model  # noqa: E402
from src.masks.multiblock3d import MaskCollator  # noqa: E402
from src.masks.utils import apply_masks  # noqa: E402
from ewm_jepa import Quantizer  # noqa: E402

MASK_CONFIGS = [
    {"aspect_ratio": (0.75, 1.5), "num_blocks": 8,
     "spatial_scale": (0.15, 0.15), "temporal_scale": (1.0, 1.0),
     "max_temporal_keep": 1.0, "max_keep": None},
    {"aspect_ratio": (0.75, 1.5), "num_blocks": 2,
     "spatial_scale": (0.7, 0.7), "temporal_scale": (1.0, 1.0),
     "max_temporal_keep": 1.0, "max_keep": None},
]

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("device:", device)

print("[1/4] building V-JEPA ViT-L/16 ...")
encoder, predictor = init_video_model(
    device=device, patch_size=16, num_frames=16, tubelet_size=2,
    model_name="vit_large", crop_size=224, pred_depth=12, pred_embed_dim=384,
    uniform_power=True, use_mask_tokens=True, num_mask_tokens=2,
    zero_init_mask_tokens=True, use_sdpa=True,
)
encoder.eval()
predictor.eval()

print("[2/4] loading checkpoint ...")
ckpt = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
strip = lambda sd: ({k[len("module."):]: v for k, v in sd.items()}
                    if any(k.startswith("module.") for k in sd) else sd)
encoder.load_state_dict(strip(ckpt["encoder"]))
predictor.load_state_dict(strip(ckpt["predictor"]))
del ckpt

print("[3/4] forward pass (random 16-frame clip) ...")
collator = MaskCollator(cfgs_mask=MASK_CONFIGS, crop_size=224,
                        num_frames=16, patch_size=16, tubelet_size=2)
torch.manual_seed(0)
sample = torch.randn(3, 16, 224, 224)
video, masks_enc, masks_pred = collator([sample])
video = video.to(device)
masks_enc = [m.to(device) for m in masks_enc]
masks_pred = [m.to(device) for m in masks_pred]
with torch.no_grad():
    h_full = encoder(video)
print("target encodings h_full:", tuple(h_full.shape))

print("[4/4] quantizing target encodings ...")
quantizer = Quantizer(dim=1024, codebook_size=CODEBOOK, seed=0, device=device)
ids = quantizer.encode(h_full[0]).cpu().tolist()
print("quantized ids:", len(ids), "unique:", len(set(ids)))

step = max(1, len(ids) // T)
frames = []
for t in range(T):
    fr_ids = ids[: (t + 1) * step]
    frames.append({"id": t + 1, "tokens": [f"tid{i}" for i in fr_ids]})

with open(OUT_PATH, "w", encoding="utf-8") as fh:
    for fr in frames:
        fh.write(json.dumps(fr) + "\n")

with open(os.path.join(OUT_DIR, "capture_jepa.meta.json"), "w") as fh:
    json.dump({
        "frontend": "v-jepa",
        "model": "vitl16",
        "T": T,
        "quantized_ids": len(ids),
        "codebook": CODEBOOK,
        "latent_shape": list(h_full.shape),
    }, fh, indent=2)

print("wrote", OUT_PATH, "frames:", len(frames),
      "tokens/frame:", [len(f["tokens"]) for f in frames[:3]], "...")
