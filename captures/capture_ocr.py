#!/usr/bin/env python3
"""Capture DeepSeek-OCR encoding IDs as a standard tid{n} JSONL stream.

Run in the `deepseek-ocr` conda env:

    CUDA_VISIBLE_DEVICES=1 /home/alexmy/.conda/envs/deepseek-ocr/bin/python \
        captures/capture_ocr.py

Produces /home/alexmy/.cache/ewm-hetero/capture_ocr.jsonl with T cumulative
frames: frame t contains the first (t+1)*step token ids of the OCR stream.
"""

import json
import os
import sys

OUT_DIR = "/home/alexmy/.cache/ewm-hetero"
OUT_PATH = os.path.join(OUT_DIR, "capture_ocr.jsonl")
MODEL_PATH = ("/home/alexmy/.cache/huggingface/hub/"
              "models--deepseek-ai--DeepSeek-OCR/snapshots/"
              "9f30c71f441d010e5429c532364a86705536c53a")
IMAGE_PATH = "/home/alexmy/SGS/DeepSeek-OCR/data/test_ocr.png"
T = 24

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

os.makedirs(OUT_DIR, exist_ok=True)

from transformers import AutoModel, AutoTokenizer  # noqa: E402

print("loading DeepSeek-OCR tokenizer ...")
ds_tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
print("vocab:", ds_tokenizer.vocab_size)

mode = "tokenizer"
ocr_text = ("The neural network model processes image data for object "
            "detection tasks across computer vision systems. It extracts "
            "hierarchical features and maps them into a latent codebook of "
            "encoding ids that feed the content addressed state machine.")

try:
    import torch  # noqa: E402
    if torch.cuda.is_available():
        print("loading DeepSeek-OCR vision encoder (bf16) ...")
        ds_model = AutoModel.from_pretrained(
            MODEL_PATH, trust_remote_code=True, use_safetensors=True,
            torch_dtype=torch.bfloat16,
        ).eval().cuda()
        print("running vision inference on", IMAGE_PATH)
        recognized = ds_model.infer(
            ds_tokenizer,
            prompt="<image>\nFree OCR.",
            image_file=IMAGE_PATH,
            output_path=os.path.join(OUT_DIR, "ocr_output"),
            base_size=1024, image_size=640, crop_mode=True,
            eval_mode=True,
        )
        if recognized and recognized.strip():
            ocr_text = recognized.strip()
            mode = "vision"
        del ds_model
        torch.cuda.empty_cache()
except Exception as exc:
    print(f"vision path failed ({type(exc).__name__}: {exc}); using tokenizer mode")

print("mode:", mode)
print("ocr text:", ocr_text[:160].replace("\n", " "), "...")

real_ids = ds_tokenizer.encode(ocr_text)
print("token ids:", len(real_ids))

step = max(1, len(real_ids) // T)
frames = []
for t in range(T):
    ids = real_ids[: (t + 1) * step]
    frames.append({"id": t + 1, "tokens": [f"tid{i}" for i in ids]})

with open(OUT_PATH, "w", encoding="utf-8") as fh:
    for fr in frames:
        fh.write(json.dumps(fr) + "\n")

with open(os.path.join(OUT_DIR, "capture_ocr.meta.json"), "w") as fh:
    json.dump({
        "frontend": "deepseek-ocr",
        "mode": mode,
        "model": MODEL_PATH.rsplit("/", 1)[-1],
        "T": T,
        "token_ids": len(real_ids),
        "vocab": ds_tokenizer.vocab_size,
    }, fh, indent=2)

print("wrote", OUT_PATH, "frames:", len(frames),
      "tokens/frame:", [len(f["tokens"]) for f in frames[:3]], "...")
