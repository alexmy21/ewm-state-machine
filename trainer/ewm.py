# ewm.py — the ewm-scene client and JSONL writers.
#
# This is the only place that talks to the Rust apparatus. Everything else in
# the trainer layer is pure Python over the JSON this client returns.

from __future__ import annotations

import json
import os
import subprocess
from typing import Optional

DEFAULT_BIN = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "target",
    "debug",
    "ewm-scene",
)


class EwmScene:
    """Thin subprocess client for the ewm-scene CLI."""

    def __init__(self, bin_path: Optional[str] = None, timeout: int = 600):
        self.bin = bin_path or DEFAULT_BIN
        self.timeout = timeout

    def __call__(self, *args) -> dict:
        r = subprocess.run(
            [self.bin, *args], capture_output=True, text=True, timeout=self.timeout
        )
        if r.returncode != 0:
            raise RuntimeError(f"ewm-scene failed: {r.stderr[-800:]}")
        return json.loads(r.stdout.strip())

    def pyramid(self, path: str) -> dict:
        return self("pyramid", path)

    def ingest(self, path: str) -> dict:
        return self("ingest", path)

    def materialize(self, path: str) -> dict:
        return self("materialize", path)

    def noether(self, path: str) -> dict:
        return self("noether", path)

    def project(self, path: str, frame_path: str) -> dict:
        return self("project", path, "--frame", frame_path)

    def sidecar(self, path: str, freeze: Optional[int] = None) -> dict:
        args = ["sidecar", path]
        if freeze is not None:
            args += ["--freeze", str(freeze)]
        return self(*args)


def write_frames(path: str, token_lists: list[list[str]], start_id: int = 1) -> str:
    """Write flat frames `{"id": i, "tokens": [...]}`."""
    with open(path, "w") as fh:
        for i, tokens in enumerate(token_lists, start_id):
            fh.write(json.dumps({"id": i, "tokens": tokens}) + "\n")
    return path


def write_pyramid(path: str, perceptron_lists: list[dict[str, list[str]]], start_id: int = 1) -> str:
    """Write pyramid frames `{"id": i, "perceptrons": {name: [...]}}`."""
    with open(path, "w") as fh:
        for i, perceptrons in enumerate(perceptron_lists, start_id):
            fh.write(json.dumps({"id": i, "perceptrons": perceptrons}) + "\n")
    return path


def write_union(path: str, token_lists: list[list[str]], start_id: int = 1) -> str:
    """Write flat frames with an explicit name `union` — identical shape to
    `write_frames`, kept as a named alias for readability."""
    return write_frames(path, token_lists, start_id=start_id)
