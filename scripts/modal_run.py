#!/usr/bin/env python3
"""Modal runner: trains the Ministral 8B style LoRA in the cloud.

What this does:
  1. Builds a CUDA image with Unsloth installed.
  2. Mounts the synthesized dataset + train script into the container.
  3. Runs scripts/train_unsloth.py on an L4 GPU (~$0.80/hr).
  4. Writes the GGUF to a persistent Modal Volume so we can fetch it back.

Usage (after `modal setup`):
  modal run scripts/modal_run.py

Then download the artifact:
  modal volume get fine-tune-outputs /outputs .
"""
from __future__ import annotations

from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parent.parent

# Single source of truth for the cloud training stack: read the requirements
# file so this image and the repo never drift apart. CUDA-only, never local.
_reqs_file = ROOT / "requirements-unsloth.txt"
CLOUD_PACKAGES = [
    line.strip()
    for line in _reqs_file.read_text().splitlines()
    if line.strip() and not line.strip().startswith("#")
]

# ── image: CUDA base + Unsloth stack (installed via uv) ───────────────────
image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04",
        add_python="3.11",
    )
    .apt_install(
        "git", "build-essential", "ffmpeg", "libsm6", "libxext6",
        # Unsloth's GGUF export calls llama.cpp, which needs these to build.
        # Pre-installing avoids an interactive prompt that crashes headless containers.
        "cmake", "curl", "libcurl4-openssl-dev",
    )
    .uv_pip_install(*CLOUD_PACKAGES)
    .add_local_dir("scripts", "/root/scripts")
    .add_local_dir("data/datasets/mlx", "/root/data/datasets/mlx")
)

app = modal.App("fine-tune-style-clone", image=image)

# Persistent volume: HF cache (avoid re-downloading the 8B model) + outputs.
hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)
outputs = modal.Volume.from_name("fine-tune-outputs", create_if_missing=True)

GPU = "l4"          # 24GB — comfortable for 8B QLoRA + headroom
TIMEOUT = 60 * 60   # 1 hour ceiling


@app.function(
    gpu=GPU,
    timeout=TIMEOUT,
    volumes={
        "/root/.cache/huggingface": hf_cache,
        "/root/outputs": outputs,
    },
)
def train() -> str:
    import subprocess
    env = {
        "MODEL": "mistralai/Mistral-7B-Instruct-v0.3",
        "MLX_DATA_DIR": "/root/data/datasets/mlx",
        "OUT_DIR": "/root/outputs",
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
    }
    # Run the single source-of-truth training script.
    # Run from /root so Unsloth can find a llama.cpp clone there if needed.
    proc = subprocess.run(
        ["python", "/root/scripts/train_unsloth.py"],
        env={**__import__("os").environ, **env},
        cwd="/root",
        text=True,
    )
    # ALWAYS commit — even on partial failure — so a successful training
    # run is never lost to a downstream export error.
    import os as _os
    saved = _os.path.exists("/root/outputs/adapter") or _os.path.exists("/root/outputs/gguf")
    outputs.commit()
    if proc.returncode != 0:
        if saved:
            # Training succeeded but GGUF export failed — report partial.
            return "partial: adapter saved, gguf export failed"
        raise RuntimeError(f"training failed with exit {proc.returncode}")
    return "ok"


@app.local_entrypoint()
def main() -> None:
    print("\n=== launching cloud training (L4 GPU, Ministral 8B QLoRA) ===")
    print("=== logs stream below; ~15-25 min expected ===\n")
    train.remote()
    print("\n=== training finished ===")
    print("GGUF is on the 'fine-tune-outputs' volume.")
    print("Fetch it with:  modal volume get fine-tune-outputs /outputs .")
