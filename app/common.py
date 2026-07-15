"""Shared Modal infra: app, images, volumes, Dict, job helpers."""
from __future__ import annotations

import json
import time
import uuid
from pathlib import Path

import modal

app = modal.App("style-clone")

# ── shared storage ─────────────────────────────────────────────────────────
# Dict = low-latency, consistent job status across containers (no commit/reload).
# Volume = large files (uploaded inputs, pairs, adapter GGUF).
DATA_VOL = "/data"
jobs_vol = modal.Volume.from_name("styleclone-data", create_if_missing=True)
hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)
status_store = modal.Dict.from_name("styleclone-status", create_if_missing=True)

# ── images ─────────────────────────────────────────────────────────────────
# Light image: web server + CPU pipeline (ingest, curate, synth via OpenRouter).
web_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("fastapi", "python-multipart")
)

# GPU image: Unsloth QLoRA training + llama.cpp adapter→GGUF export.
_gpu_pkgs = ["unsloth", "peft", "transformers", "torch", "accelerate",
             "datasets", "trl", "bitsandbytes"]
gpu_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04", add_python="3.11")
    .apt_install("git", "build-essential", "cmake", "ffmpeg", "libsm6", "libxext6")
    .uv_pip_install(*_gpu_pkgs)
    .pip_install("gguf", "sentencepiece", "protobuf")
    .run_commands(
        "git clone --branch b4990 --depth 1 "
        "https://github.com/ggml-org/llama.cpp /root/llama.cpp",
        "cd /root/llama.cpp && cmake -B build "
        "-DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=OFF "
        "-DLLAMA_CURL=OFF && cmake --build build --config Release -j",
    )
)

# ── job helpers ────────────────────────────────────────────────────────────
STAGES = ["ingesting", "curating", "synthesizing", "training", "exporting", "done"]
MODEL_LABELS = {
    "llama3.2-3b": "unsloth/Llama-3.2-3B-Instruct",
}


def new_job_id() -> str:
    return uuid.uuid4().hex[:12]


def job_dir(job_id: str) -> Path:
    return Path(DATA_VOL) / "jobs" / job_id


def set_status(job_id: str, **fields) -> None:
    """Update job status in the shared Dict (immediately visible to readers)."""
    cur = status_store.get(job_id) or {}
    cur.update(fields)
    cur["updated_at"] = time.time()
    status_store[job_id] = cur


def get_status(job_id: str) -> dict | None:
    return status_store.get(job_id)
