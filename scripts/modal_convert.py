#!/usr/bin/env python3
"""Convert the merged model (already on the volume) to GGUF for Ollama.

Does NOT re-train. Reads the merged 16-bit HuggingFace model from the
fine-tune-outputs volume (/gguf/*.safetensors), builds llama.cpp from
source, and produces a Q4_K_M quantized GGUF.

CPU-only — no GPU needed. Cheap and fast (~10 min).

Usage:
  uv run modal run scripts/modal_convert.py

Then download:
  uv run modal volume get fine-tune-outputs /gguf/<name>.gguf .
"""
from __future__ import annotations

import modal

# CPU image: build llama.cpp from source (Unsloth's bundled build is stale).
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "build-essential", "cmake")
    # Build llama.cpp at IMAGE BUILD TIME so the quantizer binary is ready.
    .run_commands(
        "git clone https://github.com/ggml-org/llama.cpp /root/llama.cpp",
        "cd /root/llama.cpp && cmake -B build "
        "-DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=OFF "
        "-DLLAMA_CURL=OFF && cmake --build build --config Release -j",
    )
    .pip_install("gguf", "sentencepiece", "transformers", "torch", "numpy", "protobuf")
)

app = modal.App("gguf-convert", image=image)

outputs = modal.Volume.from_name("fine-tune-outputs", create_if_missing=True)


@app.function(
    cpu=4,
    memory=16384,
    timeout=30 * 60,
    volumes={"/root/outputs": outputs},
)
def convert() -> str:
    import subprocess
    import os

    merged_dir = "/root/outputs/gguf"
    if not os.path.exists(os.path.join(merged_dir, "config.json")):
        raise RuntimeError(
            f"No merged model at {merged_dir}. Run training first."
        )

    bf16_out = os.path.join(merged_dir, "model-bf16.gguf")
    q4_out = os.path.join(merged_dir, "model-Q4_K_M.gguf")

    # 1. Convert HF safetensors → GGUF bf16 (pure Python via gguf pkg)
    if not os.path.exists(bf16_out):
        print("=== Converting HF → GGUF bf16 ===")
        subprocess.run(
            [
                "python", "/root/llama.cpp/convert_hf_to_gguf.py",
                merged_dir,
                "--outtype", "f16",
                "--outfile", bf16_out,
            ],
            check=True,
        )

    # 2. Quantize bf16 → Q4_K_M (compiled llama-quantize binary)
    if not os.path.exists(q4_out):
        print("=== Quantizing bf16 → Q4_K_M ===")
        subprocess.run(
            [
                "/root/llama.cpp/build/bin/llama-quantize",
                bf16_out, q4_out, "Q4_K_M",
            ],
            check=True,
        )

    # 3. Clean up the bf16 intermediate to save volume space
    if os.path.exists(bf16_out) and os.path.exists(q4_out):
        os.remove(bf16_out)

    outputs.commit()
    size_gb = os.path.getsize(q4_out) / 1e9
    print(f"\n=== DONE: {q4_out} ({size_gb:.1f} GB) ===")
    return q4_out


@app.local_entrypoint()
def main() -> None:
    print("\n=== Converting merged model → Q4_K_M GGUF (CPU only) ===\n")
    path = convert.remote()
    print(f"\nGGUF ready: {path}")
    print("Download with:")
    print("  uv run modal volume get fine-tune-outputs /gguf/model-Q4_K_M.gguf .")
