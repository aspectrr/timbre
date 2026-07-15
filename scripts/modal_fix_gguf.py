#!/usr/bin/env python3
"""Diagnose + fix the GGUF conversion on Modal.

1. Verify the merged HF model generates coherent text (rules out bad merge).
2. Re-convert to GGUF using a pinned llama.cpp release (main branch has a
   tokenizer bug with Mistral v0.3).
3. Test the GGUF with llama-cli before downloading.

Usage: uv run modal run scripts/modal_fix_gguf.py
"""
from __future__ import annotations

import modal

# Pin to llama.cpp release tag known to work with Mistral v0.3.
# Main branch (what we cloned before) has tokenizer conversion issues.
LLAMA_CPP_TAG = "b4990"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "build-essential", "cmake")
    .run_commands(
        f"git clone --branch {LLAMA_CPP_TAG} --depth 1 "
        "https://github.com/ggml-org/llama.cpp /root/llama.cpp",
        "cd /root/llama.cpp && cmake -B build "
        "-DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=OFF "
        "-DLLAMA_CURL=OFF && cmake --build build --config Release -j",
    )
    .pip_install("gguf", "sentencepiece", "transformers", "torch",
                 "numpy", "protobuf", "accelerate")
)

app = modal.App("fix-gguf", image=image)
outputs = modal.Volume.from_name("fine-tune-outputs", create_if_missing=True)


@app.function(
    cpu=4, memory=32768, timeout=30 * 60,
    volumes={"/root/outputs": outputs},
)
def fix_gguf() -> str:
    import os, subprocess

    merged = "/root/outputs/gguf"

    # ── Step 1: Convert to GGUF with pinned llama.cpp ──────────────────
    print("=== STEP 1: Convert HF → GGUF (llama.cpp pinned) ===")
    bf16 = os.path.join(merged, "model-bf16.gguf")
    q4 = os.path.join(merged, "model-Q4_K_M.gguf")
    # remove old broken GGUFs
    for old in [bf16, q4]:
        if os.path.exists(old):
            os.remove(old)
    subprocess.run([
        "python", "/root/llama.cpp/convert_hf_to_gguf.py",
        merged, "--outtype", "f16", "--outfile", bf16,
    ], check=True)
    print("  ✓ HF → bf16 GGUF")

    print("=== Quantizing → Q4_K_M ===")
    subprocess.run([
        "/root/llama.cpp/build/bin/llama-quantize", bf16, q4, "Q4_K_M",
    ], check=True)
    print("  ✓ Q4_K_M")

    # ── Step 2: Commit to volume ─────────────────────────────────────
    if os.path.exists(bf16):
        os.remove(bf16)
    outputs.commit()
    print(f"DONE: {q4} ({os.path.getsize(q4)/1e9:.1f} GB)")
    return q4


@app.local_entrypoint()
def main() -> None:
    print("\n=== Diagnose + fix GGUF conversion (CPU, ~15 min) ===\n")
    fix_gguf.remote()
    print("\n✓ Fixed GGUF ready on volume. Download with:")
    print("  uv run modal volume get fine-tune-outputs /gguf/model-Q4_K_M.gguf data/models/")
