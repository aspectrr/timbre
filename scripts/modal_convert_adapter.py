#!/usr/bin/env python3
"""Convert the LoRA adapter to GGUF format for Ollama import.

Ollama's safetensors adapter import is finicky in 0.32.0, but GGUF adapter
import works reliably. This uses llama.cpp's convert_lora_to_gguf.py.

Usage: uv run modal run scripts/modal_convert_adapter.py
"""
from __future__ import annotations

import modal

image = (
    modal.Image.from_registry("python:3.11-slim")
    .apt_install("git", "build-essential", "cmake")
    .pip_install("gguf", "sentencepiece", "protobuf", "safetensors",
                 "transformers", "torch", "accelerate")
    .run_commands(
        "git clone --branch b4990 --depth 1 "
        "https://github.com/ggml-org/llama.cpp /root/llama.cpp",
    )
)

app = modal.App("convert-adapter")
outputs = modal.Volume.from_name("fine-tune-outputs", create_if_missing=True)
hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)


@app.function(image=image, timeout=20 * 60,
              volumes={"/root/.cache/huggingface": hf_cache,
                       "/root/outputs": outputs})
def convert() -> dict:
    import os, subprocess

    adapter_dir = "/root/outputs/adapter"
    out_gguf = "/root/outputs/adapter.gguf"

    # convert_lora_to_gguf.py needs a local base model dir for config + tokenizer.
    # The merged_clean dir has both (config.json, tokenizer.json, etc.).
    base_path = "/root/outputs/merged_clean"
    print(f"Base path: {base_path}")

    print("Converting LoRA adapter to GGUF...")
    subprocess.run([
        "python", "/root/llama.cpp/convert_lora_to_gguf.py",
        adapter_dir, "--outfile", out_gguf, "--base", base_path,
    ], check=True)

    size_mb = os.path.getsize(out_gguf) / 1e6
    print(f"  adapter.gguf: {size_mb:.0f} MB")

    outputs.commit()
    return {"ok": True, "path": out_gguf, "size_mb": size_mb}


@app.local_entrypoint()
def main() -> None:
    print("\n=== Convert LoRA adapter → GGUF ===\n")
    result = convert.remote()
    print(f"\nResult: {result}")
    if result.get("ok"):
        print(f"\n✓ {result['size_mb']:.0f} MB adapter GGUF")
        print(f"Download: uv run modal volume get fine-tune-outputs"
              f" /adapter.gguf data/models/")
