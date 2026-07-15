#!/usr/bin/env python3
"""Convert-only: clear merged_clean, reload adapter in 16-bit, merge, GGUF.

The adapter is already trained and saved on the Modal volume. This script
just does the clean 16-bit merge + GGUF conversion, after clearing any
leftover safetensors from prior runs.

Usage: uv run modal run scripts/modal_convert_clean.py
"""
from __future__ import annotations

import modal

_gpu_pkgs = ["unsloth", "peft", "transformers", "torch", "accelerate",
             "datasets", "trl", "bitsandbytes"]

image = (
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

app = modal.App("convert-clean")
outputs = modal.Volume.from_name("fine-tune-outputs", create_if_missing=True)
hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)


@app.function(image=image, gpu="l4", timeout=30 * 60,
              volumes={"/root/.cache/huggingface": hf_cache,
                       "/root/outputs": outputs})
def convert() -> dict:
    import os, subprocess, torch, shutil
    from unsloth import FastLanguageModel

    adapter_dir = "/root/outputs/adapter"
    merged_dir = "/root/outputs/merged_clean"

    # 1. Clear any leftover files from prior runs (prevents duplicate-tensor error)
    if os.path.exists(merged_dir):
        shutil.rmtree(merged_dir)
        print(f"Cleared {merged_dir}")

    # 2. Reload base+adapter in FULL 16-bit (clean merge, no 4-bit corruption)
    print("Loading base + adapter in bf16...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=adapter_dir, max_seq_length=1024,
        dtype=torch.bfloat16, load_in_4bit=False)

    # 3. Merge + save
    print("Merging + saving (16-bit)...")
    model.save_pretrained_merged(merged_dir, tokenizer,
                                 save_method="merged_16bit")

    # 4. Copy tokenizer files (save_pretrained_merged sometimes omits them)
    for tf in ["tokenizer.json", "tokenizer.model", "tokenizer_config.json",
               "special_tokens_map.json"]:
        src, dst = os.path.join(adapter_dir, tf), os.path.join(merged_dir, tf)
        if os.path.exists(src) and not os.path.exists(dst):
            shutil.copy2(src, dst)
            print(f"  Copied {tf}")

    # 5. Convert to GGUF
    print("Converting to GGUF...")
    bf16 = os.path.join(merged_dir, "model-bf16.gguf")
    q4 = os.path.join(merged_dir, "model-Q4_K_M.gguf")
    subprocess.run(["python", "/root/llama.cpp/convert_hf_to_gguf.py",
                    merged_dir, "--outtype", "f16", "--outfile", bf16], check=True)
    subprocess.run(["/root/llama.cpp/build/bin/llama-quantize", bf16, q4,
                    "Q4_K_M"], check=True)
    os.remove(bf16)
    size_gb = os.path.getsize(q4) / 1e9
    print(f"  Q4_K_M: {size_gb:.2f} GB")

    outputs.commit()
    return {"ok": True, "path": q4, "size_gb": size_gb}


@app.local_entrypoint()
def main() -> None:
    print("\n=== Clean 16-bit merge + GGUF (from saved adapter) ===\n")
    result = convert.remote()
    print(f"\nResult: {result}")
    if result.get("ok"):
        print(f"\n✓ Done! {result['size_gb']:.2f} GB GGUF")
        print(f"Download: uv run modal volume get fine-tune-outputs"
              f" /merged_clean/model-Q4_K_M.gguf data/models/")
