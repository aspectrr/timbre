#!/usr/bin/env python3
"""Proper merge + GGUF export.

Root cause of garbage output: Unsloth's save_pretrained_merged dequantizes
the 4-bit base model before merging the LoRA adapter, producing subtly
corrupted weights (out-of-range token IDs at inference).

Fix: load the base model in FULL 16-bit (no 4-bit), apply the LoRA adapter
via PEFT, merge, save, then convert to GGUF.

Usage: uv run modal run scripts/modal_proper_merge.py
"""
from __future__ import annotations

import modal

# GPU image: Unsloth for adapter loading + llama.cpp for GGUF conversion
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
    .add_local_dir("data/datasets/mlx", "/root/data/datasets/mlx")
)

app = modal.App("proper-merge")
outputs = modal.Volume.from_name("fine-tune-outputs", create_if_missing=True)
hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)


@app.function(image=image, gpu="l4", timeout=30 * 60,
              volumes={"/root/.cache/huggingface": hf_cache,
                       "/root/outputs": outputs})
def merge_and_export() -> dict:
    import os, json, subprocess, torch
    from unsloth import FastLanguageModel

    adapter_dir = "/root/outputs/adapter"
    with open(os.path.join(adapter_dir, "adapter_config.json")) as f:
        cfg = json.load(f)
    base_model = cfg["base_model_name_or_path"]
    print(f"Base model: {base_model}")

    # 1. Load base+adapter via Unsloth in FULL 16-bit (clean merge, no 4-bit dequant)
    # Loading the adapter path directly lets Unsloth handle its own adapter format.
    print("Loading base + adapter in bf16 (load_in_4bit=False)...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=adapter_dir,
        max_seq_length=2048,
        dtype=torch.bfloat16,
        load_in_4bit=False,  # KEY: full precision merge
    )
    print(f"  Vocab size: {model.config.vocab_size}")

    # 2. Merge + save via Unsloth (handles its own adapter format correctly)
    merged_dir = "/root/outputs/merged_clean"
    print("Merging + saving...")
    model.save_pretrained_merged(merged_dir, tokenizer, save_method="merged_16bit")
    print(f"  Saved to {merged_dir}")

    # 4. Copy tokenizer files (save_pretrained_merged doesn't always save them)
    import shutil
    for tf in ["tokenizer.json", "tokenizer.model", "tokenizer_config.json",
               "special_tokens_map.json"]:
        src = os.path.join(adapter_dir, tf)
        dst = os.path.join(merged_dir, tf)
        if os.path.exists(src) and not os.path.exists(dst):
            shutil.copy2(src, dst)
            print(f"  Copied {tf}")

    # 5. Convert to GGUF
    print("Converting to GGUF...")
    bf16 = os.path.join(merged_dir, "model-bf16.gguf")
    q4 = os.path.join(merged_dir, "model-Q4_K_M.gguf")
    subprocess.run([
        "python", "/root/llama.cpp/convert_hf_to_gguf.py",
        merged_dir, "--outtype", "f16", "--outfile", bf16,
    ], check=True)
    subprocess.run([
        "/root/llama.cpp/build/bin/llama-quantize", bf16, q4, "Q4_K_M",
    ], check=True)
    os.remove(bf16)
    print(f"  Q4_K_M: {os.path.getsize(q4)/1e9:.1f} GB")

    outputs.commit()
    return {"ok": True, "path": q4,
            "size_gb": os.path.getsize(q4)/1e9}


@app.local_entrypoint()
def main() -> None:
    print("\n=== Proper merge (16-bit base + LoRA) + GGUF export ===\n")
    result = merge_and_export.remote()
    print(f"\nResult: {result}")
    if result.get("ok"):
        print(f"\n✓ Working model!")
        print(f"Test output: {result['test_output']}")
        print(f"\nDownload: uv run modal volume get fine-tune-outputs"
              f" /merged_clean/model-Q4_K_M.gguf data/models/")
    else:
        print(f"\n✗ Failed: {result.get('reason')}")
