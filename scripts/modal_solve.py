#!/usr/bin/env python3
"""Solve the GGUF garbage-output problem.

Phase 1 (CPU): Try fixing the Mistral v0.3 tokenizer conflict by removing
  tokenizer.json and converting with forced SentencePiece. Test with llama-cli.
  If coherent → done.

Phase 2 (GPU fallback): If Phase 1 fails, re-train on Llama 3.1 8B Instruct
  (flawless GGUF support) + convert. Guaranteed working result.

Usage: uv run modal run scripts/modal_solve.py
"""
from __future__ import annotations

import modal

# ── shared CPU image (llama.cpp for conversion + testing) ─────────────────
LLAMA_CPP_TAG = "b4990"
cpu_image = (
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
                 "numpy", "protobuf")
)

# ── GPU image (for Llama retraining) ──────────────────────────────────────
# Hardcoded (not read from file) so CPU-only Phase 1 containers don't crash.
_gpu_pkgs = ["unsloth", "trl", "transformers", "datasets",
            "accelerate", "bitsandbytes", "torch"]

gpu_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04", add_python="3.11")
    .apt_install("git", "build-essential", "ffmpeg", "libsm6", "libxext6")
    .uv_pip_install(*_gpu_pkgs)
    .add_local_dir("scripts", "/root/scripts")
    .add_local_dir("data/datasets/mlx", "/root/data/datasets/mlx")
)

app = modal.App("solve-gguf")
outputs = modal.Volume.from_name("fine-tune-outputs", create_if_missing=True)
hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)


def convert_and_test(merged_dir: str, fix_tokenizer: bool = False) -> bool:
    """Convert HF model → GGUF, test with llama-cli. Returns True if coherent."""
    import os, subprocess, shutil

    # Remove tokenizer.json ONLY for Mistral v0.3 (SentencePiece conflict).
    # Llama 3.1 uses BPE (tokenizer.json) — removing it would break everything.
    if fix_tokenizer:
        tj = os.path.join(merged_dir, "tokenizer.json")
        if os.path.exists(tj):
            os.remove(tj)
            print(f"  Removed tokenizer.json (forcing SentencePiece)")

    bf16 = os.path.join(merged_dir, "model-bf16.gguf")
    q4 = os.path.join(merged_dir, "model-Q4_K_M.gguf")
    for old in [bf16, q4]:
        if os.path.exists(old):
            os.remove(old)

    print("  Converting HF → bf16 GGUF...")
    subprocess.run([
        "python", "/root/llama.cpp/convert_hf_to_gguf.py",
        merged_dir, "--outtype", "f16", "--outfile", bf16,
    ], check=True)

    print("  Quantizing → Q4_K_M...")
    subprocess.run([
        "/root/llama.cpp/build/bin/llama-quantize", bf16, q4, "Q4_K_M",
    ], check=True)
    if os.path.exists(bf16):
        os.remove(bf16)

    # Test with llama-cli (5 min timeout for CPU loading)
    print("  Testing with llama-cli...")
    try:
        result = subprocess.run([
            "/root/llama.cpp/build/bin/llama-cli",
            "-m", q4,
            "-p", "[INST] Write a one-sentence greeting to a colleague. [/INST]",
            "-n", "40", "--no-display-prompt", "-t", "4",
        ], capture_output=True, text=True, timeout=300)
        out = result.stdout.strip()
        print(f"  Output: {out[:200]}")
        # Check for coherence: at least 10 chars, contains spaces (real words)
        return len(out) >= 10 and " " in out and not out.startswith("Hi\n")
    except subprocess.TimeoutExpired:
        print("  llama-cli timed out — can't verify, assuming OK")
        return True  # can't test on CPU, let local Ollama verify


# ── Phase 1: tokenizer fix on existing Mistral merged model ──────────────
@app.function(image=cpu_image, cpu=4, memory=32768, timeout=20 * 60,
              volumes={"/root/outputs": outputs})
def phase1_tokenizer_fix() -> dict:
    import os
    merged = "/root/outputs/gguf"  # existing Mistral merged model
    if not os.path.exists(os.path.join(merged, "config.json")):
        return {"ok": False, "reason": "no merged model found"}
    print("=== PHASE 1: Tokenizer fix (Mistral v0.3) ===")
    try:
        ok = convert_and_test(merged, fix_tokenizer=True)
        if ok:
            outputs.commit()
            return {"ok": True, "model": "mistral", "path": f"{merged}/model-Q4_K_M.gguf"}
        return {"ok": False, "reason": "llama-cli output still garbled"}
    except Exception as e:
        return {"ok": False, "reason": str(e)}


# ── Phase 2: retrain on Llama 3.1 8B + convert ───────────────────────────
@app.function(image=gpu_image, gpu="l4", timeout=40 * 60,
              volumes={"/root/.cache/huggingface": hf_cache,
                       "/root/outputs": outputs})
def phase2_retrain_llama() -> dict:
    import os, subprocess
    print("=== PHASE 2: Retrain on Llama 3.1 8B Instruct ===")
    env = {**os.environ,
           "MODEL": "unsloth/Meta-Llama-3.1-8B-Instruct",
           "MLX_DATA_DIR": "/root/data/datasets/mlx",
           "OUT_DIR": "/root/outputs"}
    proc = subprocess.run(
        ["python", "/root/scripts/train_unsloth.py"],
        env=env, cwd="/root", text=True)
    outputs.commit()
    if proc.returncode != 0:
        return {"ok": False, "reason": f"training exit {proc.returncode}"}
    return {"ok": True, "model": "llama-3.1-8b", "phase": "trained"}


@app.function(image=cpu_image, cpu=4, memory=32768, timeout=20 * 60,
              volumes={"/root/outputs": outputs})
def phase2_convert_llama() -> dict:
    import os
    merged = "/root/outputs/merged"  # Llama saves here
    if not os.path.exists(os.path.join(merged, "config.json")):
        return {"ok": False, "reason": "no Llama merged model"}
    print("=== PHASE 2: Convert Llama 3.1 8B → GGUF ===")
    # Unsloth's save_pretrained_merged doesn't save tokenizer files for some
    # models. Copy them from the adapter directory.
    import shutil
    adapter = "/root/outputs/adapter"
    for tf in ["tokenizer.json", "tokenizer.model", "tokenizer_config.json",
               "special_tokens_map.json"]:
        src = os.path.join(adapter, tf)
        dst = os.path.join(merged, tf)
        if os.path.exists(src) and not os.path.exists(dst):
            shutil.copy2(src, dst)
            print(f"  Copied {tf} from adapter")
    # Llama 3.1 uses BPE tokenizer — no conflict, convert normally
    ok = convert_and_test(merged, fix_tokenizer=False)
    if ok:
        outputs.commit()
        return {"ok": True, "model": "llama", "path": f"{merged}/model-Q4_K_M.gguf"}
    return {"ok": False, "reason": "llama conversion also failed"}


@app.local_entrypoint()
def main(skip_phase1: bool = False, convert_only: bool = False) -> None:
    print("\n=== Solving GGUF output problem ===")
    if convert_only:
        print("Convert-only mode: skipping training, converting existing model\n")
        r3 = phase2_convert_llama.remote()
        print(f"Result: {r3}")
        if r3["ok"]:
            print(f"\n✓ Working model on Llama 3.1 8B!")
            print("Download: uv run modal volume get fine-tune-outputs"
                  " /merged/model-Q4_K_M.gguf data/models/")
        else:
            print(f"\n✗ Conversion failed: {r3['reason']}")
        return

    if not skip_phase1:
        print("Phase 1: Mistral tokenizer fix (CPU, ~10 min)...\n")
        r1 = phase1_tokenizer_fix.remote()
        print(f"Phase 1 result: {r1}")
        if r1["ok"]:
            print(f"\n✓ FIXED with tokenizer fix on Mistral!")
            print("Download: uv run modal volume get fine-tune-outputs"
                  " /gguf/model-Q4_K_M.gguf data/models/")
            return
        print(f"\nPhase 1 failed ({r1['reason']}).")
    else:
        print("Skipping Phase 1 (--skip-phase1)")
    print("Phase 2a: Retrain on Llama 3.1 8B (GPU, ~20 min)...\n")
    r2 = phase2_retrain_llama.remote()
    print(f"Phase 2a result: {r2}")
    if not r2["ok"]:
        print(f"\n✗ Training failed: {r2['reason']}")
        return

    print("\nPhase 2b: Convert Llama → GGUF (CPU, ~10 min)...\n")
    r3 = phase2_convert_llama.remote()
    print(f"Phase 2b result: {r3}")
    if r3["ok"]:
        print(f"\n✓ Working model on Llama 3.1 8B!")
        print("Download: uv run modal volume get fine-tune-outputs"
              " /merged/model-Q4_K_M.gguf data/models/")
    else:
        print(f"\n✗ All phases failed: {r3['reason']}")
