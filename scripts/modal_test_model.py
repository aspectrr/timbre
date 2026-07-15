#!/usr/bin/env python3
"""Test the Llama 3.1 8B style-clone GGUF on Modal with llama-cli.

Runs 3 prompts to verify the model generates coherent Collin-style text.

Usage: uv run modal run scripts/modal_test_model.py
"""
from __future__ import annotations

import modal

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
)

app = modal.App("test-model")
outputs = modal.Volume.from_name("fine-tune-outputs", create_if_missing=True)

PROMPTS = [
    "Write a short cold email to a startup founder whose product you admire, asking to grab coffee.",
    "Reply to a colleague apologizing for missing a meeting and proposing a time to reschedule.",
    "Write a brief email to a potential client introducing your consulting services.",
]


@app.function(image=image, cpu=8, memory=32768, timeout=20 * 60,
              volumes={"/root/outputs": outputs})
def test_model() -> dict:
    import subprocess, os

    gguf = "/root/outputs/merged/model-Q4_K_M.gguf"
    if not os.path.exists(gguf):
        return {"error": f"GGUF not found at {gguf}"}

    results = {}
    for i, prompt in enumerate(PROMPTS):
        # Llama 3.1 instruct format
        formatted = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        try:
            r = subprocess.run([
                "/root/llama.cpp/build/bin/llama-cli",
                "-m", gguf,
                "-p", formatted,
                "-n", "150",
                "--no-display-prompt",
                "-t", "8",
                "--temp", "0.7",
                "-r", "<|eot_id|>",
            ], capture_output=True, text=True, timeout=300)
            output = r.stdout.strip()
            results[f"prompt_{i+1}"] = {"prompt": prompt, "output": output[:400]}
        except subprocess.TimeoutExpired:
            results[f"prompt_{i+1}"] = {"prompt": prompt, "output": "TIMED OUT"}
        except Exception as e:
            results[f"prompt_{i+1}"] = {"prompt": prompt, "output": f"ERROR: {e}"}

    return results


@app.local_entrypoint()
def main() -> None:
    print("\n=== Testing Llama 3.1 8B style clone (3 prompts) ===\n")
    results = test_model.remote()
    for key, val in results.items():
        print(f"\n{'='*60}")
        print(f"PROMPT: {val['prompt']}")
        print(f"{'='*60}")
        print(val["output"])
    print(f"\n{'='*60}")
