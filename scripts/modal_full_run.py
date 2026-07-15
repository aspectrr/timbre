#!/usr/bin/env python3
"""End-to-end: train Llama 3.2 3B LoRA + clean 16-bit merge + GGUF export.

Single Modal run. Two phases:
  Phase 1: QLoRA training (4-bit) → save adapter
  Phase 2: reload base+adapter in FULL 16-bit → clean merge → GGUF

The 16-bit reload is the key fix: Unsloth's merge of a 4-bit base produces
corrupted weights (out-of-range token IDs). Loading at full precision
sidesteps the lossy dequantization entirely.

Usage: uv run modal run scripts/modal_full_run.py
"""
from __future__ import annotations

import modal

# ── image: Unsloth (train + load) + llama.cpp (GGUF convert) ───────────────
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

app = modal.App("full-run")
outputs = modal.Volume.from_name("fine-tune-outputs", create_if_missing=True)
hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)

MODEL = "unsloth/Llama-3.2-3B-Instruct"


@app.function(image=image, gpu="l4", timeout=60 * 60,
              volumes={"/root/.cache/huggingface": hf_cache,
                       "/root/outputs": outputs})
def run_all() -> dict:
    import os, json, subprocess, torch
    from unsloth import FastLanguageModel
    from unsloth.chat_templates import get_chat_template
    from datasets import Dataset
    from trl import SFTTrainer
    from transformers import TrainingArguments

    DATA_DIR = "/root/data/datasets/mlx"
    MAX_SEQ_LEN = 1024
    LORA_R, LORA_ALPHA, LORA_DROPOUT = 16, 32, 0.05
    EPOCHS, LR, BATCH, GRAD_ACCUM = 3, 2e-5, 2, 4  # batch 2, effective 8

    def load_split(p):
        rows = [json.loads(l) for l in open(p)]
        return Dataset.from_list(rows)

    # ════════════════════════════════════════════════════════════════════
    # PHASE 1: QLoRA training (4-bit)
    # ════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("PHASE 1: QLoRA training (4-bit)")
    print("=" * 70)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL, max_seq_length=MAX_SEQ_LEN, dtype=None, load_in_4bit=True)

    tokenizer = get_chat_template(
        tokenizer, chat_template="llama-3.1",
        mapping={"role": "from", "content": "value",
                 "user": "human", "assistant": "gpt"},
        map_eos_token=True)

    def fmt(examples):
        texts = [tokenizer.apply_chat_template(c, tokenize=False,
                                               add_generation_prompt=False)
                 for c in examples["messages"]]
        return {"text": texts}

    train_ds = load_split(os.path.join(DATA_DIR, "train.jsonl")).map(fmt, batched=True)
    valid_ds = load_split(os.path.join(DATA_DIR, "valid.jsonl")).map(fmt, batched=True)
    print(f"Train: {len(train_ds)}  Valid: {len(valid_ds)}")

    model = FastLanguageModel.get_peft_model(
        model, r=LORA_R,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT, bias="none",
        use_gradient_checkpointing="unsloth", random_state=42)

    trainer = SFTTrainer(
        model=model, tokenizer=tokenizer,
        train_dataset=train_ds, eval_dataset=valid_ds,
        dataset_text_field="text", max_seq_length=MAX_SEQ_LEN,
        dataset_num_proc=2, packing=True,
        args=TrainingArguments(
            per_device_train_batch_size=BATCH,
            gradient_accumulation_steps=GRAD_ACCUM,
            warmup_ratio=0.1, num_train_epochs=EPOCHS, learning_rate=LR,
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            logging_steps=5, optim="adamw_8bit", weight_decay=0.1,
            lr_scheduler_type="linear", seed=42,
            output_dir="/root/outputs/checkpoints",
            eval_strategy="steps", eval_steps=50, save_steps=999999,
        ),
    )
    trainer.train()

    adapter_dir = "/root/outputs/adapter"
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    print(f"\nAdapter saved: {adapter_dir}")

    # Free GPU memory before reloading at full precision
    del model, trainer
    torch.cuda.empty_cache()
    import gc; gc.collect()

    outputs.commit()
    print("\nPhase 1 complete. Reloading for clean merge...")

    # ════════════════════════════════════════════════════════════════════
    # PHASE 2: clean 16-bit merge + GGUF
    # ════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("PHASE 2: 16-bit merge + GGUF export")
    print("=" * 70)

    # Reload base+adapter in FULL 16-bit (no 4-bit → no corruption)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=adapter_dir, max_seq_length=MAX_SEQ_LEN,
        dtype=torch.bfloat16, load_in_4bit=False)
    print(f"  Vocab size: {model.config.vocab_size}")

    merged_dir = "/root/outputs/merged_clean"
    print("Merging + saving (16-bit)...")
    model.save_pretrained_merged(merged_dir, tokenizer,
                                 save_method="merged_16bit")
    print(f"  Saved to {merged_dir}")

    # Copy tokenizer files (save_pretrained_merged sometimes omits them)
    import shutil
    for tf in ["tokenizer.json", "tokenizer.model", "tokenizer_config.json",
               "special_tokens_map.json"]:
        src, dst = os.path.join(adapter_dir, tf), os.path.join(merged_dir, tf)
        if os.path.exists(src) and not os.path.exists(dst):
            shutil.copy2(src, dst)
            print(f"  Copied {tf}")

    # Convert to GGUF
    print("Converting to GGUF...")
    bf16 = os.path.join(merged_dir, "model-bf16.gguf")
    q4 = os.path.join(merged_dir, "model-Q4_K_M.gguf")
    subprocess.run(["python", "/root/llama.cpp/convert_hf_to_gguf.py",
                    merged_dir, "--outtype", "f16", "--outfile", bf16], check=True)
    subprocess.run(["/root/llama.cpp/build/bin/llama-quantize", bf16, q4,
                    "Q4_K_M"], check=True)
    os.remove(bf16)
    size_gb = os.path.getsize(q4) / 1e9
    print(f"  Q4_K_M: {size_gb:.1f} GB")

    outputs.commit()
    return {"ok": True, "path": q4, "size_gb": size_gb}


@app.local_entrypoint()
def main() -> None:
    print("\n=== Llama 3.2 3B: full run (train + 16-bit merge + GGUF) ===\n")
    result = run_all.remote()
    print(f"\nResult: {result}")
    if result.get("ok"):
        print(f"\n✓ Done! {result['size_gb']:.1f} GB GGUF at {result['path']}")
        print(f"\nDownload with:")
        print(f"  uv run modal volume get fine-tune-outputs"
              f" /merged_clean/model-Q4_K_M.gguf data/models/")
