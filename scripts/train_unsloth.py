#!/usr/bin/env python3
"""Unsloth QLoRA training for the style clone.

Provider-agnostic: runs on any GPU machine (RunPod pod, Modal container,
Colab, local). Loads the synthesized chat dataset, applies a LoRA to the
base model, trains, and exports a GGUF for Ollama.

Usage:
  python scripts/train_unsloth.py

Requires (see requirements-unsloth.txt):
  pip install unsloth trl transformers datasets accelerate bitsandbytes
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import torch
from datasets import Dataset
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
from trl import SFTTrainer
from transformers import TrainingArguments

# ── config (override via env) ─────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("MLX_DATA_DIR", ROOT / "data" / "datasets" / "mlx"))
OUT_DIR = Path(os.environ.get("OUT_DIR", ROOT / "data" / "models" / "unsloth"))
MODEL = os.environ.get("MODEL", "unsloth/Llama-3.2-3B-Instruct")

MAX_SEQ_LEN = 1024
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
EPOCHS = 3
LR = 2e-5
BATCH = 2          # Llama 3.2 3B fits batch 2 on an L4 (24GB)
GRAD_ACCUM = 4    # effective batch 8


def load_split(path: Path) -> Dataset:
    rows = [json.loads(l) for l in path.open()]
    return Dataset.from_list(rows)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Model:     {MODEL}")
    print(f"Data dir:  {DATA_DIR}")

    # 1. Load model + tokenizer (4-bit QLoRA)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL,
        max_seq_length=MAX_SEQ_LEN,
        dtype=None,
        load_in_4bit=True,
    )

    # 2. Apply Llama 3 chat template (works with our {"messages":[...]} format)
    tokenizer = get_chat_template(
        tokenizer,
        chat_template="llama-3.1",
        mapping={"role": "from", "content": "value",
                 "user": "human", "assistant": "gpt"},
        map_eos_token=True,
    )

    def formatting_prompts_func(examples):
        convos = examples["messages"]
        texts = [
            tokenizer.apply_chat_template(c, tokenize=False, add_generation_prompt=False)
            for c in convos
        ]
        return {"text": texts}

    train_ds = load_split(DATA_DIR / "train.jsonl").map(
        formatting_prompts_func, batched=True)
    valid_ds = load_split(DATA_DIR / "valid.jsonl").map(
        formatting_prompts_func, batched=True)
    print(f"Train rows: {len(train_ds)}  Valid rows: {len(valid_ds)}")

    # 3. Attach LoRA
    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_R,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )

    # 4. Train
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        eval_dataset=valid_ds,
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LEN,
        dataset_num_proc=2,
        packing=True,
        args=TrainingArguments(
            per_device_train_batch_size=BATCH,
            gradient_accumulation_steps=GRAD_ACCUM,
            warmup_ratio=0.1,
            num_train_epochs=EPOCHS,
            learning_rate=LR,
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            logging_steps=5,
            optim="adamw_8bit",
            weight_decay=0.1,
            lr_scheduler_type="linear",
            seed=42,
            output_dir=str(OUT_DIR / "checkpoints"),
            eval_strategy="steps",
            eval_steps=50,
            save_steps=999999,
        ),
    )
    trainer.train()

    # 5. Save LoRA adapter + merged 16-bit model for GGUF conversion
    adapter_dir = OUT_DIR / "adapter"
    model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    print(f"LoRA adapter saved: {adapter_dir}")

    # Save merged 16-bit model: the base that convert_lora_to_gguf needs.
    # (Loading at full precision here — not 4-bit — avoids the dequant
    # corruption that produces garbage tokens. See modal_full_run.py.)
    merged_dir = OUT_DIR / "merged"
    model.save_pretrained_merged(str(merged_dir), tokenizer, save_method="merged_16bit")
    print(f"Merged 16-bit model saved: {merged_dir}")
    print("\nDone. Convert the adapter to GGUF with scripts/modal_convert_adapter.py.")


if __name__ == "__main__":
    main()
