"""GPU step: LoRA training + clean 16-bit merge + adapter→GGUF export.

Proven path (see scripts/modal_full_run.py):
  1. QLoRA train (4-bit) → save adapter
  2. Reload adapter at FULL 16-bit → merge (clean weights, no 4-bit corruption)
  3. convert_lora_to_gguf → adapter.gguf (Ollama applies this on its own base)

The adapter GGUF path sidesteps the Llama-3 vocab-padding mismatch that
breaks a full merged GGUF in Ollama. Final delivery: adapter.gguf + a
Modelfile that does `FROM llama3.2:3b` + `ADAPTER ./adapter.gguf`.

Resilience: training checkpoints every 50 steps AND commits them to the
volume durably, so a killed run resumes from the last checkpoint instead of
restarting. Each downstream step (merge, export) is guarded by a sentinel,
so a re-spawn after a deploy/crash skips finished work and picks up where it
left off.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

from .common import (app, gpu_image, jobs_vol, hf_cache, DATA_VOL,
                     set_status)

MODEL = "unsloth/Llama-3.2-3B-Instruct"
MAX_SEQ_LEN = 1024
LORA_R, LORA_ALPHA, LORA_DROPOUT = 16, 32, 0.05
EPOCHS, LR, BATCH, GRAD_ACCUM = 3, 2e-5, 2, 4  # effective batch 8
SAVE_STEPS = 50  # checkpoint cadence — enables resume after a crash/deploy


def _load_split(path: Path):
    from datasets import Dataset
    rows = [json.loads(l) for l in path.open()]
    return Dataset.from_list(rows)


@app.function(image=gpu_image, gpu="l4", timeout=60 * 60,
              volumes={"/root/.cache/huggingface": hf_cache, DATA_VOL: jobs_vol})
def train_and_export(job_id: str) -> dict:
    from unsloth import FastLanguageModel
    from unsloth.chat_templates import get_chat_template
    from trl import SFTTrainer
    from transformers import TrainingArguments, TrainerCallback
    import torch

    jdir = Path(DATA_VOL) / "jobs" / job_id
    data_dir = jdir / "data"
    ckpt_dir = jdir / "checkpoints"
    adapter_dir = jdir / "adapter"
    merged_dir = jdir / "merged"
    adapter_gguf = jdir / "adapter.gguf"

    jobs_vol.reload()

    # Already fully done (e.g. retried after completion) — nothing to do.
    if adapter_gguf.exists():
        set_status(job_id, stage="done", message="ready to download",
                   progress_pct=100)
        return {"ok": True, "skipped": True}

    training_done = adapter_dir / ".training_done"

    # ── 1. train (resume from last checkpoint if one exists) ──────────────
    if not training_done.exists():
        set_status(job_id, stage="training",
                   message="loading base model (4-bit)...", progress_pct=76)
        jobs_vol.commit()

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=MODEL, max_seq_length=MAX_SEQ_LEN, dtype=None,
            load_in_4bit=True)
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

        train_ds = _load_split(data_dir / "train.jsonl").map(fmt, batched=True)
        valid_ds = _load_split(data_dir / "valid.jsonl").map(fmt, batched=True)

        model = FastLanguageModel.get_peft_model(
            model, r=LORA_R,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
            lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT, bias="none",
            use_gradient_checkpointing="unsloth", random_state=42)

        set_status(job_id, stage="training",
                   message=f"training ({len(train_ds)} pairs, {EPOCHS} epochs)")

        class _ProgressCB(TrainerCallback):
            def on_log(self, args, state, control, logs=None, **kw):
                loss = (logs or {}).get("loss")
                set_status(job_id, stage="training",
                           training_step=state.global_step,
                           train_loss=round(loss, 4) if loss else None,
                           epoch=round(state.epoch, 2),
                           progress_pct=min(95, int(state.epoch / EPOCHS * 95)),
                           message=f"step {state.global_step}"
                                   + (f" · loss {loss:.3f}" if loss else ""))

            def on_save(self, args, state, control, **kw):
                # Make the checkpoint durable so a restart can resume from it.
                try:
                    jobs_vol.commit()
                except Exception:
                    pass

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
                output_dir=str(ckpt_dir),
                eval_strategy="steps", eval_steps=50,
                save_strategy="steps", save_steps=SAVE_STEPS,
                save_total_limit=2,
            ),
            callbacks=[_ProgressCB()],
        )
        trainer.train(resume_from_checkpoint=True)
        eval_loss = trainer.evaluate().get("eval_loss")

        model.save_pretrained(str(adapter_dir))
        tokenizer.save_pretrained(str(adapter_dir))
        (adapter_dir / ".training_done").write_text("ok")
        if eval_loss:
            (adapter_dir / ".eval_loss").write_text(str(eval_loss))
        del model, trainer
        torch.cuda.empty_cache()
        import gc; gc.collect()
        set_status(job_id, stage="exporting",
                   message="training done — exporting...",
                   eval_loss=round(eval_loss, 4) if eval_loss else None,
                   progress_pct=96)
        jobs_vol.commit()
    else:
        set_status(job_id, stage="exporting",
                   message="resuming export after restart...", progress_pct=96)

    # ── 2. clean 16-bit merge (skip if already merged) ────────────────────
    if not (merged_dir / "config.json").exists():
        set_status(job_id, stage="exporting",
                   message="merging at 16-bit (clean weights)...",
                   progress_pct=96)
        jobs_vol.commit()
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=str(adapter_dir), max_seq_length=MAX_SEQ_LEN,
            dtype=torch.bfloat16, load_in_4bit=False)
        model.save_pretrained_merged(str(merged_dir), tokenizer,
                                     save_method="merged_16bit")
        del model
        torch.cuda.empty_cache()
        import gc; gc.collect()
        jobs_vol.commit()

    # ── 3. adapter → GGUF (skip if already exported) ──────────────────────
    if not adapter_gguf.exists():
        set_status(job_id, stage="exporting",
                   message="converting adapter to GGUF...", progress_pct=98)
        jobs_vol.commit()
        subprocess.run(["python", "/root/llama.cpp/convert_lora_to_gguf.py",
                        str(adapter_dir), "--outfile", str(adapter_gguf),
                        "--base", str(merged_dir)], check=True)
        jobs_vol.commit()

    size_mb = adapter_gguf.stat().st_size / 1e6
    eval_loss = None
    elf = adapter_dir / ".eval_loss"
    if elf.exists():
        try:
            eval_loss = round(float(elf.read_text().strip()), 4)
        except ValueError:
            eval_loss = None

    (jdir / "Modelfile").write_text(
        "FROM llama3.2:3b\n"
        "ADAPTER ./adapter.gguf\n\n"
        "PARAMETER temperature 0.7\n"
        "PARAMETER top_p 0.9\n"
        "PARAMETER repeat_penalty 1.15\n")

    jobs_vol.commit()
    set_status(job_id, stage="done", message="ready to download",
               error=None,
               eval_loss=eval_loss,
               adapter_mb=round(size_mb, 1), progress_pct=100)
    jobs_vol.commit()
    return {"ok": True, "adapter_mb": round(size_mb, 1), "eval_loss": eval_loss}
