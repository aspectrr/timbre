"""Stateless GPU training app (Modal).

Called remotely from the Fly backend via:
    modal.Function.from_name("style-clone-gpu", "train_and_export").remote(...)

Takes the training JSONL as strings (small), runs the full proven pipeline
(QLoRA train → clean 16-bit merge → adapter→GGUF), and RETURNS the adapter
bytes + Modelfile. No Modal job-data volume — Modal is pure GPU compute;
durability lives on the Fly side.

Deploy: uv run modal deploy -m gpu.app
"""
from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path

import modal

app = modal.App("style-clone-gpu")

hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)

MODEL = "unsloth/Llama-3.2-3B-Instruct"
MAX_SEQ_LEN = 1024
LORA_R, LORA_ALPHA, LORA_DROPOUT = 16, 32, 0.05
EPOCHS, LR, BATCH, GRAD_ACCUM = 3, 2e-5, 2, 4

gpu_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04", add_python="3.11")
    .apt_install("git", "build-essential", "cmake", "ffmpeg", "libsm6", "libxext6")
    .uv_pip_install("unsloth", "peft", "transformers", "torch", "accelerate",
                    "datasets", "trl", "bitsandbytes")
    .pip_install("gguf", "sentencepiece", "protobuf")
    .run_commands(
        "git clone --branch b4990 --depth 1 "
        "https://github.com/ggml-org/llama.cpp /root/llama.cpp",
        "cd /root/llama.cpp && cmake -B build "
        "-DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=OFF "
        "-DLLAMA_CURL=OFF && cmake --build build --config Release -j",
    )
)


def _load_split(rows_text: str):
    from datasets import Dataset
    rows = [json.loads(l) for l in rows_text.splitlines() if l.strip()]
    return Dataset.from_list(rows)


@app.function(image=gpu_image, gpu="l4", timeout=60 * 60,
              volumes={"/root/.cache/huggingface": hf_cache})
def train_and_export(train_jsonl: str, valid_jsonl: str, job_id: str = ""):
    """Generator: yields live training events, then the final result.

    Events: {"type":"phase"}, {"type":"log"}, {"type":"eval"},
            {"type":"result"} (last), or {"type":"error"}.
    Training runs on a worker thread; a TrainerCallback pushes events onto a
    queue that this generator drains, so step/loss stream to the caller as they
    happen instead of returning once at the end."""
    import gc
    import queue
    import threading
    from unsloth import FastLanguageModel
    from unsloth.chat_templates import get_chat_template
    from trl import SFTTrainer
    from transformers import TrainingArguments, TrainerCallback
    import torch

    work = Path(tempfile.mkdtemp(prefix=f"sc_{job_id}_"))
    data_dir = work / "data"
    adapter_dir = work / "adapter"
    merged_dir = work / "merged"
    adapter_gguf = work / "adapter.gguf"
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "train.jsonl").write_text(train_jsonl)
    (data_dir / "valid.jsonl").write_text(valid_jsonl)

    q: queue.Queue = queue.Queue()
    SENTINEL = object()

    # ── 1. model + trainer setup (main thread) ─────────────────────────────
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

    train_ds = _load_split(train_jsonl).map(fmt, batched=True)
    valid_ds = _load_split(valid_jsonl).map(fmt, batched=True)

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
            output_dir=str(work / "checkpoints"),
            eval_strategy="steps", eval_steps=50,
            save_strategy="no",
        ),
    )
    max_steps = trainer.args.max_steps or 0

    class _Prog(TrainerCallback):
        def _push(self, state, logs, kind):
            q.put({"type": kind, "step": state.global_step,
                   "max_steps": max_steps,
                   "loss": (logs or {}).get("loss"),
                   "eval_loss": (logs or {}).get("eval_loss")})

        def on_log(self, args, state, logs=None, **kw):
            self._push(state, logs, "log")

        def on_evaluate(self, args, state, logs=None, **kw):
            self._push(state, logs, "eval")

    trainer.add_callback(_Prog())

    # ── 2. run training + export on a worker thread ────────────────────────
    def _run():
        try:
            q.put({"type": "phase", "phase": "training"})
            trainer.train()
            eval_loss = trainer.evaluate().get("eval_loss")

            model.save_pretrained(str(adapter_dir))
            tokenizer.save_pretrained(str(adapter_dir))
            del model, trainer
            torch.cuda.empty_cache(); gc.collect()

            q.put({"type": "phase", "phase": "merging"})
            m, tok = FastLanguageModel.from_pretrained(
                model_name=str(adapter_dir), max_seq_length=MAX_SEQ_LEN,
                dtype=torch.bfloat16, load_in_4bit=False)
            m.save_pretrained_merged(str(merged_dir), tok, save_method="merged_16bit")
            del m
            torch.cuda.empty_cache(); gc.collect()

            q.put({"type": "phase", "phase": "converting"})
            subprocess.run(["python", "/root/llama.cpp/convert_lora_to_gguf.py",
                            str(adapter_dir), "--outfile", str(adapter_gguf),
                            "--base", str(merged_dir)], check=True)
            hf_cache.commit()

            q.put({"type": "result",
                    "adapter": adapter_gguf.read_bytes(),
                    "modelfile": (
                        "FROM llama3.2:3b\n"
                        "ADAPTER ./adapter.gguf\n\n"
                        "PARAMETER temperature 0.7\n"
                        "PARAMETER top_p 0.9\n"
                        "PARAMETER repeat_penalty 1.15\n"),
                    "eval_loss": round(eval_loss, 4) if eval_loss else None,
                    "adapter_mb": round(adapter_gguf.stat().st_size / 1e6, 1)})
        except Exception as e:  # surface to the generator → DBOS step errors
            q.put({"type": "error", "error": str(e)[:500]})
        finally:
            q.put(SENTINEL)

    threading.Thread(target=_run, daemon=True).start()

    # ── 3. drain the queue, yielding events as they arrive ────────────────
    while True:
        item = q.get()
        if item is SENTINEL:
            break
        if item.get("type") == "error":
            raise RuntimeError(item["error"])
        yield item
