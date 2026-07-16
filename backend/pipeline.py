"""Durable pipeline: DBOS steps, each checkpointed to SQLite.

A crash or deploy mid-workflow → DBOS resumes from the last completed step
automatically on process restart (DBOS.launch() recovers pending workflows).
Steps are also idempotent (skip if output exists), so a re-run step never
redoes finished work.

Determinism: the workflow body is pure step calls only — no DB writes, file
I/O, or time (DBOS replays the body on recovery and requires it deterministic).
All side effects live inside steps, which may be non-deterministic.

  step_ingest     → samples.jsonl
  step_curate     → curated.jsonl
  step_synthesize → train/valid.jsonl
  step_train      → adapter.gguf   (Modal GPU, stateless, returns bytes)
  step_finalize   → done status
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from dbos import DBOS

import status
from ingest import ingest
from curate import curate
from synth import synthesize, write_chat_splits

DATA = Path("/data")


def _job_dir(job_id: str) -> Path:
    d = DATA / "jobs" / job_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def _read_jsonl(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.open()]


def _write_jsonl(p: Path, rows: list[dict]) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n")


def _count_lines(p: Path) -> int:
    return sum(1 for _ in p.open())


def _guard(job_id: str, body):
    """Run a step's work; on failure, persist error status then re-raise so
    DBOS records the workflow error. Keeps the workflow body deterministic."""
    try:
        return body()
    except Exception as e:
        msg = str(e)[:500]
        status.update_job(job_id, stage="error", error=msg, message=msg,
                          progress_pct=0)
        raise


# ── steps ──────────────────────────────────────────────────────────────────

@DBOS.step()
def step_ingest(job_id: str, author: list[str]) -> None:
    def body():
        samples_path = _job_dir(job_id) / "samples.jsonl"
        if samples_path.exists():
            status.update_job(job_id, stage="ingesting",
                              message=f"parsed {_count_lines(samples_path)} samples",
                              n_samples=_count_lines(samples_path), progress_pct=10)
            return
        status.update_job(job_id, stage="ingesting", message="parsing uploads...",
                          progress_pct=2)
        addrs = {a.strip().lower() for a in author if a.strip()}
        samples = ingest(_job_dir(job_id) / "input", addrs)
        if not samples:
            raise RuntimeError("No usable writing found. Check your author email "
                               "address, or upload .txt/.md of other writing.")
        _write_jsonl(samples_path, samples)
        status.update_job(job_id, stage="ingesting",
                          message=f"parsed {len(samples)} samples",
                          n_samples=len(samples), progress_pct=10)
    _guard(job_id, body)


@DBOS.step()
def step_curate(job_id: str) -> None:
    def body():
        curated_path = _job_dir(job_id) / "curated.jsonl"
        if curated_path.exists():
            status.update_job(job_id, stage="curating",
                              message=f"kept {_count_lines(curated_path)} samples",
                              n_curated=_count_lines(curated_path), progress_pct=35)
            return
        samples = _read_jsonl(_job_dir(job_id) / "samples.jsonl")
        api_key = os.environ["OPENROUTER_API_KEY"]
        status.update_job(job_id, stage="curating",
                          message=f"curating {len(samples)} samples...",
                          progress_pct=15, n_samples=len(samples))
        kept, cstats = curate(samples, api_key)
        _write_jsonl(curated_path, kept)
        status.update_job(job_id, stage="curating",
                          message=f"kept {len(kept)}/{len(samples)}",
                          n_curated=len(kept), progress_pct=35)
        if len(kept) < 10:
            raise RuntimeError(f"Only {len(kept)} valuable samples after curation — "
                               "need more writing to train on.")
    _guard(job_id, body)


@DBOS.step()
def step_synthesize(job_id: str, synth_model: str) -> None:
    def body():
        train_path = _job_dir(job_id) / "data" / "train.jsonl"
        if train_path.exists():
            n_train = _count_lines(train_path)
            n_valid = _count_lines(_job_dir(job_id) / "data" / "valid.jsonl")
            status.update_job(job_id, stage="synthesizing",
                              message=f"{n_train} train / {n_valid} valid pairs",
                              progress_pct=75)
            return
        kept = _read_jsonl(_job_dir(job_id) / "curated.jsonl")
        api_key = os.environ["OPENROUTER_API_KEY"]

        def _prog(done, total, ok, fail, npairs):
            status.update_job(job_id, stage="synthesizing",
                              message=f"synthesizing {done}/{total} · {npairs} pairs",
                              progress_pct=35 + int(done / max(total, 1) * 40),
                              n_pairs=npairs)
        pairs, _ = synthesize(kept, api_key, model=synth_model, on_progress=_prog)
        _write_jsonl(_job_dir(job_id) / "pairs.jsonl", pairs)
        n_train, n_valid = write_chat_splits(pairs, _job_dir(job_id) / "data")
        status.update_job(job_id, stage="synthesizing",
                          message=f"{n_train} train / {n_valid} valid pairs",
                          n_pairs=len(pairs), progress_pct=75)
    _guard(job_id, body)


@DBOS.step()
def step_train(job_id: str) -> None:
    """Call Modal GPU (stateless): send training JSONL, receive adapter bytes."""
    def body():
        gguf_path = _job_dir(job_id) / "adapter.gguf"
        if gguf_path.exists():
            return
        import modal
        status.update_job(job_id, stage="training",
                          message="training on GPU (~10 min)...", progress_pct=80)
        train_text = (_job_dir(job_id) / "data" / "train.jsonl").read_text()
        valid_text = (_job_dir(job_id) / "data" / "valid.jsonl").read_text()
        fn = modal.Function.from_name("style-clone-gpu", "train_and_export")
        result = fn.remote(train_text, valid_text, job_id)
        gguf_path.write_bytes(result["adapter"])
        (_job_dir(job_id) / "Modelfile").write_text(result["modelfile"])
        status.update_job(job_id, stage="exporting",
                          message="exporting adapter...", progress_pct=98,
                          eval_loss=result.get("eval_loss"),
                          adapter_mb=result.get("adapter_mb"))
    _guard(job_id, body)


@DBOS.step()
def step_finalize(job_id: str) -> None:
    def body():
        gguf = _job_dir(job_id) / "adapter.gguf"
        size = gguf.stat().st_size / 1e6 if gguf.exists() else None
        status.update_job(job_id, stage="done", message="ready to download",
                          progress_pct=100, error=None,
                          adapter_mb=round(size, 1) if size else None)
    _guard(job_id, body)


# ── workflow (pure: step calls only) ───────────────────────────────────────

@DBOS.workflow()
def clone_style(job_id: str, author: list[str], synth_model: str) -> None:
    step_ingest(job_id, author)
    step_curate(job_id)
    step_synthesize(job_id, synth_model)
    step_train(job_id)
    step_finalize(job_id)


def start_job(job_id: str, author: list[str], synth_model: str) -> None:
    """Start the workflow in the background (durable: survives restart).
    DBOS auto-recovers pending workflows on process restart."""
    import status as _st
    handle = DBOS.start_workflow(clone_style, job_id, author, synth_model)
    _st.update_job(job_id, workflow_id=handle.workflow_id)
