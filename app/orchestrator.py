"""Orchestrator: runs the CPU pipeline stages, then calls the GPU step.

Runs in the light web image (no GPU needed for ingest/curate/synth — they're
parsing + OpenRouter API calls). The GPU training runs as a separate
.remote() call in the GPU image.

Spawned by the upload endpoint; updates status_store at each stage so the UI
polls show live progress.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import modal

from .common import (app, web_image, jobs_vol, DATA_VOL, set_status)
from .ingest import ingest
from .curate import curate
from .synth import synthesize, write_chat_splits
from .gpu import train_and_export


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


@app.function(image=web_image, timeout=70 * 60,
              volumes={DATA_VOL: jobs_vol},
              secrets=[modal.Secret.from_name("openrouter")])
def run_job(job_id: str, author_addresses: list[str],
            synth_model: str, base_model_key: str) -> None:
    api_key = os.environ["OPENROUTER_API_KEY"]
    jdir = Path(DATA_VOL) / "jobs" / job_id
    addrs = {a.strip().lower() for a in author_addresses if a.strip()}

    try:
        # ── ingest ─────────────────────────────────────────────────────────
        set_status(job_id, stage="ingesting", message="parsing uploads...",
                   progress_pct=2)
        samples = ingest(jdir / "input", addrs)
        _write_jsonl(jdir / "samples.jsonl", samples)
        jobs_vol.commit()
        if not samples:
            raise RuntimeError("No usable writing found in uploads "
                               "(check author addresses for email, or "
                               "upload .txt/.md for other writing).")
        set_status(job_id, stage="ingesting",
                   message=f"parsed {len(samples)} samples", n_samples=len(samples),
                   progress_pct=10)

        # ── curate ─────────────────────────────────────────────────────────
        set_status(job_id, stage="curating",
                   message=f"curating {len(samples)} samples...", progress_pct=15)

        kept, cstats = curate(samples, api_key)
        _write_jsonl(jdir / "curated.jsonl", kept)
        jobs_vol.commit()
        set_status(job_id, stage="curating",
                   message=f"kept {len(kept)}/{len(samples)} "
                           f"(dropped {cstats['dropped']})",
                   n_curated=len(kept), n_dropped=cstats["dropped"],
                   progress_pct=35)
        if len(kept) < 10:
            raise RuntimeError(f"Only {len(kept)} valuable samples after "
                               "curation — need more writing to train on.")

        # ── synthesize ─────────────────────────────────────────────────────
        def _synth_progress(done, total, ok, fail, npairs):
            set_status(job_id, stage="synthesizing",
                       message=f"synthesizing {done}/{total} · {npairs} pairs",
                       progress_pct=35 + int(done / max(total, 1) * 40),
                       n_pairs=npairs)
        pairs, sstats = synthesize(kept, api_key, model=synth_model,
                                   on_progress=_synth_progress)
        _write_jsonl(jdir / "pairs.jsonl", pairs)
        n_train, n_valid = write_chat_splits(pairs, jdir / "data")
        jobs_vol.commit()
        set_status(job_id, stage="synthesizing",
                   message=f"{n_train} train / {n_valid} valid pairs",
                   n_pairs=len(pairs), n_train=n_train, n_valid=n_valid,
                   progress_pct=75)

        # ── train + export (GPU) ───────────────────────────────────────────
        jobs_vol.commit()
        result = train_and_export.remote(job_id)
        # GPU function sets its own status; nothing more to do here.

    except Exception as e:
        set_status(job_id, stage="error", message=str(e)[:500], progress_pct=0,
                   error=str(e)[:500])
        jobs_vol.commit()
        raise
