"""Watchdog: periodically re-spawns jobs that got stranded mid-run.

A `modal deploy` (or a container crash) can leave a job's status frozen in a
non-terminal stage with nothing actually executing — exactly the "stuck at
starting..." failure we hit. Because `run_job` is now fully idempotent (it
skips completed stages and the GPU step resumes from its last checkpoint),
re-spawning a stranded job is always safe: it just picks up where it left off.

Runs every 2 min on the deployed app. A job is considered stranded if its
status hasn't updated in STALE_SEC while sitting in a non-terminal stage.
"""
from __future__ import annotations

import time

import modal

from .common import app, web_image, status_store
from .orchestrator import run_job

TERMINAL = {"done", "error"}
STALE_SEC = 300  # 5 min without a status update → stranded


@app.function(image=web_image, schedule=modal.Period(seconds=120))
def watchdog() -> dict:
    now = time.time()
    respawned = []
    for k in list(status_store.keys()):
        s = status_store.get(k) or {}
        if s.get("stage") in TERMINAL:
            continue
        updated = s.get("updated_at") or 0
        if now - updated < STALE_SEC:
            continue
        # Stranded — re-spawn. run_job resumes from the last checkpoint.
        # We do NOT write status here: run_job is the sole status writer, and
        # a competing write would race its first write on the eventually-
        # consistent Dict (could clobber it, leaving the job stuck). run_job's
        # status lands within seconds and refreshes updated_at.
        author = s.get("author") or []
        synth = s.get("synth_model") or "google/gemini-2.5-flash"
        base = s.get("base_model") or "llama3.2-3b"
        run_job.spawn(k, author, synth, base)
        respawned.append(k)
    return {"respawned": respawned}
