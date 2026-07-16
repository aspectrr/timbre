"""Durable job state in SQLite.

Replaces modal.Dict (eventually consistent, which caused status-clobber bugs).
SQLite gives ACID transactions: a step writes its progress, the API reads it,
and there's never a stale-overwrite race.

The DBOS system database holds workflow/step checkpoints (for crash recovery);
this app table holds the human-facing progress (stage, counts, loss) that the
frontend polls. Both live in the same SQLite file on the Fly volume.
"""
from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path

DB_PATH = Path("/data/styleclone.db")

SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    job_id        TEXT PRIMARY KEY,
    stage         TEXT NOT NULL,        -- queued|ingesting|curating|synthesizing|training|exporting|done|error
    message       TEXT,
    progress_pct  INTEGER DEFAULT 0,
    author        TEXT,                  -- JSON list
    synth_model   TEXT,
    base_model    TEXT,
    workflow_id   TEXT,                  -- DBOS workflow handle id
    n_files       INTEGER,
    n_samples     INTEGER,
    n_curated     INTEGER,
    n_pairs       INTEGER,
    train_loss    REAL,
    eval_loss     REAL,
    adapter_mb    REAL,
    error         TEXT,
    created_at    REAL,
    updated_at    REAL
);
"""


def _connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, timeout=30, isolation_level=None)  # autocommit
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")  # concurrent reader + writer
    conn.execute("PRAGMA busy_timeout=30000")
    return conn


def init_db() -> None:
    with _connect() as c:
        c.executescript(SCHEMA)


def create_job(job_id: str, author: list[str], synth_model: str,
               base_model: str, n_files: int, workflow_id: str = None) -> None:
    now = time.time()
    with _connect() as c:
        c.execute(
            "INSERT OR REPLACE INTO jobs "
            "(job_id, stage, message, progress_pct, author, synth_model, "
            " base_model, n_files, workflow_id, created_at, updated_at) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (job_id, "queued", "starting...", 0, json.dumps(author),
             synth_model, base_model, n_files, workflow_id, now, now))


def update_job(job_id: str, **fields) -> None:
    if not fields:
        return
    fields["updated_at"] = time.time()
    cols = ", ".join(f"{k} = ?" for k in fields)
    vals = list(fields.values()) + [job_id]
    with _connect() as c:
        c.execute(f"UPDATE jobs SET {cols} WHERE job_id = ?", vals)


def get_job(job_id: str) -> dict | None:
    with _connect() as c:
        row = c.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
    if not row:
        return None
    d = dict(row)
    d["author"] = json.loads(d["author"]) if d.get("author") else []
    return d
