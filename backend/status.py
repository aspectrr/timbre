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
import os
import sqlite3
import time
from pathlib import Path

# Configurable: /data on Fly (persistent volume), override for local dev.
DATA_DIR = Path(os.environ.get("DATA_DIR", "/data"))
DB_PATH = DATA_DIR / "styleclone.db"

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
    train_step    INTEGER,
    train_loss    REAL,
    eval_loss     REAL,
    adapter_mb    REAL,
    error         TEXT,
    owner         TEXT,                  -- sha256 of the creator's API key
    created_at    REAL,
    updated_at    REAL
);

CREATE TABLE IF NOT EXISTS api_keys (
    key_hash      TEXT PRIMARY KEY,      -- sha256 hex of the plaintext key
    label         TEXT,
    created_at    REAL
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
        # Idempotent migration for existing volumes: CREATE TABLE IF NOT EXISTS
        # won't add columns to an existing /data DB on a redeployed Fly machine.
        cols = {r[1] for r in c.execute("PRAGMA table_info(jobs)")}
        if "train_step" not in cols:
            c.execute("ALTER TABLE jobs ADD COLUMN train_step INTEGER")
        if "owner" not in cols:
            c.execute("ALTER TABLE jobs ADD COLUMN owner TEXT")


def create_job(job_id: str, author: list[str], synth_model: str,
               base_model: str, n_files: int, workflow_id: str = None,
               owner_key_hash: str = None) -> None:
    now = time.time()
    with _connect() as c:
        c.execute(
            "INSERT OR REPLACE INTO jobs "
            "(job_id, stage, message, progress_pct, author, synth_model, "
            " base_model, n_files, workflow_id, owner, created_at, updated_at) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (job_id, "queued", "starting...", 0, json.dumps(author),
             synth_model, base_model, n_files, workflow_id, owner_key_hash, now, now))


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


def list_jobs(owner_key_hash: str) -> list[dict]:
    """All jobs owned by a key (newest first)."""
    with _connect() as c:
        rows = c.execute(
            "SELECT * FROM jobs WHERE owner = ? ORDER BY created_at DESC",
            (owner_key_hash,)).fetchall()
    out = []
    for r in rows:
        d = dict(r)
        d["author"] = json.loads(d["author"]) if d.get("author") else []
        out.append(d)
    return out


def count_active_jobs(owner_key_hash: str) -> int:
    """Jobs for this key not in a terminal stage (done/error)."""
    with _connect() as c:
        row = c.execute(
            "SELECT COUNT(*) FROM jobs "
            "WHERE owner = ? AND stage NOT IN ('done', 'error')",
            (owner_key_hash,)).fetchone()
    return row[0]


# ── API keys ──────────────────────────────────────────────────────────────

def create_key(key_hash: str, label: str | None) -> dict:
    now = time.time()
    with _connect() as c:
        c.execute(
            "INSERT OR REPLACE INTO api_keys (key_hash, label, created_at) "
            "VALUES (?,?,?)",
            (key_hash, label, now))
    return {"key_hash": key_hash, "label": label, "created_at": now}


def lookup_key(key_hash: str) -> dict | None:
    with _connect() as c:
        row = c.execute(
            "SELECT * FROM api_keys WHERE key_hash = ?", (key_hash,)).fetchone()
    return dict(row) if row else None
