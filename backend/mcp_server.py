"""Hosted MCP server (FastMCP v2), mounted into the FastAPI app at /mcp.

Same bearer-key auth as the REST API: each tool reads the Authorization
header via CurrentHeaders() and resolves the owner through auth.lookup. The
plain `_*` helpers take an explicit owner so tests can call them directly
without an MCP transport.
"""
from __future__ import annotations

import base64
import os
import uuid
from pathlib import Path

from fastmcp import FastMCP
from fastmcp.dependencies import CurrentHeaders

import auth
import status
from pipeline import start_job

DATA = Path(os.environ.get("DATA_DIR", "/data"))
CAP = int(os.environ.get("MAX_ACTIVE_JOBS_PER_KEY", "1"))

mcp = FastMCP("style-clone")


class MCPUnauthorized(Exception):
    """Surfaces as an MCP error result when no valid key is presented."""


def _owner_or_raise(headers: dict | None) -> str:
    header = (headers or {}).get("authorization", "")
    token = header[7:].strip() if header[:7].lower() == "bearer " else ""
    if not token:
        raise MCPUnauthorized("unauthorized")
    row = status.lookup_key(auth.hash_key(token))
    if not row:
        raise MCPUnauthorized("unauthorized")
    return row["key_hash"]


# ── owner-scoped helpers (pure; testable without MCP) ─────────────────────

def _create_job(owner: str, author: str, synth_model: str,
                files: list[dict]) -> dict:
    job_id = uuid.uuid4().hex[:12]
    indir = DATA / "jobs" / job_id / "input"
    indir.mkdir(parents=True, exist_ok=True)
    saved = 0
    for f in files:
        name = f.get("name")
        b64 = f.get("content_b64")
        if not name or not b64:
            continue
        (indir / name).write_bytes(base64.b64decode(b64))
        saved += 1
    if saved == 0:
        return {"error": "no files provided"}
    if status.count_active_jobs(owner) >= CAP:
        return {"error": "job already running"}
    addrs = [a.strip() for a in author.replace("\n", ",").split(",") if a.strip()]
    status.create_job(job_id, addrs, synth_model, "llama3.2-3b", saved,
                      owner_key_hash=owner)
    start_job(job_id, addrs, synth_model)
    return {"job_id": job_id}


def _get_job_status(owner: str, job_id: str) -> dict:
    st = status.get_job(job_id)
    if not st or st.get("owner") != owner:
        return {"error": "not found"}
    return st


def _list_jobs(owner: str) -> list[dict]:
    return status.list_jobs(owner)


def _resume_job(owner: str, job_id: str) -> dict:
    st = status.get_job(job_id)
    if not st or st.get("owner") != owner:
        return {"error": "not found"}
    if st["stage"] == "done":
        return {"ok": True, "message": "already done"}
    status.update_job(job_id, stage="queued", message="retrying...", error=None)
    start_job(job_id, st["author"], st["synth_model"])
    return {"ok": True}


def _download_model(owner: str, job_id: str) -> dict:
    st = status.get_job(job_id)
    if not st or st.get("owner") != owner:
        return {"error": "not found"}
    if st["stage"] != "done":
        return {"error": "not ready"}
    gguf = DATA / "jobs" / job_id / "adapter.gguf"
    modelfile = DATA / "jobs" / job_id / "Modelfile"
    return {
        "adapter_b64": base64.b64encode(gguf.read_bytes()).decode(),
        "modelfile": modelfile.read_text() if modelfile.exists() else "",
        "adapter_mb": st.get("adapter_mb"),
    }


# ── MCP tools (resolve owner from the request Authorization header) ────────

@mcp.tool
async def create_job(author: str, synth_model: str,
                     files: list[dict], headers: dict = CurrentHeaders()) -> dict:
    """Start a style-clone job. Each file is {"name": str, "content_b64": str}."""
    return _create_job(_owner_or_raise(headers), author, synth_model, files)


@mcp.tool
async def get_job_status(job_id: str,
                         headers: dict = CurrentHeaders()) -> dict:
    """Owner-scoped job status."""
    return _get_job_status(_owner_or_raise(headers), job_id)


@mcp.tool
async def list_jobs(headers: dict = CurrentHeaders()) -> list[dict]:
    """List the caller's jobs only."""
    return _list_jobs(_owner_or_raise(headers))


@mcp.tool
async def resume_job(job_id: str,
                     headers: dict = CurrentHeaders()) -> dict:
    """Resume a failed/interrupted job (idempotent)."""
    return _resume_job(_owner_or_raise(headers), job_id)


@mcp.tool
async def download_model(job_id: str,
                         headers: dict = CurrentHeaders()) -> dict:
    """Download {"adapter_b64","modelfile","adapter_mb"}; only when stage == done."""
    return _download_model(_owner_or_raise(headers), job_id)
