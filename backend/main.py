"""FastAPI + DBOS backend entrypoint.

Single process runs the web server AND the DBOS workflow executor. On
restart, DBOS.launch() recovers any in-flight workflows from SQLite and
resumes them from the last completed step.

Run: uvicorn backend.main:asgi_app  (or `python -m backend.main`)
"""
from __future__ import annotations

import os
import uuid
from pathlib import Path

from dbos import DBOS, DBOSConfig
from fastapi import Depends, FastAPI, File, Form, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel

import status
import pipeline  # noqa: registers @DBOS.step / @DBOS.workflow
from pipeline import start_job
import mcp_server
import auth
from auth import verify_key

DATA = Path(os.environ.get("DATA_DIR", "/data"))
# Per-key concurrent active-job cap (active = any stage except done/error).
MAX_ACTIVE_JOBS_PER_KEY = int(os.environ.get("MAX_ACTIVE_JOBS_PER_KEY", "1"))


class KeyCreate(BaseModel):
    label: str | None = None


def _setup_dbos() -> None:
    cfg: DBOSConfig = {
        "name": "style-clone",
        # SQLite on the persistent Fly volume (NOT cwd, which is ephemeral).
        # Override via DBOS_DB for local dev.
        "system_database_url": os.environ.get(
            "DBOS_DB", f"sqlite:///{os.environ.get('DATA_DIR', '/data')}/dbos.sqlite"),
        "use_listen_notify": False,  # required for SQLite
    }
    DBOS(config=cfg)


_setup_dbos()
status.init_db()
DBOS.launch()  # starts workflow executor + recovers pending workflows

# FastMCP streamable-HTTP app: built with path="/" and mounted at /mcp below.
# Its lifespan must be passed to FastAPI so MCP session management works.
_mcp_http = mcp_server.mcp.http_app(path="/")
app = FastAPI(title="Style Clone", lifespan=_mcp_http.lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get("CORS_ORIGIN", "*").split(","),
    allow_methods=["*"], allow_headers=["*"],
)


@app.exception_handler(auth.UnauthorizedError)
async def _unauthorized(request, exc):
    return JSONResponse({"error": "unauthorized"}, status_code=401)


@app.get("/health")
def health() -> dict:
    return {"ok": True}


@app.post("/api/jobs", dependencies=[Depends(verify_key)])
async def create_job(
    request: Request,
    author: str = Form(""),
    synth_model: str = Form("anthropic/claude-opus-4.8"),
    files: list[UploadFile] = File(...),
) -> JSONResponse:
    job_id = uuid.uuid4().hex[:12]
    indir = DATA / "jobs" / job_id / "input"
    indir.mkdir(parents=True, exist_ok=True)
    saved = 0
    for f in files:
        if not f.filename:
            continue
        (indir / f.filename).write_bytes(await f.read())
        saved += 1
    if saved == 0:
        return JSONResponse({"error": "no files uploaded"}, status_code=400)

    addrs = [a.strip() for a in author.replace("\n", ",").split(",") if a.strip()]
    # Email files need an author address to filter sent mail; plain docs never do.
    if any(Path(f.filename or "").suffix.lower() in (".mbox", ".eml")
           for f in files if f.filename) and not addrs:
        return JSONResponse(
            {"error": "Email files (.mbox/.eml) need your email address "
                      "to filter sent mail."},
            status_code=400)

    owner = request.state.owner
    if status.count_active_jobs(owner) >= MAX_ACTIVE_JOBS_PER_KEY:
        return JSONResponse({"error": "job already running"}, status_code=409)
    status.create_job(job_id, addrs, synth_model, "llama3.2-3b", saved,
                      owner_key_hash=owner)
    start_job(job_id, addrs, synth_model)
    return JSONResponse({"job_id": job_id})


@app.get("/api/jobs", dependencies=[Depends(verify_key)])
def list_jobs_route(request: Request) -> JSONResponse:
    return JSONResponse(status.list_jobs(request.state.owner))


@app.get("/api/jobs/{job_id}", dependencies=[Depends(verify_key)])
def job_status(job_id: str, request: Request) -> JSONResponse:
    st = status.get_job(job_id)
    if not st or st.get("owner") != request.state.owner:
        return JSONResponse({"error": "not found"}, status_code=404)
    return JSONResponse(st)


@app.post("/api/jobs/{job_id}/resume", dependencies=[Depends(verify_key)])
def resume(job_id: str, request: Request) -> JSONResponse:
    st = status.get_job(job_id)
    if not st or st.get("owner") != request.state.owner:
        return JSONResponse({"error": "not found"}, status_code=404)
    if st["stage"] == "done":
        return JSONResponse({"ok": True, "message": "already done"})
    # Idempotent steps skip completed work; a fresh workflow resumes from there.
    status.update_job(job_id, stage="queued", message="retrying...", error=None)
    start_job(job_id, st["author"], st["synth_model"])
    return JSONResponse({"ok": True})


@app.get("/api/jobs/{job_id}/download/{which}",
         dependencies=[Depends(verify_key)])
def download(job_id: str, which: str, request: Request) -> Response:
    st = status.get_job(job_id)
    if not st or st.get("owner") != request.state.owner:
        return JSONResponse({"error": "not found"}, status_code=404)
    if which not in ("adapter.gguf", "Modelfile"):
        return JSONResponse({"error": "bad file"}, status_code=400)
    path = DATA / "jobs" / job_id / which
    if not path.exists():
        return JSONResponse({"error": "not ready"}, status_code=404)
    media = "application/octet-stream" if which.endswith(".gguf") else "text/plain"
    return Response(path.read_bytes(), media_type=media,
                    headers={"Content-Disposition": f'attachment; filename="{which}"'})


# ── API keys ──────────────────────────────────────────────────────────────
# POST /api/keys is the ONE unauthenticated route: it mints a key and returns
# the plaintext exactly once. GET /api/keys is authed and never returns it.

@app.post("/api/keys")
def mint_key(body: KeyCreate) -> JSONResponse:
    key, key_hash = auth.new_key()
    status.create_key(key_hash, body.label)
    return JSONResponse({"key": key, "label": body.label})


@app.get("/api/keys", dependencies=[Depends(verify_key)])
def list_keys(request: Request) -> JSONResponse:
    row = status.lookup_key(request.state.owner)
    keys = ([{"label": row["label"], "created_at": row["created_at"]}]
            if row else [])
    return JSONResponse(keys)


# Hosted MCP server: tools reuse the same status/pipeline helpers + bearer key.
app.mount("/mcp", _mcp_http)


# uvicorn reference (for `uvicorn backend.main:asgi_app`)
asgi_app = app


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8000")))
