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
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response

import status
import pipeline  # noqa: registers @DBOS.step / @DBOS.workflow
from pipeline import start_job

DATA = Path(os.environ.get("DATA_DIR", "/data"))


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

app = FastAPI(title="Style Clone")
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get("CORS_ORIGIN", "*").split(","),
    allow_methods=["*"], allow_headers=["*"],
)


@app.get("/health")
def health() -> dict:
    return {"ok": True}


@app.post("/api/jobs")
async def create_job(
    author: str = Form(...),
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
    status.create_job(job_id, addrs, synth_model, "llama3.2-3b", saved)
    start_job(job_id, addrs, synth_model)
    return JSONResponse({"job_id": job_id})


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str) -> JSONResponse:
    st = status.get_job(job_id)
    if not st:
        return JSONResponse({"error": "not found"}, status_code=404)
    return JSONResponse(st)


@app.post("/api/jobs/{job_id}/resume")
def resume(job_id: str) -> JSONResponse:
    st = status.get_job(job_id)
    if not st:
        return JSONResponse({"error": "not found"}, status_code=404)
    if st["stage"] == "done":
        return JSONResponse({"ok": True, "message": "already done"})
    # Idempotent steps skip completed work; a fresh workflow resumes from there.
    status.update_job(job_id, stage="queued", message="retrying...", error=None)
    start_job(job_id, st["author"], st["synth_model"])
    return JSONResponse({"ok": True})


@app.get("/api/jobs/{job_id}/download/{which}")
def download(job_id: str, which: str) -> Response:
    if which not in ("adapter.gguf", "Modelfile"):
        return JSONResponse({"error": "bad file"}, status_code=400)
    path = DATA / "jobs" / job_id / which
    if not path.exists():
        return JSONResponse({"error": "not ready"}, status_code=404)
    media = "application/octet-stream" if which.endswith(".gguf") else "text/plain"
    return Response(path.read_bytes(), media_type=media,
                    headers={"Content-Disposition": f'attachment; filename="{which}"'})


# uvicorn reference (for `uvicorn backend.main:asgi_app`)
asgi_app = app


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8000")))
