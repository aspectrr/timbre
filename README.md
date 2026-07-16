# Style Clone

Train a model that writes like you — from your sent email to a local Ollama model.

## Architecture

```
[Render static: Solid.js SPA]
        │ fetch /api
        ▼
[Fly.io machine: FastAPI + DBOS]   single machine + persistent volume /data
        │ SQLite (/data/dbos.sqlite) — durable workflow + job state
        ├─ step: ingest
        ├─ step: curate
        ├─ step: synthesize
        └─ step: train → Modal [GPU L4]
```

Three services, each doing what it's best at:

| Layer | Where | Stack | Role |
|---|---|---|---|
| Frontend | Render static | Solid.js + Vite (**bun**) | SPA — upload, live progress, guide |
| Backend | Fly.io | FastAPI + DBOS (**uv**) | Durable orchestration, SQLite job state |
| GPU | Modal | Unsloth + llama.cpp | Stateless training; returns adapter bytes |

**Why DBOS:** the pipeline is a durable workflow of checkpointed steps. If the
Fly machine restarts or redeploys mid-job, DBOS resumes the workflow from the
last completed step automatically — no orphaned jobs, no lost work.

## Layout

```
frontend/   Solid.js + Vite SPA, built with bun (Render static). Adcker editorial style.
backend/    FastAPI + DBOS pipeline, uv project (Fly.io). Flat module layout.
gpu/        Modal GPU training app, its own uv project (stateless).
scripts/    standalone dev utilities (inspect_mbox.py)
```

Each service is self-contained — there is no root-level Python project.

## Deploy

### GPU (Modal) — its own uv project
```bash
cd gpu && uv run modal deploy -m app
```

### Backend (Fly.io) — uv project
```bash
cd backend
fly volumes create styleclone_data --region iad --size 1 --yes   # once
fly secrets set OPENROUTER_API_KEY=... MODAL_TOKEN_ID=... MODAL_TOKEN_SECRET=...
fly deploy
```
Deps are pinned in `backend/uv.lock`; the Dockerfile runs `uv sync --frozen`.

### Frontend (Render) — bun
Create a static site from this repo (root `frontend/`), build command
`bun install && VITE_API_URL=$API_URL bun run build`, publish `dist/`.
`render.yaml` is included for auto-detection.

## Local dev

```bash
# backend (uv, flat layout)
cd backend && uv sync && DATA_DIR=./.data uv run uvicorn main:asgi_app --reload

# frontend (bun proxies /api to localhost:8000)
cd frontend && bun install && bun run dev
```

## Inspecting raw data

```bash
uv run python scripts/inspect_mbox.py data/raw/gmail_sent.mbox
```
