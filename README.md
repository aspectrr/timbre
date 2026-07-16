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

| Layer | Where | Role |
|---|---|---|
| Frontend | Render static (Solid.js) | SPA — upload, live progress, guide |
| Backend | Fly.io (FastAPI + DBOS) | Durable orchestration, SQLite job state |
| GPU | Modal | Stateless training; returns adapter bytes |

**Why DBOS:** the pipeline is a durable workflow of checkpointed steps. If the
Fly machine restarts or redeploys mid-job, DBOS resumes the workflow from the
last completed step automatically — no orphaned jobs, no lost work. This
replaces the fragile Modal-Dict status store that caused stuck/orphaned jobs.

## Layout

```
frontend/   Solid.js + Vite SPA (Render static)
backend/    FastAPI + DBOS pipeline (Fly.io)
gpu/        Modal GPU training app (stateless)
app/        (legacy) old all-in-one Modal app — being retired
scripts/    standalone dev utilities
```

## Deploy

### GPU (Modal) — one time
```bash
uv run modal deploy -m gpu.app
```

### Backend (Fly.io)
```bash
cd backend
fly volumes create styleclone_data --region iad --size 1 --yes   # once
fly secrets set OPENROUTER_API_KEY=... MODAL_TOKEN_ID=... MODAL_TOKEN_SECRET=...
fly deploy
```

### Frontend (Render)
Create a static site from this repo (root `frontend/`), build command
`npm install && VITE_API_URL=$API_URL npm run build`, publish `dist/`.
`render.yaml` is included for auto-detection.

## Local dev
```bash
# backend
cd backend && uvicorn backend.main:asgi_app --reload
# frontend (proxies /api to localhost:8000)
cd frontend && npm install && npm run dev
```
