# uv-managed Python app. Multi-stage-free; uv sync is fast.
# Build context is backend/ (fly deploy run from backend/).
FROM python:3.11-slim

WORKDIR /app

# Install uv (single binary).
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# Resolve + install deps from the lockfile first (cached layer).
COPY pyproject.toml uv.lock ./
# --frozen: use the committed lock exactly. --no-install-project: we run the
# app from the working dir, no need to build/install the project package.
RUN uv sync --frozen --no-dev --no-install-project

# App code (flat: *.py at backend/ root).
COPY *.py ./

RUN mkdir -p /data

EXPOSE 8000

# SINGLE worker — required: SQLite is single-writer, and each worker would
# launch its own DBOS executor against the same dbos.sqlite.
# Flat layout: run from /app so `import status`, `import pipeline` resolve.
CMD ["uv", "run", "uvicorn", "main:asgi_app", "--host", "0.0.0.0", "--port", "8000"]
