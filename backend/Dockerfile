FROM python:3.11-slim

WORKDIR /app

# Build deps for any C extensions (sqlite3 is stdlib; modal/dbos are pure-ish)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY . ./backend/

# Persistent volume mount point (created via `fly volumes create`)
RUN mkdir -p /data

EXPOSE 8000

# SINGLE worker — required: SQLite is single-writer, and each worker would
# launch its own DBOS executor against the same dbos.sqlite.
CMD ["uvicorn", "backend.main:asgi_app", "--host", "0.0.0.0", "--port", "8000"]
