"""Web entrypoint. Deploys the FastAPI app on Modal.

The FastAPI app lives in app/server.py and is imported only inside the
container (so fastapi — not installed locally — never loads at deploy time).

Deploy: uv run modal deploy -m app.web
Dev:    uv run modal serve  -m app.web
"""
from __future__ import annotations

import modal

from .common import app, web_image, jobs_vol, DATA_VOL
from .orchestrator import run_job   # noqa: registers run_job in the deployed app
from .gpu import train_and_export   # noqa: registers train_and_export in the deployed app


@app.function(image=web_image, volumes={DATA_VOL: jobs_vol})
@modal.asgi_app()
def web_app():
    from .server import create_app
    return create_app()
