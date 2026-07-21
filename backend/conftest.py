"""Shared test setup.

DATA_DIR / DBOS_DB must point at a temp dir BEFORE status/main are imported
(both read these at import time). Set here so every collected test module
inherits it.
"""
import os
import tempfile
from pathlib import Path

_TMP = Path(tempfile.mkdtemp(prefix="sc_test_"))
os.environ["DATA_DIR"] = str(_TMP)
os.environ["DBOS_DB"] = f"sqlite:///{_TMP}/dbos.sqlite"
os.environ.setdefault("MAX_ACTIVE_JOBS_PER_KEY", "1")
