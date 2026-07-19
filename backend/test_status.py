"""One-file self-check (no framework): status DB round-trips the new train
progress fields and the idempotent migration is safe to run twice.

Run: DATA_DIR=./.data uv run python test_status.py
"""
import shutil
from pathlib import Path

import status

status.DATA_DIR = Path("./.data_test")
status.DB_PATH = status.DATA_DIR / "styleclone.db"
if status.DATA_DIR.exists():
    shutil.rmtree(status.DATA_DIR)

# init_db must be idempotent (Fly calls it on every boot against an existing DB).
status.init_db()
status.init_db()  # second run: must not crash, must not duplicate train_step col

status.create_job("t1", ["a@x.com"], "m", "b", 1)
status.update_job("t1", stage="training", train_step=42, train_loss=1.23)
got = status.get_job("t1")
assert got["stage"] == "training", got
assert got["train_step"] == 42, got
assert abs(got["train_loss"] - 1.23) < 1e-6, got
print("status round-trip + migration: OK")

shutil.rmtree(status.DATA_DIR)
