#!/usr/bin/env python3
"""Convert Alpaca-format pairs into MLX chat-format train/valid splits.

Input:  data/datasets/train.jsonl  ({"instruction","input","output",...})
Output: data/datasets/mlx/train.jsonl
        data/datasets/mlx/valid.jsonl

MLX chat format (one JSON object per line):
  {"messages": [
      {"role": "user", "content": "<instruction>"},
      {"role": "assistant", "content": "<real email text>"}
  ]}

With mask_prompt enabled, only the assistant tokens contribute to loss,
so the model learns to PRODUCE Collin's emails, not memorize the prompts.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "data" / "datasets" / "train.jsonl"
DST = ROOT / "data" / "datasets" / "mlx"
VALID_FRAC = 0.05

# Drops pairs whose output is degenerate or would teach bad style.
MIN_OUTPUT_CHARS = 30


def main() -> None:
    pairs = [json.loads(l) for l in SRC.open()]
    DST.mkdir(parents=True, exist_ok=True)

    rows = []
    skipped = 0
    for p in pairs:
        instr = (p["instruction"] or "").strip()
        out = (p["output"] or "").strip()
        if not instr or len(out) < MIN_OUTPUT_CHARS:
            skipped += 1
            continue
        rows.append({"messages": [
            {"role": "user", "content": instr},
            {"role": "assistant", "content": out},
        ]})

    random.seed(42)
    random.shuffle(rows)
    n_valid = max(20, int(len(rows) * VALID_FRAC))
    valid = rows[:n_valid]
    train = rows[n_valid:]

    (DST / "train.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in train) + "\n"
    )
    (DST / "valid.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in valid) + "\n"
    )

    print(f"Source pairs:  {len(pairs)}")
    print(f"Skipped (<{MIN_OUTPUT_CHARS} chars): {skipped}")
    print(f"Train:         {len(train)}")
    print(f"Valid:         {len(valid)}")
    print(f"Written to:    {DST}/")


if __name__ == "__main__":
    main()
