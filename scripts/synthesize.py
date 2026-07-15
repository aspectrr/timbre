#!/usr/bin/env python3
"""Instruction back-translation: turn each email into training pairs.

For each cleaned email, ask the teacher model to generate 2-3 plausible
instructions/prompts that this email would be a natural response to. The
email's real text becomes the TARGET completion — so the LoRA learns to
produce this voice, never the teacher's.

Output: data/datasets/train.jsonl  — {"instruction", "input", "output"} rows
        (ShareGPT/Alpaca-compatible, directly usable by Unsloth)

Usage:
  python3 scripts/synthesize.py --pilot 5             # pilot on both models
  python3 scripts/synthesize.py --model cheap --limit 50
  python3 scripts/synthesize.py --model cheap         # full run
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ENV_PATH = ROOT / ".env"
CLEANED = ROOT / "data" / "cleaned" / "emails.jsonl"
OUT_DIR = ROOT / "data" / "datasets"

OPUS = "anthropic/claude-opus-4.8"
CHEAP = "google/gemini-2.5-flash"   # good instruction-following, $0.30/$2.50

# Cap output tokens per request. 3 instructions + JSON wrapper.
MAX_OUT = 600

SYSTEM = (
    "You are a data-synthesis assistant. You are given a real email written by "
    "a specific person. Your job: brainstorm short, plausible WRITING REQUESTS "
    "that this email would be a natural response to. Think of requests a "
    "colleague, client, or assistant might send that would elicit this kind of "
    "email. Vary them across: topic, formality, and length."
)

USER_TMPL = """Here is a real email:
<email>
__BODY__
</email>

Write exactly 3 distinct writing requests (instructions/prompts) that this email plausibly answers. Each request should be a realistic task like:
- "Write a cold email to {recipient type} about {topic}"
- "Reply to {scenario} with your availability"
- "Draft a follow-up to {situation}"
- "Compose an email introducing yourself to {person}"

Rules:
- Requests must be generic (do NOT copy the recipient's real name or specifics — generalize them, e.g. "a potential client", "a research colleague").
- Requests must plausibly produce an email of THIS tone and length.
- Return ONLY a JSON object: {"requests": ["request 1", "request 2", "request 3"]}
"""


def load_env() -> None:
    if not ENV_PATH.exists():
        return
    for line in ENV_PATH.read_text().splitlines():
        if "=" in line and not line.strip().startswith("#"):
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())


def call_model(model: str, body: str) -> tuple[list[str], dict]:
    """Return (requests, usage). Raises on persistent failure."""
    key = os.environ["OPENROUTER_API_KEY"]
    payload = json.dumps({
        "model": model,
        "max_tokens": MAX_OUT,
        "temperature": 0.8,
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": USER_TMPL.replace("__BODY__", body[:3500])},
        ],
    }).encode()
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    }
    last_err = None
    for attempt in range(4):
        try:
            req = urllib.request.Request(
                "https://openrouter.ai/api/v1/chat/completions",
                data=payload, headers=headers,
            )
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = json.loads(resp.read())
            content = data["choices"][0]["message"]["content"]
            usage = data.get("usage", {})
            requests = parse_requests(content)
            return requests, usage
        except urllib.error.HTTPError as e:
            last_err = e
            if e.code in (429, 500, 502, 503):
                time.sleep(2 ** attempt)
                continue
            raise
        except Exception as e:
            last_err = e
            time.sleep(2 ** attempt)
    raise RuntimeError(f"failed after retries: {last_err}")


def parse_requests(content: str) -> list[str]:
    """Extract the 'requests' list from a JSON response (tolerant of prose)."""
    # find first { ... last }
    start = content.find("{")
    end = content.rfind("}")
    if start == -1 or end == -1:
        return []
    try:
        obj = json.loads(content[start:end + 1])
        reqs = obj.get("requests", [])
        if isinstance(reqs, list):
            return [str(r).strip() for r in reqs if str(r).strip()]
    except json.JSONDecodeError:
        pass
    return []


def run_batch(records: list[dict], model: str, tag: str) -> tuple[list[dict], dict]:
    """Synthesize for a batch. Returns (pairs, cost-summary)."""
    pairs: list[dict] = []
    total_in = total_out = n_ok = n_fail = 0

    def work(idx: int, rec: dict):
        try:
            reqs, usage = call_model(model, rec["text"])
            return idx, rec, reqs, usage, None
        except Exception as e:
            return idx, rec, [], {}, str(e)

    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(work, i, r): i for i, r in enumerate(records)}
        done = 0
        for fut in as_completed(futures):
            idx, rec, reqs, usage, err = fut.result()
            done += 1
            if err:
                n_fail += 1
                print(f"  [{tag}] {done}/{len(records)} FAIL: {err[:80]}", file=sys.stderr)
                continue
            n_ok += 1
            total_in += usage.get("prompt_tokens", 0)
            total_out += usage.get("completion_tokens", 0)
            for req in reqs:
                pairs.append({
                    "instruction": req,
                    "input": "",
                    "output": rec["text"],   # real email = target
                    "meta": {"source": rec["source"], "date": rec["date"]},
                })
            if done % 10 == 0 or done == len(records):
                print(f"  [{tag}] {done}/{len(records)} ok={n_ok} fail={n_fail}")

    summary = {"model": model, "n_in": len(records), "n_ok": n_ok, "n_fail": n_fail,
               "tokens_in": total_in, "tokens_out": total_out,
               "pairs": len(pairs)}
    return pairs, summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", type=int, help="pilot N emails on BOTH models, compare")
    ap.add_argument("--limit", type=int, help="process first N emails only")
    ap.add_argument("--model", choices=["opus", "cheap"], default="cheap",
                    help="model to use for non-pilot runs")
    args = ap.parse_args()

    load_env()
    records = [json.loads(l) for l in CLEANED.open()]
    if args.limit:
        records = records[:args.limit]

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.pilot:
        subset = records[:args.pilot]
        print(f"\n=== PILOT: {len(subset)} emails on both models ===\n")
        results = {}
        for label, model in [("OPUS", OPUS), ("CHEAP", CHEAP)]:
            print(f"\n--- {label} ({model}) ---")
            pairs, summ = run_batch(subset, model, label)
            results[label] = (pairs, summ, model)

        # write both for inspection
        for label, (pairs, summ, model) in results.items():
            out = OUT_DIR / f"pilot_{label.lower()}.jsonl"
            with out.open("w") as f:
                for p in pairs:
                    f.write(json.dumps(p, ensure_ascii=False) + "\n")

        print("\n=== PILOT COST + VOLUME ===")
        for label, (pairs, summ, model) in results.items():
            # project to full 333
            scale = len(records) / len(subset)
            proj_in = int(summ["tokens_in"] * scale)
            proj_out = int(summ["tokens_out"] * scale)
            print(f"\n{label} ({model}):")
            print(f"  pilot: {summ['pairs']} pairs, {summ['tokens_in']} tok in / {summ['tokens_out']} tok out")
            print(f"  projected full run ({len(records)} emails):")
            print(f"    ~{int(proj_in+proj_out):,} tokens, {int(summ['pairs']*scale)} pairs")

        print("\nPilot files written. Review pilot_opus.jsonl vs pilot_cheap.jsonl")
        return

    model = OPUS if args.model == "opus" else CHEAP
    print(f"\n=== FULL RUN: {len(records)} emails on {model} ===\n")
    pairs, summ = run_batch(records, model, args.model)
    out = OUT_DIR / "train.jsonl"
    with out.open("w") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    print(f"\n=== DONE ===")
    print(f"  pairs written: {len(pairs)}")
    print(f"  tokens: {summ['tokens_in']:,} in / {summ['tokens_out']:,} out")
    print(f"  output: {out}")


if __name__ == "__main__":
    main()
