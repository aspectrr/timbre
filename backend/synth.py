"""Synthesize: turn each sample into style training pairs via a teacher model.

Instruction back-translation: for each writing sample, the teacher invents
plausible requests that the sample would answer. The sample's real text
becomes the TARGET — so the LoRA learns to produce this voice, not the
teacher's. Reuses the proven approach from scripts/synthesize.py.

Also writes chat-format train/valid splits for the trainer.
"""
from __future__ import annotations

import json
import random
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

SYNTH_MODEL = "anthropic/claude-opus-4.8"  # high-quality instruction generation
PAIRS_PER_SAMPLE = 3
MAX_OUT = 600
MAX_CALLS = 8
VALID_FRAC = 0.05
MIN_OUTPUT_CHARS = 30

SYSTEM = (
    "You are a data-synthesis assistant. You are given a real piece of writing "
    "by a specific person. Your job: brainstorm short, plausible WRITING "
    "REQUESTS that this writing would be a natural response to. Think of "
    "requests a colleague, client, or assistant might send that would elicit "
    "this kind of writing. Vary them across topic, formality, and length."
)

USER_TMPL = """Here is a real piece of writing:
<writing>
__BODY__
</writing>

Write exactly 3 distinct writing requests (instructions/prompts) that this
writing plausibly answers. Each should be a realistic task like:
- "Write a cold email to {recipient type} about {topic}"
- "Reply to {scenario} with your availability"
- "Draft a follow-up to {situation}"

Rules:
- Requests must be generic (generalize any real names/specifics).
- Requests must plausibly produce writing of THIS tone and length.
- Return ONLY a JSON object: {"requests": ["request 1", "request 2", "request 3"]}
"""


def _call_openrouter(model: str, messages: list[dict], max_tokens: int,
                     api_key: str, retries: int = 4) -> dict:
    payload = json.dumps({
        "model": model, "max_tokens": max_tokens, "temperature": 0.8,
        "messages": messages,
    }).encode()
    headers = {"Authorization": f"Bearer {api_key}",
               "Content-Type": "application/json"}
    last_err = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(
                "https://openrouter.ai/api/v1/chat/completions",
                data=payload, headers=headers)
            with urllib.request.urlopen(req, timeout=90) as resp:
                return json.loads(resp.read())
        except urllib.error.HTTPError as e:
            last_err = e
            if e.code in (429, 500, 502, 503):
                time.sleep(2 ** attempt)
                continue
            raise
        except Exception as e:
            last_err = e
            time.sleep(2 ** attempt)
    raise RuntimeError(f"OpenRouter call failed after retries: {last_err}")


def _parse_requests(content: str) -> list[str]:
    start, end = content.find("{"), content.rfind("}")
    if start == -1 or end == -1:
        return []
    try:
        reqs = json.loads(content[start:end + 1]).get("requests", [])
        if isinstance(reqs, list):
            return [str(r).strip() for r in reqs if str(r).strip()]
    except json.JSONDecodeError:
        pass
    return []


def synthesize(samples: list[dict], api_key: str,
               model: str = SYNTH_MODEL,
               on_progress=None) -> tuple[list[dict], dict]:
    """Return (pairs, stats). pairs: {instruction, output}."""
    pairs: list[dict] = []
    n_ok = n_fail = 0

    def work(idx, s):
        try:
            data = _call_openrouter(model, [
                {"role": "system", "content": SYSTEM},
                {"role": "user",
                 "content": USER_TMPL.replace("__BODY__", s["text"][:3500])},
            ], MAX_OUT, api_key)
            reqs = _parse_requests(data["choices"][0]["message"]["content"])
            return idx, reqs, None
        except Exception as e:
            return idx, [], str(e)

    done = 0
    with ThreadPoolExecutor(max_workers=MAX_CALLS) as pool:
        futs = {pool.submit(work, i, s): i for i, s in enumerate(samples)}
        for fut in as_completed(futs):
            idx, reqs, err = fut.result()
            done += 1
            if err:
                n_fail += 1
            else:
                n_ok += 1
                for req in reqs:
                    pairs.append({"instruction": req,
                                  "output": samples[idx]["text"]})
            if on_progress and (done % 10 == 0 or done == len(samples)):
                on_progress(done, len(samples), n_ok, n_fail, len(pairs))

    stats = {"model": model, "samples": len(samples), "ok": n_ok,
             "fail": n_fail, "pairs": len(pairs)}
    return pairs, stats


def write_chat_splits(pairs: list[dict], out_dir: Path,
                      seed: int = 42) -> tuple[int, int]:
    """Convert pairs → chat-format train/valid jsonl. Returns (n_train, n_valid)."""
    rows = [{"messages": [
                {"role": "user", "content": p["instruction"].strip()},
                {"role": "assistant", "content": p["output"].strip()}]}
            for p in pairs
            if p["instruction"].strip() and len(p["output"].strip()) >= MIN_OUTPUT_CHARS]
    random.seed(seed)
    random.shuffle(rows)
    n_valid = max(20, int(len(rows) * VALID_FRAC)) if len(rows) > 40 else max(1, len(rows) // 10)
    n_valid = min(n_valid, len(rows))
    valid, train = rows[:n_valid], rows[n_valid:]
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "train.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in train) + "\n")
    (out_dir / "valid.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in valid) + "\n")
    return len(train), len(valid)
