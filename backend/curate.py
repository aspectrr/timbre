"""Curate: a judgmental agent that keeps only valuable writing samples.

Not every uploaded chunk is worth cloning. This calls a teacher model to
classify each sample KEEP / DROP with a reason, dropping:
  - forwarded / quoted others' content
  - auto-generated mail (calendar, receipts, OTPs, system notices)
  - newsletters / press releases / form letters
  - signatures, disclaimers, boilerplate-only fragments
  - too short or not in the author's own voice

Uses a cheaper model than synthesis — classification, not generation.
Batched to save tokens/cost. Reuses OpenRouter via stdlib urllib.
"""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

CURATE_MODEL = "google/gemini-2.5-flash"  # cheap, good at classification
BATCH_SIZE = 8
MAX_CALLS = 4  # concurrency

SYSTEM = (
    "You are a data curator building a dataset to clone ONE specific person's "
    "writing style. You receive candidate writing samples and must decide which "
    "are genuine, valuable examples of THAT person's own voice."
)

USER_TMPL = """Decide KEEP or DROP for each writing sample below.

KEEP a sample if it is:
  - genuinely written by this person in their own words
  - substantive enough to convey style (real sentences, a point of view)
  - representative of how they actually write day-to-day

DROP a sample if it is:
  - forwarded or quoted content from someone else (you're seeing replies,
    cited text, newsletters, press releases, or articles they pasted)
  - auto-generated (calendar invites, receipts, OTP/login codes, system
    notifications, delivery updates, out-of-office auto-replies)
  - form letters, legal disclaimers, signatures, or boilerplate-only fragments
  - too short or generic to carry personal style ("ok thanks!", "see you then")

Samples:
__SAMPLES__

Return ONLY a JSON array, one object per sample id:
[{"id": 0, "keep": true, "reason": "short phrase"}, ...]
"""


def _call_openrouter(model: str, messages: list[dict], max_tokens: int,
                     api_key: str, retries: int = 4) -> dict:
    payload = json.dumps({
        "model": model, "max_tokens": max_tokens, "temperature": 0.2,
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


def _parse_verdicts(content: str, n_expected: int) -> list[dict]:
    start, end = content.find("["), content.rfind("]")
    if start == -1 or end == -1:
        # fallback: keep everything if model misbehaves (fail-open for recall)
        return [{"id": i, "keep": True, "reason": "parse-fail-kept"}
                for i in range(n_expected)]
    try:
        arr = json.loads(content[start:end + 1])
        return arr if isinstance(arr, list) else []
    except json.JSONDecodeError:
        return [{"id": i, "keep": True, "reason": "parse-fail-kept"}
                for i in range(n_expected)]


def curate(samples: list[dict], api_key: str,
           model: str = CURATE_MODEL) -> tuple[list[dict], dict]:
    """Return (kept_samples, stats). Drops samples the agent rejects."""
    if not samples:
        return [], {"in": 0, "kept": 0, "dropped": 0}

    # Batch samples by id.
    batches = []
    for i in range(0, len(samples), BATCH_SIZE):
        chunk = samples[i:i + BATCH_SIZE]
        block = "\n\n".join(
            f"[SAMPLE {i + j}]\n{s['text'][:1400]}" for j, s in enumerate(chunk))
        batches.append((list(range(i, i + len(chunk))), block))

    verdicts: dict[int, dict] = {}

    def work(batch):
        ids, block = batch
        user = USER_TMPL.replace("__SAMPLES__", block)
        data = _call_openrouter(model, [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": user},
        ], max_tokens=400, api_key=api_key)
        content = data["choices"][0]["message"]["content"]
        return {v.get("id"): v for v in _parse_verdicts(content, len(ids))}

    with ThreadPoolExecutor(max_workers=MAX_CALLS) as pool:
        for fut in as_completed({pool.submit(work, b): b for b in batches}):
            try:
                verdicts.update(fut.result())
            except Exception as e:
                # fail-open: on a batch error, keep its samples (lose recall
                # only if the agent is down, not precision)
                pass

    kept, dropped = [], 0
    for i, s in enumerate(samples):
        v = verdicts.get(i)
        if v is None or v.get("keep", True):
            kept.append(s)
        else:
            dropped += 1

    stats = {"in": len(samples), "kept": len(kept), "dropped": dropped,
             "model": model}
    return kept, stats
