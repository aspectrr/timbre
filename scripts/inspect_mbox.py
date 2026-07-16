#!/usr/bin/env python3
"""Inspect mbox files: accurate counts, date ranges, token estimates.

Read-only. Does not modify the raw files.
"""
from __future__ import annotations

import mailbox
import statistics
from datetime import datetime
from email.utils import parsedate_to_datetime
from pathlib import Path

RAW = Path(__file__).resolve().parent.parent / "data" / "raw"

# Rough token estimate. Real models (Llama/Mistral/Gemma) tokenize slightly
# differently, but chars/4 is within ~15% and good enough for planning.
CHARS_PER_TOKEN = 4


def parse_date(raw: str | None) -> datetime | None:
    if not raw:
        return None
    try:
        return parsedate_to_datetime(raw)
    except (TypeError, ValueError):
        return None


def extract_body(msg) -> str:
    """Pull the plain-text body, preferring text/plain over HTML."""
    if msg.is_multipart():
        for part in msg.walk():
            if part.get_content_type() == "text/plain":
                payload = part.get_payload(decode=True)
                if payload:
                    return payload.decode(part.get_content_charset() or "utf-8", errors="replace")
        return ""
    payload = msg.get_payload(decode=True)
    if payload:
        return payload.decode(msg.get_content_charset() or "utf-8", errors="replace")
    return msg.get_payload() or ""


def inspect(path: Path) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {path.name}  ({path.stat().st_size / 1e6:.1f} MB)")
    print('=' * 60)

    box = mailbox.mbox(str(path))
    bodies_chars = []
    dates = []
    senders = {}

    for msg in box:
        body = extract_body(msg).strip()
        if body:
            bodies_chars.append(len(body))
        d = parse_date(msg.get("Date"))
        if d:
            dates.append(d)
        frm = msg.get("From", "unknown")
        # group by email address only
        if "<" in frm:
            frm = frm.split("<", 1)[1].rstrip(">").strip().lower()
        senders[frm] = senders.get(frm, 0) + 1

    total = len(box)
    with_body = len(bodies_chars)
    print(f"  Total messages:       {total}")
    print(f"  With usable body:     {with_body}")
    print(f"  Empty/no-text body:   {total - with_body}")

    if dates:
        dates.sort()
        print(f"  Date range:           {dates[0].date()}  ->  {dates[-1].date()}")

    print(f"\n  Top senders (confirms authorship filtering later):")
    for sender, n in sorted(senders.items(), key=lambda x: -x[1])[:5]:
        print(f"    {n:>4}  {sender[:70]}")

    if bodies_chars:
        total_chars = sum(bodies_chars)
        total_tokens = total_chars / CHARS_PER_TOKEN
        print(f"\n  Body character stats:")
        print(f"    total chars:        {total_chars:,}")
        print(f"    est. tokens (~c/4): {int(total_tokens):,}")
        print(f"    median body chars:  {int(statistics.median(bodies_chars)):,}")
        print(f"    mean body chars:    {int(statistics.mean(bodies_chars)):,}")
        # distribution buckets
        buckets = [(0, 50), (50, 200), (200, 1000), (1000, 5000), (5000, 10**9)]
        labels = ["<50 (junk)", "50-200", "200-1k", "1k-5k", "5k+"]
        print(f"    length distribution:")
        for (lo, hi), label in zip(buckets, labels):
            n = sum(1 for c in bodies_chars if lo <= c < hi)
            print(f"      {label:>12}: {n:>4}")


def main() -> None:
    files = sorted(RAW.glob("*.mbox"))
    if not files:
        print(f"No .mbox files in {RAW}")
        return
    for f in files:
        inspect(f)


if __name__ == "__main__":
    main()
