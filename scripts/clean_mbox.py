#!/usr/bin/env python3
"""Clean mbox exports into authorship-filtered plain text.

Steps:
  1. Parse mbox, keep only messages authored by the target addresses.
  2. Extract body (text/plain preferred, HTML stripped via stdlib fallback).
  3. Strip quote blocks, forwarded sections, signatures, footers, artifacts.
  4. Frequency-strip trailing boilerplate lines (signatures that repeat).
  5. Dedup by body hash, drop junk under min length.
  6. Write data/cleaned/emails.jsonl + print summary + samples.

Reads from data/raw/, writes to data/cleaned/. Never touches raw files.
"""
from __future__ import annotations

import collections
import hashlib
import json
import mailbox
import re
from email.utils import getaddresses, parsedate_to_datetime
from html.parser import HTMLParser
from pathlib import Path

# ── config ────────────────────────────────────────────────────────────────
AUTHOR_ADDRESSES = {"cpfeifer@madcactus.org", "collinpfeifer@icloud.com"}
MIN_CHARS = 50          # drop "ok" / "thanks" noise
CHARS_PER_TOKEN = 4     # rough estimate for planning
BOILERPLATE_MIN_FREQ = 4  # line in >=4 emails => trailing boilerplate

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "cleaned" / "emails.jsonl"

# ── HTML → text (stdlib only) ─────────────────────────────────────────────
class _HTMLText(HTMLParser):
    """Minimal HTML-to-text: block tags => newline, ignore the rest."""

    _BLOCK = {"p", "br", "div", "tr", "li", "h1", "h2", "h3", "h4", "hr"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._out: list[str] = []

    def handle_starttag(self, tag, attrs):
        if tag in self._BLOCK:
            self._out.append("\n")

    def handle_endtag(self, tag):
        if tag in self._BLOCK:
            self._out.append("\n")

    def handle_data(self, data):
        self._out.append(data)

    def text(self) -> str:
        return "".join(self._out)


def html_to_text(html: str) -> str:
    p = _HTMLText()
    p.feed(html)
    return p.text()


# ── body extraction ───────────────────────────────────────────────────────
def get_body(msg) -> str:
    plain, html = "", ""
    if msg.is_multipart():
        for part in msg.walk():
            ctype = part.get_content_type()
            if ctype == "text/plain" and not plain:
                payload = part.get_payload(decode=True)
                if payload:
                    plain = payload.decode(part.get_content_charset() or "utf-8", errors="replace")
            elif ctype == "text/html" and not html:
                payload = part.get_payload(decode=True)
                if payload:
                    html = payload.decode(part.get_content_charset() or "utf-8", errors="replace")
    else:
        payload = msg.get_payload(decode=True)
        if payload:
            decoded = payload.decode(msg.get_content_charset() or "utf-8", errors="replace")
            if msg.get_content_type() == "text/html":
                html = decoded
            else:
                plain = decoded
    return plain if plain.strip() else (html_to_text(html) if html.strip() else "")


def author_address(msg) -> str:
    raw = msg.get("From", "")
    addrs = getaddresses([raw])
    if addrs and addrs[0][1]:
        return addrs[0][1].lower().strip()
    return ""


# ── cleaning ──────────────────────────────────────────────────────────────
# Forward/original-message separators. Cut everything below these.
_FORWARD_SEPARATORS = [
    re.compile(r"^-{4,}\s*$", re.M),                      # Outlook ____ rule
    re.compile(r"^On .{0,200}wrote:\s*$", re.M),          # Gmail "On X, Y wrote:"
    re.compile(r"^Begin forwarded message:\s*$", re.M),
    re.compile(r"^-{1,3} (Original|Forwarded) Message -{1,3}\s*$", re.M),
]

# Known footer phrases — cut at the line and everything after.
_FOOTER_PHRASES = re.compile(
    r"^(sent from my|get (outlook|gmail) for|confidentiality notice|"
    r"this e-?mail .* confidential|disclaimer:|— ?sent from|"
    r"CAUTION: This email)",
    re.I,
)

# Artifacts to remove from any line.
_URL_BRACKET = re.compile(r"\[https?://\S+\]")            # [https://image.png]
_TEL_MAILTO = re.compile(r"<(tel|mailto):[^>]+>")          # <tel:+1...>
_BARE_URL = re.compile(r"https?://\S+(?=\s|$)")


def cut_at_first_match(text: str, patterns) -> str:
    cut = len(text)
    for pat in patterns:
        m = pat.search(text)
        if m:
            cut = min(cut, m.start())
    return text[:cut]


def strip_inline_artifacts(line: str) -> str:
    line = _TEL_MAILTO.sub("", line)
    line = _URL_BRACKET.sub("", line)
    line = _BARE_URL.sub("", line)
    return line


def clean_body(raw: str, boilerplate: set[str]) -> str:
    # 1. Cut forwarded/original sections first.
    text = cut_at_first_match(raw, _FORWARD_SEPARATORS)

    lines = text.splitlines()
    kept: list[str] = []
    for line in lines:
        s = line.rstrip()
        # 2. Drop quote lines (any leading '>')
        if s.lstrip().startswith(">"):
            continue
        # 3. Drop footer phrases (and stop, since rest is footer)
        if _FOOTER_PHRASES.match(s.strip()):
            break
        # 4. Drop exact boilerplate signature lines
        if s.strip() in boilerplate:
            continue
        kept.append(s)

    # 5. Strip trailing empty lines, then strip trailing boilerplate again
    while kept and not kept[-1].strip():
        kept.pop()

    cleaned = [strip_inline_artifacts(l) for l in kept]
    result = "\n".join(cleaned).strip()
    # collapse 3+ blank lines to 2
    result = re.sub(r"\n{3,}", "\n\n", result)
    return result


def build_boilerplate_set(records: list[dict]) -> set[str]:
    """Lines that appear near the END of many emails = signature boilerplate.

    Position-aware: a line is boilerplate only if it lands in the trailing
    half of >= N emails AND in the trailing half at least 2x as often as in
    the leading half. This catches signatures/footers (end-heavy) while
    leaving greetings like 'Hi Shannon,' (start-heavy) intact.
    """
    leading = collections.Counter()
    trailing = collections.Counter()
    for r in records:
        text = cut_at_first_match(r["_raw"], _FORWARD_SEPARATORS)
        non_empty = [
            l.strip()
            for l in text.splitlines()
            if l.strip() and not l.lstrip().startswith(">")
        ]
        if not non_empty:
            continue
        mid = len(non_empty) // 2
        # dedupe within one email so a repeated line counts once per side
        leading.update(set(non_empty[:mid]))
        trailing.update(set(non_empty[mid:]))
    boilerplate = set()
    for line, t_count in trailing.items():
        if t_count >= BOILERPLATE_MIN_FREQ and t_count >= 2 * leading.get(line, 0):
            boilerplate.add(line)
    return boilerplate


# ── main ──────────────────────────────────────────────────────────────────
def process_mbox(path: Path) -> list[dict]:
    box = mailbox.mbox(str(path))
    source = path.stem
    records: list[dict] = []
    for msg in box:
        addr = author_address(msg)
        if addr not in AUTHOR_ADDRESSES:
            continue
        body = get_body(msg)
        if not body.strip():
            continue
        try:
            date = parsedate_to_datetime(msg.get("Date", "")) if msg.get("Date") else None
        except (TypeError, ValueError):
            date = None
        records.append({
            "source": source,
            "from": addr,
            "date": date.isoformat() if date else None,
            "_raw": body,
        })
    return records


def main() -> None:
    all_records: list[dict] = []
    for path in sorted(RAW.glob("*.mbox")):
        recs = process_mbox(path)
        print(f"{path.name}: {len(recs)} authored emails after author filter")
        all_records.extend(recs)

    boilerplate = build_boilerplate_set(all_records)
    print(f"\nBoilerplate lines detected ({len(boilerplate)}):")
    for line in sorted(boilerplate)[:25]:
        print(f"    {line[:70]}")

    # Final clean + dedup
    seen: set[str] = set()
    clean: list[dict] = []
    dropped_short = dropped_dup = 0
    for r in all_records:
        text = clean_body(r["_raw"], boilerplate)
        if len(text) < MIN_CHARS:
            dropped_short += 1
            continue
        h = hashlib.sha256(text.encode()).hexdigest()
        if h in seen:
            dropped_dup += 1
            continue
        seen.add(h)
        clean.append({
            "source": r["source"],
            "from": r["from"],
            "date": r["date"],
            "chars": len(text),
            "tokens_est": len(text) // CHARS_PER_TOKEN,
            "text": text,
        })

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w") as f:
        for r in clean:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    total_chars = sum(r["chars"] for r in clean)
    total_tokens = sum(r["tokens_est"] for r in clean)
    print(f"\n{'=' * 60}")
    print(f"  CLEANED OUTPUT")
    print(f"{'=' * 60}")
    print(f"  Author-filtered in:   {len(all_records)}")
    print(f"  Clean & kept:         {len(clean)}")
    print(f"  Dropped (too short):  {dropped_short}")
    print(f"  Dropped (duplicate):  {dropped_dup}")
    print(f"  Total clean chars:    {total_chars:,}")
    print(f"  Total clean tokens:   {total_tokens:,} (est)")
    print(f"  Written to:           {OUT}")
    print(f"\n  Length distribution of clean emails:")
    buckets = [(0, 100), (100, 400), (400, 1500), (1500, 5000), (5000, 10**9)]
    labels = ["<100", "100-400", "400-1.5k", "1.5k-5k", "5k+"]
    for (lo, hi), label in zip(buckets, labels):
        n = sum(1 for r in clean if lo <= r["chars"] < hi)
        print(f"    {label:>10}: {n:>4}")


if __name__ == "__main__":
    main()
