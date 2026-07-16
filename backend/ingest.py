"""Ingest: parse uploaded writing into cleaned author samples.

Supports:
  .mbox / .eml  — email, author-filtered (keep only target addresses)
  .txt / .md    — free-form writing (treated as the author's own words)

Cleaning logic ported from scripts/clean_mbox.py (quote/signature/boilerplate
stripping, position-aware boilerplate detection). Generalized so author
addresses are per-job, not hardcoded.

Returns: list of {"source", "text"} — plain writing samples ready to curate.
"""
from __future__ import annotations

import hashlib
import mailbox
import re
from email.utils import getaddresses
from html.parser import HTMLParser
from pathlib import Path

MIN_CHARS = 50
BOILERPLATE_MIN_FREQ = 4
MAX_SAMPLE_CHARS = 4000  # cap very long docs so synth stays focused


# ── HTML → text ────────────────────────────────────────────────────────────
class _HTMLText(HTMLParser):
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


def _html_to_text(html: str) -> str:
    p = _HTMLText()
    p.feed(html)
    return p.text()


# ── email body extraction ──────────────────────────────────────────────────
def _get_body(msg) -> str:
    plain, html = "", ""
    if msg.is_multipart():
        for part in msg.walk():
            ctype = part.get_content_type()
            payload = part.get_payload(decode=True)
            if not payload:
                continue
            decoded = payload.decode(part.get_content_charset() or "utf-8",
                                     errors="replace")
            if ctype == "text/plain" and not plain:
                plain = decoded
            elif ctype == "text/html" and not html:
                html = decoded
    else:
        payload = msg.get_payload(decode=True)
        if payload:
            decoded = payload.decode(msg.get_content_charset() or "utf-8",
                                     errors="replace")
            if msg.get_content_type() == "text/html":
                html = decoded
            else:
                plain = decoded
    return plain if plain.strip() else (_html_to_text(html) if html.strip() else "")


def _author_address(msg) -> str:
    addrs = getaddresses([msg.get("From", "")])
    if addrs and addrs[0][1]:
        return addrs[0][1].lower().strip()
    return ""


# ── cleaning ───────────────────────────────────────────────────────────────
_FORWARD_SEPARATORS = [
    re.compile(r"^-{4,}\s*$", re.M),
    re.compile(r"^On .{0,200}wrote:\s*$", re.M),
    re.compile(r"^Begin forwarded message:\s*$", re.M),
    re.compile(r"^-{1,3} (Original|Forwarded) Message -{1,3}\s*$", re.M),
    # Non-English quote headers (e.g. French "Le 12 juin ... a écrit :")
    re.compile(r"^.{0,200}a écrit\s*:\s*$", re.M),
    re.compile(r"^.{0,200}schrieb\s*:\s*$", re.M),
]
_FOOTER_PHRASES = re.compile(
    r"^(sent from my|get (outlook|gmail) for|confidentiality notice|"
    r"this e-?mail .* confidential|disclaimer:|— ?sent from|"
    r"CAUTION: This email)",
    re.I,
)
_URL_BRACKET = re.compile(r"\[https?://\S+\]")
_TEL_MAILTO = re.compile(r"<(tel|mailto):[^>]+>")
_BARE_URL = re.compile(r"https?://\S+(?=\s|$)")


def _cut_at_first(text: str, patterns) -> str:
    cut = len(text)
    for pat in patterns:
        m = pat.search(text)
        if m:
            cut = min(cut, m.start())
    return text[:cut]


def _strip_inline(line: str) -> str:
    line = _TEL_MAILTO.sub("", line)
    line = _URL_BRACKET.sub("", line)
    line = _BARE_URL.sub("", line)
    return line


def _clean_body(raw: str, boilerplate: set[str]) -> str:
    text = _cut_at_first(raw, _FORWARD_SEPARATORS)
    kept: list[str] = []
    for line in text.splitlines():
        s = line.rstrip()
        if s.lstrip().startswith(">"):
            continue
        if _FOOTER_PHRASES.match(s.strip()):
            break
        if s.strip() in boilerplate:
            continue
        kept.append(s)
    while kept and not kept[-1].strip():
        kept.pop()
    cleaned = [_strip_inline(l) for l in kept]
    result = "\n".join(cleaned).strip()
    return re.sub(r"\n{3,}", "\n\n", result)


def _build_boilerplate(records: list[dict]) -> set[str]:
    import collections
    leading = collections.Counter()
    trailing = collections.Counter()
    for r in records:
        text = _cut_at_first(r["_raw"], _FORWARD_SEPARATORS)
        non_empty = [l.strip() for l in text.splitlines()
                     if l.strip() and not l.lstrip().startswith(">")]
        if not non_empty:
            continue
        mid = len(non_empty) // 2
        leading.update(set(non_empty[:mid]))
        trailing.update(set(non_empty[mid:]))
    bp = set()
    for line, t_count in trailing.items():
        if t_count >= BOILERPLATE_MIN_FREQ and t_count >= 2 * leading.get(line, 0):
            bp.add(line)
    return bp


# ── text/doc chunking ──────────────────────────────────────────────────────
def _chunk_text(text: str, max_chars: int = 1800) -> list[str]:
    """Split a long free-form doc into paragraph-grouped chunks."""
    paras = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
    chunks, cur = [], ""
    for p in paras:
        if len(cur) + len(p) + 2 > max_chars and cur:
            chunks.append(cur)
            cur = p
        else:
            cur = f"{cur}\n\n{p}".strip() if cur else p
    if cur.strip():
        chunks.append(cur)
    return chunks or [text[:max_chars]]


# ── entry point ────────────────────────────────────────────────────────────
def ingest(input_dir: Path, author_addresses: set[str]) -> list[dict]:
    """Parse all files in input_dir into cleaned author samples."""
    email_records: list[dict] = []
    doc_samples: list[dict] = []

    for path in sorted(input_dir.iterdir()):
        suf = path.suffix.lower()
        if suf in (".mbox",):
            box = mailbox.mbox(str(path))
            for msg in box:
                if _author_address(msg) not in author_addresses:
                    continue
                body = _get_body(msg)
                if body.strip():
                    email_records.append({"source": path.name, "_raw": body})
        elif suf == ".eml":
            with path.open("rb") as f:
                msg = mailbox.mboxMessage(f)
            if _author_address(msg) in author_addresses:
                body = _get_body(msg)
                if body.strip():
                    email_records.append({"source": path.name, "_raw": body})
        elif suf in (".txt", ".md"):
            raw = path.read_text(encoding="utf-8", errors="replace")
            for chunk in _chunk_text(raw):
                doc_samples.append({"source": path.name, "_raw": chunk})

    # Clean email records (boilerplate-aware) + dedup
    boilerplate = _build_boilerplate(email_records)
    samples: list[dict] = []
    seen: set[str] = set()
    for r in email_records + doc_samples:
        text = _clean_body(r["_raw"], boilerplate)
        if len(text) < MIN_CHARS:
            continue
        text = text[:MAX_SAMPLE_CHARS]
        h = hashlib.sha256(text.encode()).hexdigest()
        if h in seen:
            continue
        seen.add(h)
        samples.append({"source": r["source"], "text": text})
    return samples
