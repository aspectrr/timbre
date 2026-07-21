"""API-key auth.

Keys are random 32-byte hex tokens. Only sha256(key) is stored; the plaintext
is returned to the caller exactly once at creation. Every request hashes its
Bearer token and looks the hash up in `api_keys`.
"""
from __future__ import annotations

import hashlib
import secrets

from fastapi import Request

import status


class UnauthorizedError(Exception):
    """Raised by verify_key on a missing/invalid Bearer token; main maps it
    to a 401 `{"error":"unauthorized"}` body (never leaks whether the key
    exists)."""


def hash_key(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()


def new_key() -> tuple[str, str]:
    """Return (plaintext_key, key_hash). The plaintext is shown once."""
    key = secrets.token_hex(32)  # 32 random bytes -> 64 hex chars
    return key, hash_key(key)


async def verify_key(request: Request) -> str:
    """FastAPI dependency: resolve the Bearer token to an owner key_hash.

    Missing/invalid -> UnauthorizedError (401). On success the owner is
    attached to request.state.owner for handlers and returned for Depends()."""
    header = request.headers.get("authorization", "")
    token = header[7:].strip() if header[:7].lower() == "bearer " else ""
    owner = None
    if token:
        row = status.lookup_key(hash_key(token))
        if row:
            owner = row["key_hash"]
    if not owner:
        raise UnauthorizedError()
    request.state.owner = owner
    return owner
