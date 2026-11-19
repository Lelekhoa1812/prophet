"""Stable pseudonyms. Codes are derived from a local salt, so re-extracts give the same codes."""
from __future__ import annotations

import hashlib
import secrets

from ..config import DATA

_SALT_FILE = DATA / ".salt"


def _salt() -> bytes:
    if not _SALT_FILE.exists():
        _SALT_FILE.write_text(secrets.token_hex(16))
        _SALT_FILE.chmod(0o600)
    return _SALT_FILE.read_text().encode()


def stable_hash(kind: str, real_id: str) -> str:
    return hashlib.sha256(_salt() + kind.encode() + b":" + str(real_id).encode()).hexdigest()


class CodeBook:
    """Assigns C-0001 style codes ordered by hash, so codes carry no real ordering."""

    def __init__(self, kind: str, prefix: str, ids: list[str]):
        ordered = sorted({str(i) for i in ids}, key=lambda i: stable_hash(kind, i))
        self.kind, self.prefix = kind, prefix
        self.codes = {i: f"{prefix}-{n + 1:04d}" for n, i in enumerate(ordered)}

    def __call__(self, real_id: str) -> str:
        return self.codes[str(real_id)]
