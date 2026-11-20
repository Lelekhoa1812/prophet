"""NIM transport errors. Messages are always passed through `redact` so a key can never leak through an exception."""
from __future__ import annotations

import re

_KEY = re.compile(r"nvapi-[A-Za-z0-9_\-]+")


def redact(text: object, limit: int = 300) -> str:
    """Replace anything shaped like an NVIDIA API key and cap the length."""
    return _KEY.sub("[key]", str(text))[:limit]


class NimError(RuntimeError):
    """Base class. `str(err)` is already redacted."""

    def __init__(self, message: object):
        super().__init__(redact(message))


class NotConfigured(NimError):
    """No API key is configured (names only are ever reported, never values)."""


class Exhausted(NimError):
    """Every key and model in the chain failed, or the wall-clock budget ran out."""


class SchemaViolation(NimError):
    """The model answered but its JSON does not satisfy the caller's schema."""
