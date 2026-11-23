"""NIM transport errors. Messages are always passed through `redact` so a key can never leak through an exception."""
from __future__ import annotations

import re

_KEY = re.compile(r"nvapi-[A-Za-z0-9_\-]+")
# The other providers' key formats: OpenAI and OpenRouter sk-, Anthropic sk-ant-, Google AIza, xAI xai-.
_VENDOR_KEY = re.compile(r"\b(?:sk-[A-Za-z0-9_\-]{8,}|AIza[0-9A-Za-z_\-]{20,}|xai-[A-Za-z0-9_\-]{8,})")
_BEARER = re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/=\-]{8,}")
# A key-like name with a value, as a query or form pair, or as JSON. ``key=`` counts only after ? or &, so ordinary
# code such as ``sorted(x, key=len)`` in sandbox output is left alone.
_NAMED = re.compile(r"(?i)\b(api[_-]?key|apikey|access[_-]?token|auth[_-]?token|token|secret|password)=[^\s&\"',;}\]]{8,}")
_QUERY_KEY = re.compile(r"(?i)(?<=[?&])key=[^\s&\"',;}\]]{8,}")
_NAMED_JSON = re.compile(r"(?i)\"(api[_-]?key|apikey|access[_-]?token|auth[_-]?token|token|secret|password)\"\s*:\s*\"[^\"]+\"")


def scrub_keys(text: str, mark: str = "[key]") -> str:
    """Replace key-shaped text: NVIDIA keys, the other providers' key formats and Bearer tokens. Used on code and files,
    where a name=value pair may be ordinary code, so those are left as they are."""
    text = _KEY.sub(mark, text)
    text = _VENDOR_KEY.sub(mark, text)
    return _BEARER.sub(f"Bearer {mark}", text)


def redact(text: object, limit: int = 300) -> str:
    """Replace anything shaped like a key (see scrub_keys), and a key-like name=value pair, and cap the length."""
    out = scrub_keys(str(text))
    out = _NAMED.sub(lambda m: f"{m.group(1)}=[key]", out)
    out = _QUERY_KEY.sub("key=[key]", out)
    out = _NAMED_JSON.sub(lambda m: f'"{m.group(1)}": "[key]"', out)
    return out[:limit]


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
