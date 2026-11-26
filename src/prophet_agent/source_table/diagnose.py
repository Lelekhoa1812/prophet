"""Classify a failed download. No network.

Motivation vs Logic
Motivation: a refused file is a signal, not a reason to store a different series under the requested name.
Logic: the status in the error text wins over the words around it. HTML is a page to read, not a spreadsheet type.
"""
from __future__ import annotations

import re

_AUTH = re.compile(r"\b40[13]\b")
_MISSING = re.compile(r"\b(?:404|410)\b")
_HTML = re.compile(r"text/html|application/xhtml|the body is html", re.I)
_NOT_TABLE = re.compile(r"no data rows|not a table|does not hold a table|numeric", re.I)


def diagnose_failure(message: str, *, content_type: str = "") -> dict:
    """One signal for a download error or a content type. The message is the tool error text."""
    text = message if isinstance(message, str) else ""
    media = content_type if isinstance(content_type, str) else ""
    blob = f"{text}\n{media}"
    if _AUTH.search(blob):
        signal = "auth_denied"
    elif _MISSING.search(blob):
        signal = "not_found"
    elif _HTML.search(blob):
        signal = "html_page"
    elif _NOT_TABLE.search(text):
        signal = "not_a_table"
    else:
        signal = "unsupported_type"
    return {"signal": signal, "error": text[:300], "content_type": media[:120]}
