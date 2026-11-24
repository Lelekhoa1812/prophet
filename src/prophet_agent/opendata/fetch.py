"""Download one open-data URL with the address, redirect, type and size rules applied. Never reads a key or a cookie.

Motivation vs Logic
Motivation: the platform downloads public series without a person in the loop. Each download must be bounded in size and
time, must not be steered to an internal address by a redirect, and must hand back only data the agent can parse.
Logic: each hop goes through netguard, which checks the scheme, port and name and then connects to a checked address
(open_pinned), so a DNS change cannot move the connection. Redirects are followed by hand, at most MAX_REDIRECTS, with
the same limit the research fetch uses. The body is streamed and cut at max_bytes. Only CSV, plain text and JSON types
are accepted; an octet-stream body is accepted only when it parses as CSV or JSON. The one network seam is open_client(),
which tests replace, so the suite never touches the network.
"""
from __future__ import annotations

import csv
import io
import json
import time
from datetime import datetime, timezone
from urllib.parse import urljoin

import httpx

from prophet_agent.research.fetch import MAX_REDIRECTS, REDIRECT_CODES
from prophet_agent.research.netguard import Resolver, UnsafeURL, open_pinned, read_capped

USER_AGENT = "ProphetOpenData/1.0"
DEFAULT_MAX_BYTES = 25 * 1024 * 1024
TEXT_TYPES = ("text/csv", "application/csv", "text/plain", "application/json")
# A generic binary type is accepted only when the body parses as CSV or JSON. The ECB SDMX csvdata endpoint answers
# text/csv. FRED's graph CSV answered application/csv on 2026-11-24, so that type is listed too.
SNIFF_TYPES = ("application/octet-stream",)


class DownloadError(ValueError):
    """A download that was refused or failed. The message is safe to show the planner."""


def open_client(timeout_s: float) -> httpx.Client:
    """The one network seam: every hop is sent through the client this returns. Tests replace this function."""
    return httpx.Client(follow_redirects=False, timeout=timeout_s, headers={"User-Agent": USER_AGENT})


def media_type(content_type: str) -> str:
    return (content_type or "").split(";", 1)[0].strip().lower()


def download(url: str, *, max_bytes: int = DEFAULT_MAX_BYTES, timeout_s: float = 45,
             resolver: Resolver | None = None) -> dict:
    """Fetch url and return {bytes, content_type, final_url, retrieved_at}. Raises DownloadError when the download is
    refused or fails. ``resolver(host, port) -> [ip, ...]`` is injectable so tests never resolve a name."""
    if not isinstance(max_bytes, int) or isinstance(max_bytes, bool) or max_bytes <= 0:
        raise ValueError("max_bytes must be a positive whole number")
    if timeout_s <= 0:
        raise ValueError("timeout_s must be positive")
    deadline = time.monotonic() + timeout_s
    client = open_client(timeout_s)
    try:
        return _follow(url, client=client, max_bytes=max_bytes, timeout_s=timeout_s, deadline=deadline,
                       resolver=resolver)
    finally:
        client.close()


def _follow(url, *, client: httpx.Client, max_bytes: int, timeout_s: float, deadline: float,
            resolver: Resolver | None) -> dict:
    current = url
    for _ in range(MAX_REDIRECTS + 1):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise DownloadError(f"no complete response within {timeout_s:g} seconds")
        try:
            response = open_pinned(client, current, resolver=resolver, headers=None, timeout=remaining,
                                   deadline=deadline)
        except UnsafeURL as exc:
            raise DownloadError(f"refused: {exc}") from None
        except httpx.HTTPError as exc:
            raise DownloadError(f"transport error: {type(exc).__name__}") from None
        try:
            if response.status_code in REDIRECT_CODES and response.headers.get("location"):
                current = urljoin(current, response.headers["location"])   # a hop: checked again on the next pass
                continue
            return _read(current, response, max_bytes=max_bytes, deadline=deadline)
        finally:
            response.close()
    raise DownloadError(f"more than {MAX_REDIRECTS} redirects")


def _read(final_url: str, response: httpx.Response, *, max_bytes: int, deadline: float) -> dict:
    if response.status_code != 200:
        raise DownloadError(f"HTTP {response.status_code}; only a 200 response is used")
    media = media_type(response.headers.get("content-type", ""))
    if media not in TEXT_TYPES and media not in SNIFF_TYPES:
        raise DownloadError(f"content type {media or '(none)'} is not allowed; use CSV, plain text or JSON")
    declared = response.headers.get("content-length", "")
    if declared.isdigit() and int(declared) > max_bytes:
        raise DownloadError(f"the response is {int(declared)} bytes; the limit is {max_bytes}")
    body, stopped = read_capped(response, max_bytes, deadline)
    if stopped == "size":
        raise DownloadError(f"the response is larger than {max_bytes} bytes")
    if stopped == "time":
        raise DownloadError("the response did not finish in time")
    if not body:
        raise DownloadError("the response body is empty")
    text = _decoded(body)
    if media in SNIFF_TYPES:
        if text is None or not (_csv_like(text) or _json_like(text)):
            raise DownloadError(f"content type {media} is accepted only when the body parses as CSV or JSON")
    elif text is None:
        raise DownloadError("the body is not UTF-8 text")
    return {"bytes": body, "content_type": media, "final_url": final_url, "retrieved_at": _now()}


def _decoded(body: bytes) -> str | None:
    try:
        text = body.decode("utf-8-sig")
    except UnicodeDecodeError:
        return None
    return None if "\x00" in text else text


def _csv_like(text: str) -> bool:
    # Lines starting with '#' are comments, as in the store's preview, so they do not count as the header.
    lines = [line for line in text.splitlines() if not line.startswith("#")]
    try:
        rows = [row for row in csv.reader(lines) if any(cell.strip() for cell in row)]
    except csv.Error:
        return False
    return bool(rows) and len(rows[0]) >= 2


def _json_like(text: str) -> bool:
    try:
        json.loads(text)
    except (ValueError, RecursionError):   # too deeply nested to parse is not JSON we can use
        return False
    return True


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")
