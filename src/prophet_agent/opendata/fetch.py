"""Download one open-data URL with the address, redirect, type and size rules applied. Never reads a key or a cookie.

Motivation vs Logic
Motivation: the platform downloads public series without a person in the loop. Each download must be bounded in size and
time, must not be steered to an internal address by a redirect, and must hand back only data the agent can parse.
Logic: each hop goes through netguard, which checks the scheme, port and name and then connects to a checked address
(open_pinned), so a DNS change cannot move the connection. Redirects are followed by hand, at most MAX_REDIRECTS, with
the same limit the research fetch uses. The body is streamed and cut at max_bytes. CSV, plain text, JSON and SDMX CSV
are accepted. A ZIP or XLSX body is accepted as bytes and turned into a table later. An octet-stream body is accepted
only when it parses as CSV or JSON, or when it is a ZIP. The one network seam is open_client(), which tests replace,
so the suite never touches the network.
"""
from __future__ import annotations

import csv
import io
import json
import hashlib
import time
from datetime import datetime, timezone
from urllib.parse import urljoin

import httpx

from prophet_agent.opendata.csvlines import table_lines
from prophet_agent.research.fetch import MAX_REDIRECTS, REDIRECT_CODES
from prophet_agent.research.netguard import Resolver, UnsafeURL, open_pinned, read_capped

USER_AGENT = "ProphetOpenData/1.0"
DEFAULT_MAX_BYTES = 25 * 1024 * 1024
TEXT_TYPES = ("text/csv", "application/csv", "text/plain", "application/json", "application/vnd.sdmx.data+csv")
# A generic binary type is accepted only when the body parses as CSV or JSON, or when it is a ZIP (an XLSX is a ZIP).
# The ECB SDMX csvdata endpoint answers text/csv. OECD and ABS answer the SDMX CSV type above.
SNIFF_TYPES = ("application/octet-stream",)
# An HTML page is accepted; its data tables are landed by html_landing, not returned as the body.
HTML_TYPES = ("text/html", "application/xhtml+xml")
# A PDF is accepted when its body starts with %PDF-; its tables are landed by pdf_landing, not returned as the body.
PDF_TYPES = ("application/pdf",)
CONTAINER_TYPES = (
    "application/zip",
    "application/x-zip-compressed",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
)


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
        got = _follow(url, client=client, max_bytes=max_bytes, timeout_s=timeout_s, deadline=deadline,
                      resolver=resolver)
    finally:
        client.close()
    record_download(url, got.get("final_url") or url, got["bytes"], got.get("retrieved_at", ""))
    return got


def download_record_path(url: str):
    """Where the fetch record of a downloaded URL is kept: beside the research cache's page records."""
    from prophet_agent.research.search import cache_key, cache_root
    return cache_root() / "downloads" / f"{cache_key('download', url)}.json"


def record_download(url: str, final_url: str, body: bytes, retrieved_at: str) -> None:
    """Write a fetch record for a download that answered 200 with an accepted body.

    Motivation: panel GISTEMP round 2 and NEM round 1 (2026-11-26): a series landed by download_web_data named its URL as
    its source, and backtest_series refused it ("has no fetch record, so the research did not read it"), because only
    web research wrote records. Logic: the record holds the URL, final URL, sha256 and time; a write failure is ignored,
    since the download itself succeeded.
    """
    doc = {"ok": True, "url": url, "final_url": final_url, "sha256": hashlib.sha256(body).hexdigest(),
           "retrieved_at": retrieved_at}
    for key in dict.fromkeys((url, final_url)):
        try:
            path = download_record_path(key)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(doc), encoding="utf-8")
        except OSError:
            pass


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
    if media not in TEXT_TYPES and media not in SNIFF_TYPES and media not in CONTAINER_TYPES and media not in HTML_TYPES \
            and media not in PDF_TYPES:
        raise DownloadError(f"content type {media or '(none)'} is not allowed; use CSV, plain text, JSON, HTML, PDF, ZIP "
                            "or XLSX")
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
    if media in CONTAINER_TYPES or (media in SNIFF_TYPES and body[:2] == b"PK"):
        if body[:2] != b"PK":
            raise DownloadError(f"content type {media} is accepted only when the body is a ZIP or XLSX")
        return {"bytes": body, "content_type": media, "final_url": final_url, "retrieved_at": _now()}
    if media in PDF_TYPES:
        if body[:5] != b"%PDF-":
            raise DownloadError("content type application/pdf is accepted only when the body starts with %PDF-")
        return {"bytes": body, "content_type": media, "final_url": final_url, "retrieved_at": _now()}
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
    # Comment lines are dropped by the same rule the store's preview uses (csvlines), so they do not count as the header.
    lines = table_lines(text)
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
