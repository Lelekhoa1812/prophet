"""Guarded GET that returns the raw body of a page or image, for chart extraction.

Motivation vs Logic
Motivation: chart extraction needs the page's own markup (its scripts and SVG), and chart images need their bytes. The text
fetcher in fetch.py returns extracted text only, and it is not edited here.
Logic: the same guards as fetch.py, from netguard: syntax check, a pinned connection to a checked public address on each
hop, redirects followed by hand (at most MAX_REDIRECTS), the host throttle, a total deadline and a byte cap. The media type
must be on the caller's list. Never raises; a refusal returns ok False with a note.
"""
from __future__ import annotations

import time
from urllib.parse import urljoin

import httpx

from prophet_agent.research.fetch import HEADERS, REDIRECT_CODES
from prophet_agent.research.netguard import (DEFAULT_THROTTLE, LookupTimeout, UnsafeURL, check_syntax, open_pinned,
                                             read_capped)

MAX_REDIRECTS = 4
TOTAL_SECONDS = 45.0
HTML_MEDIA = ("text/html", "application/xhtml+xml")


def fetch_bytes(url: str, *, max_bytes: int, media: tuple[str, ...], timeout: float = 15.0,
                client: httpx.Client | None = None, resolver=None, throttle=None,
                total_seconds: float = TOTAL_SECONDS) -> dict:
    """Returns {"ok", "status", "media", "body", "final_url", "note"}. body is bytes, empty when not ok."""
    def refused(note: str, final: str = url, status: int | None = None) -> dict:
        return {"ok": False, "status": status, "media": "", "body": b"", "final_url": final, "note": note}

    if not isinstance(url, str) or not url.strip():
        return refused("the URL is empty")
    url = url.strip()
    try:
        check_syntax(url)
    except UnsafeURL as exc:
        return refused(f"refused: {exc}")
    own = client is None
    http = client or httpx.Client(follow_redirects=False, timeout=timeout)
    deadline = time.monotonic() + total_seconds
    try:
        current = url
        for _ in range(MAX_REDIRECTS + 1):
            if time.monotonic() >= deadline:
                return refused(f"no complete response within {total_seconds:g} seconds", current)
            try:
                check_syntax(current)
            except UnsafeURL as exc:
                return refused(f"refused: {exc}", current)
            if (throttle or DEFAULT_THROTTLE).wait(_host(current), deadline=deadline) is None:
                return refused("the host is busy past the deadline", current)
            try:
                response = open_pinned(http, current, resolver=resolver, headers=HEADERS,
                                       timeout=min(timeout, max(deadline - time.monotonic(), 0.001)), deadline=deadline)
            except UnsafeURL as exc:
                return refused(f"refused: {exc}", current)
            except LookupTimeout:
                return refused("the DNS lookup did not finish before the deadline", current)
            except httpx.HTTPError as exc:
                return refused(f"fetch failed: {type(exc).__name__}", current)
            try:
                if response.status_code in REDIRECT_CODES and response.headers.get("location"):
                    current = urljoin(current, response.headers["location"])
                    continue
                if response.status_code != 200:
                    return refused(f"HTTP {response.status_code}", current, response.status_code)
                media = response.headers.get("content-type", "").split(";", 1)[0].strip().lower()
                if media not in media:
                    return refused(f"the media type {media or 'unknown'} is not read here", current, 200)
                body, stop = read_capped(response, max_bytes, deadline)
                if stop:
                    return refused(f"the body was cut ({stop})", current, 200)
                if not body:
                    return refused("the body is empty", current, 200)
                return {"ok": True, "status": 200, "media": media, "body": body, "final_url": current, "note": ""}
            finally:
                response.close()
        return refused(f"more than {MAX_REDIRECTS} redirects", current)
    except Exception as exc:                       # the caller always gets an answer
        return refused(f"fetch failed: {type(exc).__name__}")
    finally:
        if own:
            http.close()


def _host(url: str) -> str:
    from urllib.parse import urlsplit
    return urlsplit(url).netloc.lower()
