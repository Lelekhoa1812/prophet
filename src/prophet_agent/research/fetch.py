"""Fetch one web page as readable text, with the address, redirect, robots and size rules applied, and never raise.

Motivation vs Logic
Motivation: search snippets are too thin to quote from. The agent needs the body of the page, the title and date, and a
fingerprint of the exact text it read, so that a quotation can be checked later. Pages are also the riskiest input the
agent touches, so they are fetched in the most restricted way possible.
Logic: a plain GET with httpx, never JavaScript, never a form, never a file:// URL. Each redirect hop is checked by
netguard and paced per host; robots.txt is honoured; only text, HTML and JSON are read (a PDF is refused with a note).
The body is read up to a byte cap. Text is extracted with trafilatura, and BeautifulSoup takes over when trafilatura
returns nothing. The stored text is clipped and fingerprinted with SHA-256, so the citation can be verified against it.
Every failure becomes a Page with ``ok`` False and a plain note; the function does not raise.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, fields
from datetime import datetime, timezone
from urllib.parse import urljoin

import httpx
import trafilatura
from bs4 import BeautifulSoup

from prophet_agent.research.netguard import (DEFAULT_ROBOTS, DEFAULT_THROTTLE, USER_AGENT, HostThrottle, Resolver,
                                             Robots, UnsafeURL, check_url)
from prophet_agent.research.search import cache_key, cache_root, read_cache, write_cache

MAX_REDIRECTS = 4
MAX_TEXT_CHARS = 60_000
CACHE_TTL_S = 24 * 3600
ALLOWED_MEDIA = ("text/html", "application/xhtml+xml", "text/plain", "application/json")
REDIRECT_CODES = (301, 302, 303, 307, 308)
HEADERS = {"User-Agent": USER_AGENT,
           "Accept": "text/html,application/xhtml+xml,text/plain;q=0.9,application/json;q=0.8"}
_CHARSET = re.compile(r"charset=([\w.:-]+)", re.I)


@dataclass
class Page:
    url: str
    final_url: str
    title: str = ""
    date: str | None = None
    text: str = ""
    sha256: str = ""
    fetched_at: str = ""
    status: int | None = None
    note: str = ""
    ok: bool = False
    cached: bool = False


PAGE_FIELDS = {f.name for f in fields(Page)}


def _failed(url: str, note: str, *, final_url: str | None = None, status: int | None = None) -> Page:
    return Page(url=url, final_url=final_url or url, status=status, note=note, ok=False,
                fetched_at=_now())


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _media_and_charset(content_type: str) -> tuple[str, str | None]:
    media = content_type.split(";", 1)[0].strip().lower()
    match = _CHARSET.search(content_type or "")
    return media, (match.group(1) if match else None)


def _decode(body: bytes, charset: str | None) -> str:
    for name in (charset, "utf-8"):
        if not name:
            continue
        try:
            return body.decode(name, errors="replace")
        except LookupError:
            continue
    return body.decode("utf-8", errors="replace")


def _bs4_text(html: str) -> tuple[str, str]:
    """Visible text from the article, main or body element; also returns the <title>."""
    soup = BeautifulSoup(html, "lxml")
    title = " ".join((soup.title.get_text() if soup.title else "").split())
    for tag in soup(["script", "style", "noscript", "template"]):
        tag.decompose()
    root = soup.find("article") or soup.find("main") or soup.body or soup
    lines = [" ".join(line.split()) for line in root.get_text("\n").splitlines()]
    return "\n".join(line for line in lines if line), title


def extract(body: str, media: str) -> tuple[str, str, str | None, str]:
    """Return (text, title, date, method) from a decoded body."""
    if media == "application/json":
        try:
            return json.dumps(json.loads(body), indent=1, ensure_ascii=False), "", None, "json"
        except ValueError:
            return body, "", None, "raw"
    if media == "text/plain":
        return body, "", None, "plain"
    text = trafilatura.extract(body, include_comments=False, include_tables=True, favor_recall=True,
                               with_metadata=False) or ""
    title, date = "", None
    try:
        meta = trafilatura.extract_metadata(body)
        if meta is not None:
            title = meta.title or ""
            date = meta.date or None
    except Exception:
        pass
    method = "trafilatura"
    fallback_title = ""
    if not text.strip():
        text, fallback_title = _bs4_text(body)
        method = "bs4"
    return text, title or fallback_title, date, method


def _get_checked(client: httpx.Client, url: str, *, resolver: Resolver | None, throttle: HostThrottle,
                 timeout: float):
    """One GET hop after the address check and pacing. Returns a streaming response; the caller reads it."""
    parts = check_url(url, resolver=resolver)
    throttle.wait(parts.netloc.lower())
    return client.stream("GET", url, headers=HEADERS, timeout=timeout)


def fetch_page(url: str, *, max_bytes: int = 2_000_000, timeout: float = 15.0, use_cache: bool = True,
               cache_dir=None, client: httpx.Client | None = None, resolver: Resolver | None = None,
               robots: Robots | None = None, throttle: HostThrottle | None = None) -> Page:
    """Fetch and extract one page. Never raises; a refused or failed fetch returns Page(ok=False, note=...)."""
    try:
        return _fetch(url, max_bytes=max_bytes, timeout=timeout, use_cache=use_cache, cache_dir=cache_dir,
                      client=client, resolver=resolver, robots=robots, throttle=throttle)
    except Exception as exc:   # last line of defence: the research tools must always get an answer
        return _failed(url, f"fetch failed: {type(exc).__name__}")


def _fetch(url, *, max_bytes, timeout, use_cache, cache_dir, client, resolver, robots, throttle) -> Page:
    if not isinstance(url, str) or not url.strip():
        return _failed(url or "", "the URL is empty")
    url = url.strip()
    try:
        check_url(url, resolver=resolver)
    except UnsafeURL as exc:
        return _failed(url, f"refused: {exc}")

    path = (cache_dir or cache_root()) / "pages" / f"{cache_key('page', url)}.json"
    if use_cache:
        cached = read_cache(path, CACHE_TTL_S)
        if cached and cached.get("ok"):
            return Page(**{**{k: v for k, v in cached.items() if k in PAGE_FIELDS}, "cached": True})

    own_client = client is None
    http = client or httpx.Client(follow_redirects=False, timeout=timeout)
    robots = robots or (Robots(client=client, resolver=resolver) if client or resolver else DEFAULT_ROBOTS)
    throttle = throttle or DEFAULT_THROTTLE
    try:
        if not robots.allowed(url):
            return _failed(url, "disallowed by robots.txt")
        current = url
        for _ in range(MAX_REDIRECTS + 1):
            try:
                opened = _get_checked(http, current, resolver=resolver, throttle=throttle, timeout=timeout)
            except UnsafeURL as exc:   # a redirect hop to a forbidden address is refused, not followed
                return _failed(url, f"refused: {exc}", final_url=current)
            with opened as response:
                if response.status_code in REDIRECT_CODES and response.headers.get("location"):
                    current = urljoin(current, response.headers["location"])
                    continue
                return _read_response(url, current, response, max_bytes=max_bytes, use_cache=use_cache, path=path)
        return _failed(url, f"more than {MAX_REDIRECTS} redirects", final_url=current)
    finally:
        if own_client:
            http.close()


def _read_response(url: str, final_url: str, response: httpx.Response, *, max_bytes: int, use_cache: bool,
                   path) -> Page:
    status = response.status_code
    if status >= 400:
        return _failed(url, f"HTTP {status}", final_url=final_url, status=status)
    media, charset = _media_and_charset(response.headers.get("content-type", ""))
    if media == "application/pdf":
        return _failed(url, "PDF files are not read; look for the HTML version or a text source",
                       final_url=final_url, status=status)
    if media and media not in ALLOWED_MEDIA:
        return _failed(url, f"content type {media} is not text; only HTML, plain text and JSON are read",
                       final_url=final_url, status=status)
    body = bytearray()
    truncated = False
    for chunk in response.iter_bytes():
        room = max_bytes - len(body)
        if len(chunk) > room:
            body.extend(chunk[:max(room, 0)])
            truncated = True
            break
        body.extend(chunk)
    notes = [f"response cut at {max_bytes} bytes"] if truncated else []
    decoded = _decode(bytes(body), charset)
    text, title, date, method = extract(decoded, media or "text/html")
    if method == "bs4":
        notes.append("main text taken by fallback extraction")
    text = text.strip()
    if not text:
        return _failed(url, "no readable text found", final_url=final_url, status=status)
    if len(text) > MAX_TEXT_CHARS:
        text = text[:MAX_TEXT_CHARS]
        notes.append(f"text clipped to {MAX_TEXT_CHARS} characters")
    page = Page(url=url, final_url=final_url, title=" ".join(title.split())[:300], date=date, text=text,
                sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(), fetched_at=_now(), status=status,
                note="; ".join(notes), ok=True)
    if use_cache:
        write_cache(path, asdict(page))
    return page
