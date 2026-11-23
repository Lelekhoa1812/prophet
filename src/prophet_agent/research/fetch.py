"""Fetch one web page as readable text, with the address, redirect, robots and size rules applied, and never raise.

Motivation vs Logic
Motivation: search snippets are too thin to quote from. The agent needs the body of the page, the title and date, and a
fingerprint of the exact text it read, so that a quotation can be checked later. Pages are also the riskiest input the
agent touches, so they are fetched in the most restricted way possible.
Logic: a plain GET with httpx, never JavaScript, never a form, never a file:// URL. Each redirect hop is checked by
netguard and paced per host; robots.txt is honoured; only text, HTML and JSON are read (a PDF is refused with a note).
The body is read up to a byte cap. Text is extracted with trafilatura, and BeautifulSoup takes over when trafilatura
returns nothing. The stored text is clipped and fingerprinted with SHA-256, so the citation can be verified against it.
Dates need care: an index or calendar page carries template dates (a year start, a "last updated" stamp), so its date is
trusted only when the page's own metadata gives it. Elsewhere a date comes from the metadata first (article:published_time,
JSON-LD datePublished, a time element, or trafilatura's date), and a date in the URL path is a weaker fallback. Each date
carries ``date_source``: "meta", "url" or "none".
Every failure becomes a Page with ``ok`` False and a plain note; the function does not raise.
"""
from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import asdict, dataclass, fields
from datetime import datetime, timezone
from urllib.parse import urljoin, urlsplit

import httpx
import trafilatura
from bs4 import BeautifulSoup

from prophet_agent.research.netguard import (DEFAULT_ROBOTS, DEFAULT_THROTTLE, USER_AGENT, HostThrottle, LookupTimeout,
                                             Resolver, Robots, UnsafeURL, check_syntax, open_pinned, read_capped)
from prophet_agent.research.rank import parse_date
from prophet_agent.research.search import cache_key, cache_root, read_cache, write_cache

MAX_REDIRECTS = 4
TOTAL_SECONDS = 45.0   # the whole fetch: every redirect hop and the body. Each socket wait is also bounded by timeout.
MAX_TEXT_CHARS = 60_000
CACHE_TTL_S = 24 * 3600
ALLOWED_MEDIA = ("text/html", "application/xhtml+xml", "text/plain", "application/json")
HTML_MEDIA = ("text/html", "application/xhtml+xml")
REDIRECT_CODES = (301, 302, 303, 307, 308)
HEADERS = {"User-Agent": USER_AGENT,
           "Accept": "text/html,application/xhtml+xml,text/plain;q=0.9,application/json;q=0.8",
           "Accept-Encoding": "gzip, deflate"}   # only the codings read_capped decodes, so no brotli or zstd arrives
_CHARSET = re.compile(r"charset=([\w.:-]+)", re.I)
INDEX_PATH = re.compile(r"calendar|schedule|archive", re.I)
INDEX_FILE = re.compile(r"^(index|default)\.[a-z]{3,4}$", re.I)
URL_DATE = re.compile(r"(?<!\d)((?:19|20)\d{2})[/-](0[1-9]|1[0-2])[/-](0[1-9]|[12]\d|3[01])(?!\d)")
META_DATE_KEYS = {"article:published_time", "datepublished"}


@dataclass
class Page:
    url: str
    final_url: str
    title: str = ""
    date: str | None = None
    date_source: str = "none"
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


def _is_wide_text(body: bytes, charset: str | None) -> bool:
    """UTF-16 and UTF-32 text has NUL bytes by design; it is text when declared or marked by a byte order mark."""
    declared = (charset or "").lower().replace("_", "-")
    return declared.startswith(("utf-16", "utf-32")) or body[:2] in (b"\xff\xfe", b"\xfe\xff")


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


def is_index_page(url: str) -> bool:
    """True for a home, index, calendar or schedule page, whose dates are template or listing dates."""
    path = urlsplit(url).path
    last = path.rsplit("/", 1)[-1]
    return path in ("", "/") or path.endswith("/") or bool(INDEX_FILE.match(last)) or bool(INDEX_PATH.search(path))


def _iso(value) -> str | None:
    parsed = parse_date(str(value)) if value else None
    return parsed.date().isoformat() if parsed else None


def _walk_json_dates(node) -> list[str]:
    found = []
    if isinstance(node, dict):
        for key, value in node.items():
            if key == "datePublished" and isinstance(value, str):
                found.append(value)
            else:
                found.extend(_walk_json_dates(value))
    elif isinstance(node, list):
        for item in node:
            found.extend(_walk_json_dates(item))
    return found


def explicit_date(html: str, *, allow_time: bool) -> str | None:
    """A publication date the page states about itself: article:published_time, a datePublished meta or JSON-LD
    value, or (when allow_time) the first time element with a datetime attribute in the article or main part."""
    soup = BeautifulSoup(html, "lxml")
    for tag in soup.find_all("meta"):
        key = (tag.get("property") or tag.get("name") or tag.get("itemprop") or "").lower()
        if key in META_DATE_KEYS and _iso(tag.get("content")):
            return _iso(tag.get("content"))
    for script in soup.find_all("script", attrs={"type": "application/ld+json"}):
        try:
            values = _walk_json_dates(json.loads(script.string or ""))
        except (ValueError, RecursionError):   # a block nested too deep for the parser is not a date source
            continue
        for value in values:
            if _iso(value):
                return _iso(value)
    if allow_time:
        root = soup.find("article") or soup.find("main") or soup
        for tag in root.find_all("time", attrs={"datetime": True}):
            if _iso(tag.get("datetime")):
                return _iso(tag.get("datetime"))
    return None


def url_date(url: str) -> str | None:
    match = URL_DATE.search(urlsplit(url).path)
    return f"{match.group(1)}-{match.group(2)}-{match.group(3)}" if match else None


def publication_date(html: str, url: str, extracted: str | None, media: str) -> tuple[str | None, str]:
    """(date, date_source). Explicit page metadata wins. On an index or calendar page nothing else is trusted, since
    its date is a template or listing date. Elsewhere trafilatura's date counts as metadata, and a date in the URL path
    is the last resort, marked "url"."""
    index = is_index_page(url)
    if media in HTML_MEDIA:
        found = explicit_date(html, allow_time=not index)
        if found:
            return found, "meta"
    if index:
        return None, "none"
    if _iso(extracted):
        return _iso(extracted), "meta"
    found = url_date(url)
    return (found, "url") if found else (None, "none")


def extract(body: str, media: str) -> tuple[str, str, str | None, str]:
    """Return (text, title, date, method) from a decoded body."""
    if media == "application/json":
        try:
            return json.dumps(json.loads(body), indent=1, ensure_ascii=False), "", None, "json"
        except (ValueError, RecursionError):   # too deeply nested to pretty-print: keep the raw text
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


def fetch_page(url: str, *, max_bytes: int = 2_000_000, timeout: float = 15.0, use_cache: bool = True,
               cache_dir=None, client: httpx.Client | None = None, resolver: Resolver | None = None,
               robots: Robots | None = None, throttle: HostThrottle | None = None,
               total_seconds: float = TOTAL_SECONDS) -> Page:
    """Fetch and extract one page. Never raises; a refused or failed fetch returns Page(ok=False, note=...)."""
    try:
        return _fetch(url, max_bytes=max_bytes, timeout=timeout, use_cache=use_cache, cache_dir=cache_dir,
                      client=client, resolver=resolver, robots=robots, throttle=throttle,
                      total_seconds=total_seconds)
    except Exception as exc:   # last line of defence: the research tools must always get an answer
        return _failed(url, f"fetch failed: {type(exc).__name__}")


def _fetch(url, *, max_bytes, timeout, use_cache, cache_dir, client, resolver, robots, throttle,
           total_seconds) -> Page:
    if not isinstance(url, str) or not url.strip():
        return _failed(url or "", "the URL is empty")
    url = url.strip()
    try:
        check_syntax(url)   # no lookup here: each hop resolves and checks its host once, right before it connects
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
    deadline = time.monotonic() + total_seconds
    try:
        current = url
        for _ in range(MAX_REDIRECTS + 1):
            if time.monotonic() >= deadline:
                return _failed(url, f"no complete response within {total_seconds:g} seconds", final_url=current)
            try:
                check_syntax(current)
            except UnsafeURL as exc:
                return _failed(url, f"refused: {exc}", final_url=current)
            # robots.txt is per origin, so every hop is checked against the origin it is about to contact
            reason = _robots_reason(robots, current, deadline)
            if reason:
                return _failed(url, reason, final_url=current)
            throttle.wait(urlsplit(current).netloc.lower())
            try:
                response = open_pinned(http, current, resolver=resolver, headers=HEADERS,
                                       timeout=min(timeout, max(deadline - time.monotonic(), 0.001)),
                                       deadline=deadline)
            except UnsafeURL as exc:   # a redirect hop to a forbidden address is refused, not followed
                return _failed(url, f"refused: {exc}", final_url=current)
            except LookupTimeout:
                return _failed(url, "the DNS lookup did not finish before the deadline", final_url=current)
            try:
                if response.status_code in REDIRECT_CODES and response.headers.get("location"):
                    current = urljoin(current, response.headers["location"])
                    continue
                return _read_response(url, current, response, max_bytes=max_bytes, use_cache=use_cache, path=path,
                                      deadline=deadline, total_seconds=total_seconds)
            finally:
                response.close()
        return _failed(url, f"more than {MAX_REDIRECTS} redirects", final_url=current)
    finally:
        if own_client:
            http.close()


def _robots_reason(robots, url: str, deadline: float) -> str | None:
    """Why robots.txt refuses ``url``, or None. A caller's own robots object with only ``allowed(url)`` is honoured too."""
    if isinstance(robots, Robots):
        return robots.check(url, deadline=deadline)
    return None if robots.allowed(url) else "disallowed by robots.txt"


def _read_response(url: str, final_url: str, response: httpx.Response, *, max_bytes: int, use_cache: bool,
                   path, deadline: float | None = None, total_seconds: float = TOTAL_SECONDS) -> Page:
    status = response.status_code
    if status >= 300:   # a redirect with a location was followed by the caller; anything else is not content
        return _failed(url, f"HTTP {status}", final_url=final_url, status=status)
    media, charset = _media_and_charset(response.headers.get("content-type", ""))
    if media == "application/pdf":
        return _failed(url, "PDF files are not read; look for the HTML version or a text source",
                       final_url=final_url, status=status)
    if media and media not in ALLOWED_MEDIA:
        return _failed(url, f"content type {media} is not text; only HTML, plain text and JSON are read",
                       final_url=final_url, status=status)
    body, stopped = read_capped(response, max_bytes, deadline)
    notes = []
    if stopped == "encoding":
        return _failed(url, "the body uses a content encoding this reader does not decode, or is damaged; "
                            "only plain, gzip and deflate bodies are read", final_url=final_url, status=status)
    if stopped == "size":
        notes.append(f"response cut at {max_bytes} bytes")
    elif stopped == "time":
        notes.append(f"response stopped after {total_seconds:g} seconds")
    if body.startswith(b"%PDF-"):
        return _failed(url, "PDF files are not read; look for the HTML version or a text source",
                       final_url=final_url, status=status)
    if b"\x00" in body[:4096] and not _is_wide_text(body, charset):
        return _failed(url, "the body is binary, not text; only HTML, plain text and JSON are read",
                       final_url=final_url, status=status)
    decoded = _decode(body, charset)
    text, title, extracted, method = extract(decoded, media or "text/html")
    if method == "bs4":
        notes.append("main text taken by fallback extraction")
    text = text.strip()
    if not text:
        return _failed(url, "no readable text found", final_url=final_url, status=status)
    if len(text) > MAX_TEXT_CHARS:
        text = text[:MAX_TEXT_CHARS]
        notes.append(f"text clipped to {MAX_TEXT_CHARS} characters")
    date, date_source = publication_date(decoded, final_url, extracted, media or "text/html")
    page = Page(url=url, final_url=final_url, title=" ".join(title.split())[:300], date=date,
                date_source=date_source, text=text,
                sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(), fetched_at=_now(), status=status,
                note="; ".join(notes), ok=True)
    if use_cache:
        write_cache(path, asdict(page))
    return page
