"""Web search through DuckDuckGo's ddgs library, with a backend cascade, health tracking, query variants, URL normalising
and an on-disk cache.

Motivation vs Logic
Motivation: on the machines this runs on, the default ddgs backends often answer "No results found" while Bing, Google
or Yahoo answer. A single backend choice therefore gives the agent silent empty results. A backend that keeps failing
should not be asked again on every call. A transient error (a timeout or a rate limit) is often gone a moment later.
Data pages are also often missed by one wording of a query, so several wordings are needed. The agent also needs the
same query not to cost a live call every time, and needs URLs in a stable form so that the same article is not counted
twice.
Logic: the backends are tried in order and the first non-empty answer wins. Exceptions and empty answers are recorded in
``diagnostics`` instead of raised, so the caller always gets a list (possibly empty) and can say why. In process, each
backend keeps a streak of bad answers (an exception or an empty answer); three in a row put it in a ten-minute cooldown,
during which it is skipped, and any good answer clears the streak. A transient exception is retried once after a
backoff. ``discover`` runs the query variants from ``queryplan`` and merges them with ``rank``; registry entries are
appended last and labelled, never mixed in as search hits. Results are cached as JSON under the Prophet data folder,
keyed by a hash of the query settings; news expires after six hours, text after a day. Tracking parameters (utm_*,
fbclid, gclid) and fragments are removed before a URL is stored or compared.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from prophet_agent.config import DATA
from prophet_agent.research.queryplan import MAX_VARIANTS, plan_query

DEFAULT_BACKENDS = ("auto", "bing", "google", "yahoo", "brave", "mojeek")
KINDS = ("text", "news")
TIMELIMITS = ("d", "w", "m", "y")
TTL_SECONDS = {"text": 24 * 3600, "news": 6 * 3600}
TRACKING_KEYS = {"fbclid", "gclid", "mc_cid", "mc_eid"}
MAX_QUERY_CHARS = 300
REGISTRY_SOURCE = "registry"
COOLDOWN_AFTER = 3          # consecutive bad answers before a backend cools down
COOLDOWN_S = 600.0          # seconds a cooling backend is skipped
DISCOVER_WORKERS = 3
VARIANT_FETCH = 10          # hits asked from each variant before ranking
TRANSIENT_NAMES = frozenset({"RatelimitException", "TimeoutException", "ConnectionException"})
TRANSIENT_TEXT = re.compile(r"\b429\b|rate.?limit|timed? ?out|temporar", re.I)


@dataclass
class Hit:
    title: str
    url: str
    snippet: str = ""
    date: str | None = None
    source: str = ""
    backend: str = ""
    rank: int = 0
    tier: int | None = None
    score: int | None = None


class SearchResult(list):
    """A list of Hit objects that also says which backend answered and what failed on the way.

    ``tried`` lists the backends asked (in order), ``answered`` those that gave a usable answer, empty or not. A
    ``discover`` result also carries ``variants`` (the wordings run) and ``answered_by`` (wording to backend)."""

    def __init__(self, hits=(), *, backend: str | None = None, diagnostics=None, cached: bool = False,
                 tried=None, answered=None, variants=None, answered_by=None):
        super().__init__(hits)
        self.backend = backend
        self.diagnostics: list[str] = list(diagnostics or [])
        self.cached = cached
        self.tried: list[str] = list(tried or [])
        self.answered: list[str] = list(answered or [])
        self.variants: list[str] = list(variants or [])
        self.answered_by: dict[str, str] = dict(answered_by or {})


@dataclass
class _BackendState:
    calls: int = 0
    ok: int = 0
    empty: int = 0
    failed: int = 0
    streak: int = 0
    cooling_until: float | None = None


_HEALTH: dict[str, _BackendState] = {}
_HEALTH_LOCK = threading.Lock()
_clock = time.monotonic      # tests replace this to move time


def normalise_url(url: str) -> str:
    """Lower-case scheme and host, drop the fragment, the tracking parameters and a trailing slash."""
    parts = urlsplit(url.strip())
    query = [(k, v) for k, v in parse_qsl(parts.query, keep_blank_values=True)
             if not k.lower().startswith("utm_") and k.lower() not in TRACKING_KEYS]
    path = parts.path.rstrip("/")
    return urlunsplit((parts.scheme.lower(), parts.netloc.lower(), path, urlencode(query), ""))


def configured_backends() -> tuple[str, ...]:
    raw = os.environ.get("PROPHET_SEARCH_BACKENDS", "")
    names = tuple(b.strip() for b in raw.split(",") if b.strip())
    return names or DEFAULT_BACKENDS


def cache_root() -> Path:
    env = os.environ.get("PROPHET_RESEARCH_CACHE")
    return Path(env).expanduser() if env else DATA / "research_cache"


def cache_key(*parts: Any) -> str:
    return hashlib.sha256(json.dumps(parts, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def read_cache(path: Path, ttl_s: float) -> dict | None:
    """The cached JSON document if it is younger than ``ttl_s``; None on any problem (a bad cache is a miss)."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if time.time() - float(data["saved_at"]) <= ttl_s:
            return data
    except (OSError, ValueError, KeyError, TypeError):
        return None
    return None


def write_cache(path: Path, data: dict) -> None:
    """Write atomically; a failed write only means the next call is live."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps({**data, "saved_at": time.time()}, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, path)
    except OSError:
        return


def _short(exc: BaseException) -> str:
    text = " ".join(str(exc).split())
    return text[:120]


def _is_transient(exc: BaseException) -> bool:
    """A timeout, connection problem or rate limit: worth one retry. Anything else is taken as a real refusal."""
    if isinstance(exc, (TimeoutError, ConnectionError, OSError)):
        return True
    return type(exc).__name__ in TRANSIENT_NAMES or bool(TRANSIENT_TEXT.search(str(exc)))


def _cooling_left(backend: str) -> float:
    """Seconds left in the backend's cooldown; 0 when it may be asked."""
    with _HEALTH_LOCK:
        state = _HEALTH.get(backend)
        if state is None or state.cooling_until is None:
            return 0.0
        return max(0.0, state.cooling_until - _clock())


def _record(backend: str, outcome: str) -> None:
    """Note one answer: ``ok``, ``empty`` or ``failed``. Three bad answers in a row start a cooldown."""
    with _HEALTH_LOCK:
        state = _HEALTH.setdefault(backend, _BackendState())
        state.calls += 1
        if outcome == "ok":
            state.ok += 1
            state.streak = 0
            state.cooling_until = None
            return
        state.empty += outcome == "empty"
        state.failed += outcome == "failed"
        state.streak += 1
        if state.streak >= COOLDOWN_AFTER:
            state.cooling_until = _clock() + COOLDOWN_S
            state.streak = 0


def backend_health() -> dict[str, dict]:
    """A snapshot of per-backend counts and seconds left in cooldown, for diagnostics."""
    with _HEALTH_LOCK:
        names = list(_HEALTH)
    return {name: {"calls": s.calls, "ok": s.ok, "empty": s.empty, "failed": s.failed,
                   "cooling_s": round(_cooling_left(name), 1)}
            for name, s in ((n, _HEALTH[n]) for n in names)}


def reset_backend_health() -> None:
    with _HEALTH_LOCK:
        _HEALTH.clear()


def _call_backend(kind: str, query: str, backend: str, *, timelimit: str | None, region: str, max_results: int):
    from ddgs import DDGS   # looked up at call time so tests can replace ddgs.DDGS

    client = DDGS()
    kwargs = {"region": region, "timelimit": timelimit, "max_results": max_results, "backend": backend}
    if kind == "news":
        return client.news(query, **kwargs)
    return client.text(query, **kwargs)


def _ask(kind: str, query: str, backend: str, *, timelimit: str | None, region: str, max_results: int,
         backoff_s: float):
    """(raw rows, None) or (None, exception). A transient exception gets one retry after a backoff."""
    for attempt in (0, 1):
        try:
            return _call_backend(kind, query, backend, timelimit=timelimit, region=region, max_results=max_results), None
        except Exception as exc:   # a failing backend must never stop the cascade
            if attempt == 0 and _is_transient(exc):
                if backoff_s > 0:
                    time.sleep(backoff_s * 2)
                continue
            return None, exc
    return None, None


def _to_hits(raw, kind: str, backend: str) -> list[Hit]:
    hits: list[Hit] = []
    seen: set[str] = set()
    for item in list(raw or []):
        if not isinstance(item, dict):
            continue
        url = str(item.get("href") or item.get("url") or "").strip()
        if not url.lower().startswith(("http://", "https://")):
            continue
        url = normalise_url(url)
        if url in seen:
            continue
        seen.add(url)
        hits.append(Hit(
            title=" ".join(str(item.get("title") or "").split()),
            url=url,
            snippet=" ".join(str(item.get("body") or item.get("snippet") or "").split()),
            date=(str(item["date"]) if kind == "news" and item.get("date") else None),
            source=str(item.get("source") or "") if kind == "news" else "",
            backend=backend,
            rank=len(hits) + 1,
        ))
    return hits


def _validate(query: Any, kind: str, max_results: Any, timelimit: str | None) -> str:
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty string")
    if len(query) > MAX_QUERY_CHARS:
        raise ValueError(f"query is longer than {MAX_QUERY_CHARS} characters")
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {list(KINDS)}")
    if isinstance(max_results, bool) or not isinstance(max_results, int) or not 1 <= max_results <= 25:
        raise ValueError("max_results must be a whole number from 1 to 25")
    if timelimit is not None and timelimit not in TIMELIMITS:
        raise ValueError(f"timelimit must be one of {list(TIMELIMITS)} or None")
    return " ".join(query.split())


def web_search(query: str, *, kind: str = "text", max_results: int = 8, timelimit: str | None = None,
               region: str = "wt-wt", backends: tuple[str, ...] | list[str] | None = None,
               backoff_s: float = 0.5, use_cache: bool = True) -> SearchResult:
    """Search the web with one wording. Never raises: failures are listed in ``result.diagnostics`` and the result is empty."""
    query = _validate(query, kind, max_results, timelimit)
    names = tuple(backends) if backends else configured_backends()

    key = cache_key(kind, query, timelimit, region)
    path = cache_root() / "search" / f"{key}.json"
    if use_cache:
        cached = read_cache(path, TTL_SECONDS[kind])
        if cached and cached.get("hits"):
            hits = [Hit(**h) for h in cached["hits"]][:max_results]
            return SearchResult(hits, backend=cached.get("backend"), cached=True,
                                tried=[], answered=[cached.get("backend")])

    diagnostics: list[str] = []
    live = [b for b in names if _cooling_left(b) <= 0]
    for backend in names:
        if backend not in live:
            diagnostics.append(f"{backend}: cooling down for {int(_cooling_left(backend))} s, skipped")
    if names and not live:
        soonest = min(names, key=_cooling_left)
        diagnostics.append(f"all backends cooling; probing {soonest}")
        live = [soonest]

    tried: list[str] = []
    answered: list[str] = []
    for index, backend in enumerate(live):
        if index and backoff_s > 0:
            time.sleep(backoff_s)
        tried.append(backend)
        raw, error = _ask(kind, query, backend, timelimit=timelimit, region=region, max_results=max_results,
                          backoff_s=backoff_s)
        if error is not None:
            diagnostics.append(f"{backend}: {type(error).__name__}: {_short(error)}")
            _record(backend, "failed")
            continue
        answered.append(backend)
        hits = _to_hits(raw, kind, backend)
        if hits:
            _record(backend, "ok")
            if use_cache:
                write_cache(path, {"query": query, "kind": kind, "backend": backend,
                                   "hits": [asdict(h) for h in hits]})
            return SearchResult(hits[:max_results], backend=backend, diagnostics=diagnostics, tried=tried,
                                answered=answered)
        _record(backend, "empty")
        diagnostics.append(f"{backend}: no results")
    diagnostics.append("every backend failed or returned nothing; try another wording, or check the network")
    return SearchResult([], backend=None, diagnostics=diagnostics, tried=tried, answered=answered)


def discover(query: str, *, kind: str = "text", max_results: int = 8, timelimit: str | None = None,
             region: str = "wt-wt", backends: tuple[str, ...] | list[str] | None = None,
             backoff_s: float = 0.5, use_cache: bool = True, max_variants: int = MAX_VARIANTS) -> SearchResult:
    """Search with several wordings of a data query and merge them.

    Motivation vs Logic
    Motivation: one wording of a data query often misses the data page, while a shorter, file-word or site-targeted
    wording finds it. The agent should not have to guess which wording works.
    Logic: ``plan_query`` gives the variants. Each runs through ``web_search`` (so backend health applies), is ranked
    on its own so that a strong variant cannot fill the list, and the variants are interleaved, de-duplicated by
    normalised URL and capped at ``max_results``. Registry entries for the query's series family follow the search
    hits, labelled with source "registry" and never counted as search results.
    """
    query = _validate(query, kind, max_results, timelimit)
    # Imported here: rank imports Hit and normalise_url from this module.
    from prophet_agent.research.rank import interleave, rank_report, registry_hits

    plan = plan_query(query)
    variants = list(plan.variants[:max_variants])
    with ThreadPoolExecutor(max_workers=max(1, min(DISCOVER_WORKERS, len(variants)))) as pool:
        results = list(pool.map(
            lambda v: web_search(v, kind=kind, max_results=max(max_results, VARIANT_FETCH), timelimit=timelimit,
                                 region=region, backends=backends, backoff_s=backoff_s, use_cache=use_cache),
            variants))

    diagnostics: list[str] = []
    answered_by: dict[str, str] = {}
    per_variant: list[list[Hit]] = []
    for variant, result in zip(variants, results):
        ranked, _repeats = rank_report(list(result), queries=[query], kind=kind, data_seeking=plan.data_seeking)
        per_variant.append(ranked)
        if result.backend:
            answered_by[variant] = result.backend
        if not result:
            diagnostics.append(f"{variant!r}: " + ("; ".join(result.diagnostics[-2:]) or "no results"))

    registry = registry_hits(query)
    registry_urls = {h.url for h in registry}
    search_hits = [h for h in interleave(per_variant) if h.url not in registry_urls][:max_results]
    if not search_hits:
        diagnostics.append("no search hits for any wording; registry entries, if any, are listed separately")
    winners = list(dict.fromkeys(answered_by.values()))
    return SearchResult(search_hits + registry, backend=winners[0] if winners else None,
                        diagnostics=diagnostics, tried=[], answered=winners, variants=variants,
                        answered_by=answered_by)
