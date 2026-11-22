"""The three web research tools the goal loop can call: search, fetch one page, and a combined research pass.

Motivation vs Logic
Motivation: the planner needs outside facts with sources it can cite. One search rarely gives enough; reading whole pages
floods the context. The planner should be able to ask for several angles at once and get back quotable passages, each
tied to a page fingerprint, plus a list of what could not be read.
Logic: each tool validates its arguments strictly (unknown names are refused and the allowed ones are named), then calls
the research modules. web_research works in rounds: round 0 reads the primary registry pages for the series family (no
search, the same fetch path as any page), round 1 searches the queries and reads the top new pages. Repeats are removed by
normalised URL and near-identical title, and URLs the planner already read (exclude_urls) are never fetched again. Each
call reports metrics (domains, tier-1 sources, dated sources, date range, new and duplicate counts) so that the planner
can stop when a round adds nothing. The three tools are idempotent: the ledger keeps the result of a finished call, so a
resumed mission never searches or fetches twice, and the duplicate check uses only the arguments, never earlier calls.
The observe function trims what the planner sees to about 9,000 characters and says how much was cut; the ledger keeps
all. No tool runs JavaScript, submits a form or opens a file:// URL.
"""
from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from prophet_agent.research.fetch import fetch_page
from prophet_agent.research.passages import best_passages
from prophet_agent.research.primary import PRIMARY_SOURCES, detect_family, registry_pages
from prophet_agent.research.rank import credibility, host_of, parse_date, rank_hits, rank_report
from prophet_agent.research.search import KINDS, Hit, normalise_url, web_search
from prophet_agent.runtime.runner import Tool

WORKERS = 4
MAX_QUERIES = 8
MAX_QUERY_CHARS = 300
MAX_FETCH = 8
MAX_PASSAGES_PER_PAGE = 6
PASSAGE_SEEN_CHARS = 500
OBSERVE_CHARS = 9000
RECENCY = {"day": ("d", 1), "week": ("w", 7), "month": ("m", 31), "year": ("y", 366)}
SEARCH_ARGS = {"query", "kind", "max_results", "recency"}
FETCH_ARGS = {"url", "queries", "max_chars"}
RESEARCH_ARGS = {"queries", "kind", "recency", "fetch_top", "passages_per_page", "exclude_urls", "family"}
MAX_EXCLUDE = 200
MAX_REGISTRY = 6


def _allow(args: Any, allowed: set[str]) -> dict:
    if not isinstance(args, dict):
        raise ValueError("arguments must be an object")
    extra = sorted(set(args) - allowed)
    if extra:
        raise ValueError(f"unknown arguments {extra}; allowed: {sorted(allowed)}")
    return args


def _int(args: dict, name: str, default: int, low: int, high: int) -> int:
    value = args.get(name, default)
    if value is None:
        value = default
    if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
        raise ValueError(f"{name} must be a whole number from {low} to {high}")
    return value


def _query(value: Any, name: str = "query") -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    if len(value) > MAX_QUERY_CHARS:
        raise ValueError(f"{name} is longer than {MAX_QUERY_CHARS} characters")
    return " ".join(value.split())


def _queries(value: Any, *, low: int, high: int) -> list[str]:
    if not isinstance(value, list) or not low <= len(value) <= high:
        raise ValueError(f"queries must be a list of {low} to {high} search strings, each a different angle")
    return [_query(q, "each query") for q in value]


def _kind(args: dict) -> str:
    kind = args.get("kind", "text")
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {list(KINDS)}")
    return kind


def _recency(args: dict) -> tuple[str | None, float | None]:
    value = args.get("recency")
    if value is None:
        return None, None
    if value not in RECENCY:
        raise ValueError(f"recency must be one of {list(RECENCY)} or left out")
    return RECENCY[value]


def _exclude(value: Any) -> set[str]:
    """Normalised URLs the planner has already read; they are counted as repeats and never fetched again."""
    if value is None:
        return set()
    if not isinstance(value, list) or len(value) > MAX_EXCLUDE:
        raise ValueError(f"exclude_urls must be a list of up to {MAX_EXCLUDE} http or https addresses")
    keys = set()
    for item in value:
        if not isinstance(item, str) or not item.strip().lower().startswith(("http://", "https://")):
            raise ValueError("each exclude_urls entry must be an http or https address")
        keys.add(normalise_url(item.strip()))
    return keys


def _family(args: dict, queries: list[str]) -> str | None:
    """The series family given by the planner, or detected from the query words when it is left out."""
    value = args.get("family")
    if value is None:
        return detect_family(" ".join(queries))
    if value not in PRIMARY_SOURCES:
        raise ValueError(f"family must be one of {list(PRIMARY_SOURCES)} or left out")
    return value


def _date_of(page, hit: Hit | None) -> tuple[str | None, str]:
    """The page's own publication date when it has one; otherwise the search engine's news date, marked as such."""
    if page.date and page.date_source != "none":
        return page.date, page.date_source
    if hit is not None and hit.date and parse_date(hit.date):
        return hit.date, "search"
    return None, "none"


def _metrics(sources: list[dict]) -> dict:
    """Coverage of one call: how many domains, how many official sources, how many dated, and the date range."""
    stamps = [(parse_date(s["date"]), s["date"]) for s in sources if s.get("date") and parse_date(s["date"])]
    newest = max(stamps)[0].date().isoformat() if stamps else None
    oldest = min(stamps)[0].date().isoformat() if stamps else None
    return {"unique_domains": len({host_of(s["final_url"] or s["url"]) for s in sources}),
            "tier1_sources": sum(1 for s in sources if s["tier"] == 1),
            "dated_sources": len(stamps), "newest_date": newest, "oldest_date": oldest,
            "new_sources": len(sources)}


def _hit_view(hit) -> dict:
    return {"rank": hit.rank, "title": hit.title, "url": hit.url, "snippet": hit.snippet, "date": hit.date,
            "source": hit.source, "tier": hit.tier, "backend": hit.backend}


def _page_view(page, url: str) -> dict:
    tier, _ = credibility(page.final_url or url)
    return {"url": url, "final_url": page.final_url, "title": page.title, "date": page.date,
            "date_source": page.date_source, "tier": tier,
            "ok": page.ok, "sha256": page.sha256, "chars": len(page.text), "note": page.note,
            "cached": page.cached}


def web_search_tool(args: Any, _key: str) -> dict:
    args = _allow(args, SEARCH_ARGS)
    query = _query(args.get("query"))
    kind = _kind(args)
    max_results = _int(args, "max_results", 8, 1, 10)
    timelimit, days = _recency(args)
    result = web_search(query, kind=kind, max_results=max_results, timelimit=timelimit)
    hits = rank_hits(list(result), recency_days=days, queries=[query], kind=kind)
    out: dict[str, Any] = {"query": query, "kind": kind, "backend": result.backend, "cached": result.cached,
                           "hits": [_hit_view(h) for h in hits]}
    if not hits:
        out["diagnostics"] = result.diagnostics or ["no results"]
    return out


def web_fetch_tool(args: Any, _key: str) -> dict:
    args = _allow(args, FETCH_ARGS)
    url = args.get("url")
    if not isinstance(url, str) or not url.strip():
        raise ValueError("url must be a non-empty http or https string")
    queries = args.get("queries")
    if queries is not None:
        queries = _queries(queries, low=1, high=MAX_QUERIES)
    max_chars = _int(args, "max_chars", 6000, 500, 20000)
    page = fetch_page(url.strip())
    out = _page_view(page, url.strip())
    if not page.ok:
        return out
    if queries:
        passages = best_passages(page, queries, k=4)
        if passages:
            out["passages"] = passages
            return out
        out["note"] = "; ".join(n for n in [page.note, "no passage matched the queries; first part of the page follows"]
                                if n)
    out["text"] = page.text[:max_chars]
    out["text_truncated"] = len(page.text) > max_chars
    return out


def web_research_tool(args: Any, _key: str) -> dict:
    args = _allow(args, RESEARCH_ARGS)
    queries = _queries(args.get("queries"), low=1, high=MAX_QUERIES)
    kind = _kind(args)
    timelimit, days = _recency(args)
    fetch_top = _int(args, "fetch_top", 4, 1, MAX_FETCH)
    per_page = _int(args, "passages_per_page", 3, 1, MAX_PASSAGES_PER_PAGE)
    exclude = _exclude(args.get("exclude_urls"))
    family = _family(args, queries)

    # round 0: the primary pages for the family, read directly without a search
    registry = registry_pages(family)[:MAX_REGISTRY]
    registry_keys = {normalise_url(p["url"]) for p in registry}
    registry_todo = [p for p in registry if normalise_url(p["url"]) not in exclude]
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        registry_read = list(pool.map(lambda p: fetch_page(p["url"]), registry_todo))

    # round 1: searches for the queries, then the top new pages
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        results = list(pool.map(lambda q: web_search(q, kind=kind, max_results=10, timelimit=timelimit), queries))
    merged, diagnostics = [], []
    for query, result in zip(queries, results):
        merged.extend(result)
        if not result:
            diagnostics.append(f"{query!r}: " + ("; ".join(result.diagnostics[-2:]) or "no results"))
    fresh = [h for h in merged if normalise_url(h.url) not in exclude and normalise_url(h.url) not in registry_keys]
    ranked, repeats = rank_report(fresh, recency_days=days, queries=queries, kind=kind)
    duplicates = (len(merged) - len(fresh)) + repeats + (len(registry) - len(registry_todo))
    chosen = ranked[:fetch_top]
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        pages = list(pool.map(lambda h: fetch_page(h.url), chosen))

    reads = ([("registry", p["url"], page, None, p["name"]) for p, page in zip(registry_todo, registry_read)]
             + [("search", h.url, page, h, None) for h, page in zip(chosen, pages)])
    sources, failed = [], []
    for origin, url, page, hit, name in reads:
        if not page.ok:
            failed.append({"url": url, "why": page.note or "fetch failed", "origin": origin})
            continue
        passages = best_passages(page, queries, k=per_page)
        date, date_source = _date_of(page, hit)
        source = {"origin": origin, "url": url, "final_url": page.final_url,
                  "title": page.title or (hit.title if hit else name), "date": date, "date_source": date_source,
                  "tier": credibility(page.final_url or url)[0], "sha256": page.sha256, "passages": passages}
        if name:
            source["name"] = name
        if not passages:
            source["note"] = "no passage matched the queries"
        sources.append(source)
    for number, source in enumerate(sources, 1):
        source["id"] = f"S{number}"
    out: dict[str, Any] = {"queries": queries, "family": family, "sources": sources, "failed": failed,
                           "searched": len(ranked), "new_sources": len(sources), "duplicate_sources": duplicates,
                           "metrics": _metrics(sources)}
    if diagnostics:
        out["search_diagnostics"] = diagnostics
    return out


def _clip_passages(data: dict) -> int:
    clipped = 0
    for source in data.get("sources", []):
        for passage in source.get("passages", []):
            if len(passage["text"]) > PASSAGE_SEEN_CHARS:
                passage["text"] = passage["text"][:PASSAGE_SEEN_CHARS] + "..."
                clipped += 1
    return clipped


def _drop_last_passage(data: dict) -> bool:
    for source in reversed(data.get("sources", [])):
        if source.get("passages"):
            source["passages"].pop()
            return True
    return False


def observe(result: dict) -> dict:
    """The planner's view: each passage at most 500 characters, the whole observation about 9,000 characters."""
    data = json.loads(json.dumps(result, default=str, ensure_ascii=False))
    size = lambda: len(json.dumps(data, ensure_ascii=False))  # noqa: E731
    original = size()
    clipped = _clip_passages(data)
    dropped: dict[str, int] = {}
    while size() > OBSERVE_CHARS:
        if _drop_last_passage(data):
            dropped["passages"] = dropped.get("passages", 0) + 1
        elif data.get("hits"):
            data["hits"].pop()
            dropped["hits"] = dropped.get("hits", 0) + 1
        elif data.get("sources"):
            data["sources"].pop()
            dropped["sources"] = dropped.get("sources", 0) + 1
        elif isinstance(data.get("text"), str):
            excess = size() - OBSERVE_CHARS
            data["text"] = data["text"][:max(len(data["text"]) - excess - 200, 500)]
            dropped["text_chars"] = dropped.get("text_chars", 0) + excess
            break
        else:
            break
    if clipped or dropped:
        bits = []
        if clipped:
            bits.append(f"{clipped} passages clipped to {PASSAGE_SEEN_CHARS} characters")
        for name, count in dropped.items():
            bits.append(f"{count} {name} left out")
        data["observation_note"] = f"observation cut from {original} to about {min(size(), OBSERVE_CHARS)} characters: " \
                                   + "; ".join(bits) + " (the ledger keeps the full result)"
    return data


def _spec_doc(spec: dict) -> str:
    args = "; ".join(f"{name}: {desc}" for name, desc in spec["args"].items())
    return f"{spec['description']} Arguments (use these names only): {args}."


WEB_SEARCH_SPEC = {
    "name": "web_search",
    "title": "Web search",
    "activity": "Searching the web",
    "description": "Search the public web for candidate sources. Returns ranked hits with a source tier (1 official or "
                   "primary, 2 established outlet, 3 other, 4 low quality), url, snippet and date. Search is one query per "
                   "call; use web_research to try several angles and read the pages.",
    "args": {"query": "the search, at most 300 characters",
             "kind": "'text' (default) for web pages or 'news' for recent news",
             "max_results": "whole number 1 to 10 (default 8)",
             "recency": "optional 'day', 'week', 'month' or 'year'; older dated hits are dropped"},
}

WEB_FETCH_SPEC = {
    "name": "web_fetch",
    "title": "Read a page",
    "activity": "Reading a web page",
    "description": "Read one web page (http or https only; robots.txt is honoured; no scripts or forms). Returns title, date, "
                   "final_url, tier and sha256 of the text. With queries, returns the best-matching quotable passages "
                   "(each with start offset and score); without them, the first max_chars of the text.",
    "args": {"url": "the page address from a search hit",
             "queries": "optional list of 1 to 8 strings: what you want from the page",
             "max_chars": "whole number 500 to 20000 (default 6000), used when no passage is asked for"},
}

WEB_RESEARCH_SPEC = {
    "name": "web_research",
    "title": "Research",
    "activity": "Researching the question from several angles",
    "description": "Research a question from 1 to 8 angles in one call. Round 0 reads the primary pages for the series "
                   "family (origin 'registry', detected from the queries unless family is given); round 1 runs the searches "
                   "in parallel and reads the top new pages (origin 'search'). Each source gives its best passages for all "
                   "the queries, its date and date_source ('meta' from the page, 'url' from the address, 'search' from the "
                   "news index, 'none'), and a sha256 of the page text. Quote only from these passages and cite the source "
                   "id (S1, S2 ...). Pass in exclude_urls the URLs you already read, so they are not fetched again; the "
                   "result counts them in duplicate_sources. metrics shows unique_domains, tier1_sources, dated_sources, "
                   "the date range and new_sources: stop when a round adds no new primary domain and no dated claim. "
                   "Pages that could not be read are listed in failed.",
    "args": {"queries": "list of 1 to 8 search strings, each a different angle",
             "kind": "'text' (default) or 'news'",
             "recency": "optional 'day', 'week', 'month' or 'year'",
             "fetch_top": "how many top-ranked new pages to read, whole number 1 to 8 (default 4)",
             "passages_per_page": "passages per source, whole number 1 to 6 (default 3)",
             "exclude_urls": "optional list of up to 200 URLs already read in this mission",
             "family": "optional 'oil', 'gas', 'gold', 'fx', 'rates', 'equities' or 'inflation'; left out, detected from the queries"},
}


def tools() -> list[tuple[Tool, dict]]:
    """The three research tools with their specs, in the same shape as the other tool modules."""
    specs = [(WEB_SEARCH_SPEC, web_search_tool), (WEB_FETCH_SPEC, web_fetch_tool),
             (WEB_RESEARCH_SPEC, web_research_tool)]
    return [(Tool(spec["name"], fn, idempotent=True, doc=_spec_doc(spec), title=spec["title"],
                  activity=spec["activity"], observe=observe), spec) for spec, fn in specs]
