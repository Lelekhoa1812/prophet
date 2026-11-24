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
from datetime import datetime, timezone
from typing import Any

from prophet_agent.research.events import dated_events, iso_day
from prophet_agent.research.fetch import PLAIN_MEDIA, Page, fetch_page, looks_like_data_table
from prophet_agent.research.passages import best_passages, select_passages, stated_date
from prophet_agent.research.primary import (REGIONAL_FAMILIES, detect_family, family_names, is_us_grid_query,
                                            is_us_region, registry_pages, resolve_family)
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
RESEARCH_ARGS = {"queries", "kind", "recency", "fetch_top", "passages_per_page", "exclude_urls", "family", "region",
                 "units", "data_end", "horizon_end"}
MAX_EXCLUDE = 200
MAX_REGISTRY = 6
MIN_USABLE_CHARS = 100   # a fetched page with less text than this is thin: it counts as failed and is backfilled (A3-07)
MAX_CHARS_LIMIT = 20000  # the web_fetch schema cap; a larger request is clamped and the clamp is reported (A3-07)


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
    """The series family given by the planner (aliases such as 'tariffs' are mapped), or detected from the query words
    when it is left out. An unknown family is refused and the valid names are listed (A3-06)."""
    value = args.get("family")
    if value is None:
        return detect_family(" ".join(queries))
    family = resolve_family(value) if isinstance(value, str) else None
    if family is None:
        raise ValueError(f"family must be one of {family_names()} or left out")
    return family


def _short_text(args: dict, name: str, limit: int) -> str:
    value = args.get(name)
    if value is None:
        return ""
    if not isinstance(value, str) or len(value) > limit:
        raise ValueError(f"{name} must be a string of up to {limit} characters or left out")
    return " ".join(value.split())


def _scope_queries(queries: list[str], region: str, units: str) -> tuple[list[str], list[dict]]:
    """The queries to send, scoped to the data's region and units (A4-08), and the queries dropped because they name
    US grid operators for a region that is not the United States (A3-06). Without a region nothing is added."""
    kept, dropped = [], []
    for query in queries:
        if region and not is_us_region(region) and is_us_grid_query(query):
            dropped.append({"query": query, "why": f"names US grid operators; the data's region is {region!r}"})
            continue
        extra = [part for part in (region, units) if part and part.lower() not in query.lower()]
        kept.append(" ".join([query, *extra]))
    return kept, dropped


def _date_of(page, hit: Hit | None) -> tuple[str | None, str]:
    """The page's own publication date when it has one; then a date the text states about itself ("released November
    10", source 'text', or 'text_year_inferred' when the year is not written); then the search engine's news date."""
    if page.date and page.date_source != "none":
        return page.date, page.date_source
    hit_year = parse_date(hit.date).year if hit is not None and hit.date and parse_date(hit.date) else None
    stated = stated_date(page.text, hit_year or datetime.now(timezone.utc).year)
    if stated:
        return stated[0], "text" if stated[1] else "text_year_inferred"
    if hit is not None and hit.date and parse_date(hit.date):
        return hit.date, "search"
    return None, "none"


def _dedupe_queries(queries: list[str]) -> tuple[list[str], int]:
    """Queries with case and spacing ignored, first occurrence kept; and how many were repeats (A3-07, round 3:
    the electricity researcher sent 96 query strings, 45 distinct)."""
    seen, kept = set(), []
    for query in queries:
        key = " ".join(query.lower().split())
        if key not in seen:
            seen.add(key)
            kept.append(query)
    return kept, len(queries) - len(kept)


def _fetch_with_backfill(ranked: list[Hit], fetch_top: int) -> tuple[list[tuple[Hit, Page]], list[dict]]:
    """Read ranked candidates until fetch_top are usable (ok and at least MIN_USABLE_CHARS of text), with at most
    2 x fetch_top attempts. Returns the usable (hit, page) pairs and the failed or thin attempts (A3-07: round 3 WTI
    run had 3 failed search pages, so the cap of 6 was spent and the OPEC+ page was never read)."""
    usable: list[tuple[Hit, Page]] = []
    failed: list[dict] = []
    index, cap = 0, 2 * fetch_top
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        while len(usable) < fetch_top and index < len(ranked) and index < cap:
            batch = ranked[index:index + min(fetch_top - len(usable), cap - index)]
            index += len(batch)
            for hit, page in zip(batch, pool.map(lambda h: fetch_page(h.url), batch)):
                if not page.ok:
                    failed.append({"url": hit.url, "why": page.note or "fetch failed", "origin": "search"})
                elif len(page.text) < MIN_USABLE_CHARS:
                    failed.append({"url": hit.url, "why": f"thin: {len(page.text)} characters of text",
                                   "origin": "search"})
                else:
                    usable.append((hit, page))
    return usable, failed


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
    requested = args.get("max_chars", 6000)
    clamp = None
    if isinstance(requested, int) and not isinstance(requested, bool) and requested > MAX_CHARS_LIMIT:
        clamp = f"max_chars {requested} clamped to {MAX_CHARS_LIMIT} (the schema limit)"   # A3-07 (e)
        requested = MAX_CHARS_LIMIT
    max_chars = _int({"max_chars": requested}, "max_chars", 6000, 500, MAX_CHARS_LIMIT)
    page = fetch_page(url.strip())
    out = _page_view(page, url.strip())
    if clamp:
        out["note"] = "; ".join(n for n in [out.get("note"), clamp] if n)
    if not page.ok:
        return out
    if (page.media or "") in (*PLAIN_MEDIA, "application/json") and looks_like_data_table(page.text):
        # The research cap keeps a passage, not the file. SciRes (2026-11-24) fetched the NOAA daily file and the sandbox
        # only saw the clipped text, then stalled on the same fetch. The full file is download_web_data's job.
        preview_lines = [line for line in page.text.splitlines() if line.strip() and not line.strip().startswith("#")]
        steer = ("this is a data table, not an article. preview is the first rows only. Call download_web_data with this "
                 "url so the full file is mounted for sandbox_python at in/<name>. Do not paste rows into sandbox code "
                 "and do not build a stand-in series.")
        out["data_file"] = True
        out["preview"] = "\n".join(preview_lines[:8])
        out["text_truncated"] = "clipped" in (page.note or "")
        out["note"] = "; ".join(part for part in (out.get("note"), steer) if part)
        return out
    if queries:
        passages = select_passages(page, queries, k=4)
        out["passages"] = passages
        return out
    out["text"] = page.text[:max_chars]
    out["text_truncated"] = len(page.text) > max_chars
    return out


def web_research_tool(args: Any, _key: str) -> dict:
    args = _allow(args, RESEARCH_ARGS)
    queries, repeated = _dedupe_queries(_queries(args.get("queries"), low=1, high=MAX_QUERIES))
    kind = _kind(args)
    timelimit, days = _recency(args)
    fetch_top = _int(args, "fetch_top", 4, 1, MAX_FETCH)
    per_page = _int(args, "passages_per_page", 3, 1, MAX_PASSAGES_PER_PAGE)
    exclude = _exclude(args.get("exclude_urls"))
    family = _family(args, queries)
    region = _short_text(args, "region", 80)
    units = _short_text(args, "units", 40)
    data_end = iso_day(args.get("data_end"), "data_end")
    horizon_end = iso_day(args.get("horizon_end"), "horizon_end")
    if family in REGIONAL_FAMILIES and not region:
        # A4-08 and A3-06: a regional family is never searched without the data's region (round 3 electricity run)
        return {"queries": queries, "family": family, "region_unknown": True, "sources": [], "failed": [],
                "events": [], "searched": 0, "new_sources": 0, "duplicate_sources": 0, "dropped_queries": [],
                "metrics": _metrics([]),
                "not_obtained": f"region unknown: no search was run for the {family} family. Read the region or "
                                "units from inspect_data or ask the user, then call web_research again with region."}
    scoped, dropped_queries = _scope_queries(queries, region, units)

    # round 0: the primary pages for the family, read directly without a search
    registry = registry_pages(family)[:MAX_REGISTRY]
    registry_keys = {normalise_url(p["url"]) for p in registry}
    registry_todo = [p for p in registry if normalise_url(p["url"]) not in exclude]
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        registry_read = list(pool.map(lambda p: fetch_page(p["url"]), registry_todo))

    # round 1: searches for the queries, then the top new pages
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        results = list(pool.map(lambda q: web_search(q, kind=kind, max_results=10, timelimit=timelimit), scoped))
    merged, diagnostics = [], []
    for query, result in zip(scoped, results):
        merged.extend(result)
        if not result:
            diagnostics.append(f"{query!r}: " + ("; ".join(result.diagnostics[-2:]) or "no results"))
    fresh = [h for h in merged if normalise_url(h.url) not in exclude and normalise_url(h.url) not in registry_keys]
    ranked, repeats = rank_report(fresh, recency_days=days, queries=scoped, kind=kind)
    duplicates = (len(merged) - len(fresh)) + repeats + (len(registry) - len(registry_todo))
    chosen, search_failed = _fetch_with_backfill(ranked, fetch_top)

    reads = ([("registry", p["url"], page, None, p["name"]) for p, page in zip(registry_todo, registry_read)]
             + [("search", h.url, page, h, None) for h, page in chosen])
    sources, failed = [], list(search_failed)
    for origin, url, page, hit, name in reads:
        if not page.ok:
            failed.append({"url": url, "why": page.note or "fetch failed", "origin": origin})
            continue
        passages = select_passages(page, scoped, k=per_page)
        date, date_source = _date_of(page, hit)
        source = {"origin": origin, "url": url, "final_url": page.final_url,
                  "title": page.title or (hit.title if hit else name), "date": date, "date_source": date_source,
                  "tier": credibility(page.final_url or url)[0], "sha256": page.sha256, "passages": passages}
        if name:
            source["name"] = name
        if all(p["reason"] == "head" for p in passages):
            source["note"] = "no passage matched the queries; the opening of the page is shown"
        sources.append(source)
    for number, source in enumerate(sources, 1):
        source["id"] = f"S{number}"
    out: dict[str, Any] = {"queries": queries, "repeated_queries": repeated, "searched_queries": scoped,
                           "dropped_queries": dropped_queries,
                           "family": family, "region": region or None, "units": units or None,
                           "sources": sources, "failed": failed, "searched": len(ranked),
                           "new_sources": len(sources), "duplicate_sources": duplicates,
                           "metrics": _metrics(sources)}
    out["events"] = dated_events(sources, data_end, horizon_end)
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
        elif data.get("events"):
            data["events"].pop()
            dropped["events"] = dropped.get("events", 0) + 1
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
    "description": "Read one web page (http or https only; no scripts or forms). Returns title, date, final_url, tier and "
                   "sha256 of the text. With queries, returns the best-matching quotable passages (each with start offset "
                   "and score); without them, the first max_chars of the text. When the page is a data table, data_file is "
                   "set and only a short preview is returned: call download_web_data with the same url for the full file, "
                   "and do not paste the preview into sandbox code.",
    "args": {"url": "the page address from a search hit",
             "queries": "optional list of 1 to 8 strings: what you want from the page",
             "max_chars": "whole number 500 to 20000 (default 6000; a larger value is clamped and reported), used when "
                          "no passage is asked for"},
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
                   "Pages that could not be read or are thin are replaced by the next ranked results and listed in "
                   "failed. Repeated queries are searched once (repeated_queries counts them). Dated passages are kept "
                   "even without query words. events lists each dated sentence (date, event, source_url, how_sure: high, medium or "
                   "low, a rule not a judgement) and its position against data_end and horizon_end.",
    "args": {"queries": "list of 1 to 8 search strings, each a different angle",
             "kind": "'text' (default) or 'news'",
             "recency": "optional 'day', 'week', 'month' or 'year'",
             "fetch_top": "how many top-ranked new pages to read, whole number 1 to 8 (default 4)",
             "passages_per_page": "passages per source, whole number 1 to 6 (default 3)",
             "exclude_urls": "optional list of up to 200 URLs already read in this mission",
             "family": f"optional, one of {', '.join(family_names())}; left out, detected from the queries",
             "region": "the place the data describes, e.g. 'Portugal' or 'Beijing'; queries are scoped to it. Regional "
                       "families (electricity, retail, weather, prices, pm25) are not searched without it",
             "units": "optional units of the data, e.g. 'kWh per day'; added to the queries",
             "data_end": "optional ISO date (YYYY-MM-DD) of the last observed point; each event is placed against it",
             "horizon_end": "optional ISO date (YYYY-MM-DD) of the forecast horizon end; events between the two are in the window"},
}


def tools() -> list[tuple[Tool, dict]]:
    """The three research tools with their specs, in the same shape as the other tool modules."""
    specs = [(WEB_SEARCH_SPEC, web_search_tool), (WEB_FETCH_SPEC, web_fetch_tool),
             (WEB_RESEARCH_SPEC, web_research_tool)]
    return [(Tool(spec["name"], fn, idempotent=True, doc=_spec_doc(spec), title=spec["title"],
                  activity=spec["activity"], observe=observe), spec) for spec, fn in specs]
