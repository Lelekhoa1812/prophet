"""Source credibility tiers and ordering of search hits: deduplicate, cap per domain, weigh relevance, tier and recency.

Motivation vs Logic
Motivation: a forecast brief is only as good as its sources. A central bank, a statistics office or a major wire should
outweigh a content farm, and the same article should not fill five of the eight slots. An old official index page must
not beat this week's news, because a 20-day forecast starts from what is happening now.
Logic: a small, explicit table gives a tier from 1 (official or primary) to 4 (low quality), checked in that order:
official suffixes and domains, then low-quality domains and URL patterns, then academic and established outlets, then
everything else. Tier 1 is the only primary tier: universities and preprint servers are tier 2, because they are
evidence for a claim, not the place where a market figure is published. Hits are deduplicated by normalised URL and by
near-identical title (titles of twelve characters or more), capped at two per domain, and then ordered by a combined
score. The score is relevance (how many query terms the title and snippet carry), tier, and recency, with the weights
set by the kind of search: news leans on recency, text leans on tier and moderately on recency. Undated hits get no
recency credit, so they rank below dated hits of the same tier. The tier table is a judgement list for this project,
not an authority ranking; it is meant to be edited as the team learns which sources matter.
"""
from __future__ import annotations

import difflib
import re
import time
from dataclasses import replace
from datetime import datetime, timezone
from urllib.parse import urlsplit

from prophet_agent.research.passages import tokens
from prophet_agent.research.primary import detect_family, registry_pages
from prophet_agent.research.search import REGISTRY_SOURCE, Hit, normalise_url

# Official publishers: central banks, statistics offices, regulators and international agencies.
TIER1_DOMAINS = {
    "federalreserve.gov", "ecb.europa.eu", "bankofengland.co.uk", "boj.or.jp", "rba.gov.au", "bis.org",
    "imf.org", "worldbank.org", "oecd.org", "eia.gov", "fred.stlouisfed.org", "bls.gov", "bea.gov", "census.gov",
    "ons.gov.uk", "abs.gov.au", "opec.org", "iea.org", "sec.gov", "snb.ch", "bundesbank.de", "bok.or.kr",
    "bankofcanada.ca", "lbma.org.uk", "usgs.gov",
}
# Official suffixes: any .gov (and .gov.xx), .gouv.fr, Japan's go.jp, Canada's gc.ca, New Zealand's govt.nz, EU europa.eu.
TIER1_SUFFIX = re.compile(r"(^|\.)(gov|gov\.[a-z]{2}|gouv\.fr|go\.jp|gc\.ca|govt\.nz|europa\.eu)$")
# Academic and preprint hosts: evidence, but not a primary source for market figures.
TIER2_SUFFIX = re.compile(r"(^|\.)(edu|ac\.[a-z]{2}|edu\.[a-z]{2})$")
TIER2_DOMAINS = {
    "reuters.com", "bloomberg.com", "ft.com", "wsj.com", "apnews.com", "bbc.co.uk", "economist.com", "cnbc.com",
    "nikkei.com", "nature.com", "science.org", "arxiv.org",
}
TIER4_DOMAINS = {"medium.com", "tumblr.com", "wordpress.com", "ehow.com", "answers.com", "hubpages.com",
                 "ezinearticles.com", "brainly.com"}
TIER4_HOST_MARKERS = ("blogspot.",)
TIER4_PATH = re.compile(r"[/-]best-|/top-?10\b|top-10|affiliate|[?&]aff=", re.I)
SCORE = {1: 100, 2: 80, 3: 55, 4: 20}
MAX_PER_DOMAIN = 2
MIN_TITLE_CHARS = 12
TITLE_SIMILARITY = 0.9
# (relevance, tier, recency) weights and the recency half-life in days, by search kind.
WEIGHTS = {"news": (0.2, 0.3, 0.5), "text": (0.3, 0.5, 0.2)}
HALF_LIFE_DAYS = {"news": 7.0, "text": 120.0}
TIER_POINTS = {tier: score / 100 for tier, score in SCORE.items()}
# A data-seeking query prefers pages that carry the data itself: a file, an API or a download path, or a table-heavy host.
DATA_WEIGHT = 0.25
DATA_EXTENSIONS = (".csv", ".tsv", ".json", ".xlsx", ".xls", ".zip", ".txt", ".xml", ".parquet")
DATA_PATH = re.compile(r"/(data|api|download|downloads|dataset|datasets|history|historical|table|tables|csv|json)(/|$)")
TABLE_HOSTS = {"wikipedia.org", "stooq.com", "nasdaq.com", "stockanalysis.com", "investing.com", "macrotrends.net",
               "fred.stlouisfed.org", "ourworldindata.org", "statista.com"}


def host_of(url: str) -> str:
    host = (urlsplit(url).hostname or "").lower()
    return host[4:] if host.startswith("www.") else host


def _matches(host: str, domain: str) -> bool:
    return host == domain or host.endswith("." + domain)


def credibility(url: str) -> tuple[int, int]:
    """(tier, score) for a URL. Tier 1 is official and primary, 4 the least trusted; score is a 0 to 100 guide."""
    host = host_of(url)
    parts = urlsplit(url)
    path_and_query = f"{parts.path}?{parts.query}" if parts.query else parts.path
    if any(_matches(host, d) for d in TIER1_DOMAINS) or TIER1_SUFFIX.search(host):
        tier = 1
    elif any(_matches(host, d) for d in TIER4_DOMAINS) or any(m in host for m in TIER4_HOST_MARKERS) \
            or TIER4_PATH.search(path_and_query):
        tier = 4
    elif any(_matches(host, d) for d in TIER2_DOMAINS) or TIER2_SUFFIX.search(host):
        tier = 2
    else:
        tier = 3
    return tier, SCORE[tier]


def is_primary(url: str) -> bool:
    """True for official publishers (tier 1). Academic and preprint pages are evidence, never primary here."""
    return credibility(url)[0] == 1


def parse_date(value: str | None) -> datetime | None:
    """A timestamp from an ISO-like or common English date string; None when it cannot be read."""
    if not value:
        return None
    text = str(value).strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        parsed = None
        for fmt in ("%d %b %Y", "%b %d, %Y", "%d %B %Y", "%B %d, %Y", "%Y/%m/%d"):
            try:
                parsed = datetime.strptime(text, fmt)
                break
            except ValueError:
                continue
        if parsed is None:
            return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def relevance(hit: Hit, queries: list[str] | None) -> float:
    """Share of the best query's terms found in the hit's title and snippet, from 0 to 1."""
    if not queries:
        return 0.0
    words = set(tokens(f"{hit.title} {hit.snippet}"))
    best = 0.0
    for query in queries:
        terms = set(tokens(query))
        if terms:
            best = max(best, len(terms & words) / len(terms))
    return best


def recency_points(when: datetime | None, now_ts: float, kind: str) -> float:
    """1 for today, halving every half-life; 0 for an undated hit. Future dates count as today."""
    if when is None:
        return 0.0
    age_days = max((now_ts - when.timestamp()) / 86400, 0.0)
    return 0.5 ** (age_days / HALF_LIFE_DAYS.get(kind, HALF_LIFE_DAYS["text"]))


def combined_score(hit: Hit, when: datetime | None, now_ts: float, *, tier: int, queries: list[str] | None,
                   kind: str) -> float:
    rel_w, tier_w, rec_w = WEIGHTS.get(kind, WEIGHTS["text"])
    return (rel_w * relevance(hit, queries) + tier_w * TIER_POINTS[tier]
            + rec_w * recency_points(when, now_ts, kind))


def _title_key(title: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9]+", " ", (title or "").lower()).split())


def _near_same_title(key: str, kept: list[str]) -> bool:
    if len(key) < MIN_TITLE_CHARS:
        return False
    return any(difflib.SequenceMatcher(None, key, other).ratio() >= TITLE_SIMILARITY for other in kept)


def data_bearing(url: str) -> float:
    """How much the URL itself points at data: 1 for a data file, 0.6 for an API or download path, 0.5 for a
    table-heavy host, 0 otherwise."""
    parts = urlsplit(url)
    path = parts.path.lower()
    if path.endswith(DATA_EXTENSIONS):
        return 1.0
    if DATA_PATH.search(path):
        return 0.6
    if any(_matches(host_of(url), host) for host in TABLE_HOSTS):
        return 0.5
    return 0.0


def rank_report(hits: list[Hit], *, recency_days: float | None = None, now: float | None = None,
                max_per_domain: int = MAX_PER_DOMAIN, queries: list[str] | None = None,
                kind: str = "text", data_seeking: bool = False) -> tuple[list[Hit], int]:
    """Rank hits as ``rank_hits`` does, and also count the repeats dropped for a duplicate URL or title.

    Motivation vs Logic
    Motivation: for a data query the useful page is the one that carries the numbers, and a news article about the
    same subject should not outrank it.
    Logic: when ``data_seeking`` is set, each hit gets ``DATA_WEIGHT`` times ``data_bearing`` added to its score. The
    rest of the ranking is unchanged, so the flag only reorders hits that are otherwise close.
    """
    now_ts = time.time() if now is None else now
    cutoff = now_ts - recency_days * 86400 if recency_days is not None else None
    scored = []
    for hit in hits:
        tier, score = credibility(hit.url)
        when = parse_date(hit.date)
        if cutoff is not None and when is not None and when.timestamp() < cutoff:
            continue
        value = combined_score(hit, when, now_ts, tier=tier, queries=queries, kind=kind)
        if data_seeking:
            value += DATA_WEIGHT * data_bearing(hit.url)
        scored.append((-value, hit.rank or 10_000, replace(hit, tier=tier, score=score)))
    scored.sort(key=lambda row: row[:2])
    seen: set[str] = set()
    titles: list[str] = []
    per_domain: dict[str, int] = {}
    ranked: list[Hit] = []
    duplicates = 0
    for *_, hit in scored:
        key = normalise_url(hit.url)
        title = _title_key(hit.title)
        if key in seen or _near_same_title(title, titles):
            duplicates += 1
            continue
        host = host_of(hit.url)
        if per_domain.get(host, 0) >= max_per_domain:
            continue
        seen.add(key)
        if len(title) >= MIN_TITLE_CHARS:
            titles.append(title)
        per_domain[host] = per_domain.get(host, 0) + 1
        ranked.append(replace(hit, url=key))
    return ranked, duplicates


def rank_hits(hits: list[Hit], *, recency_days: float | None = None, now: float | None = None,
              max_per_domain: int = MAX_PER_DOMAIN, queries: list[str] | None = None,
              kind: str = "text", data_seeking: bool = False) -> list[Hit]:
    """Deduplicate, cap per domain, drop hits older than ``recency_days`` (undated hits are kept), and order them by
    the combined relevance, tier and recency score. Each returned hit carries its tier and credibility score."""
    return rank_report(hits, recency_days=recency_days, now=now, max_per_domain=max_per_domain, queries=queries,
                       kind=kind, data_seeking=data_seeking)[0]


def interleave(lists: list[list[Hit]]) -> list[Hit]:
    """Round-robin the ranked lists (first hit of each, then second of each, ...), dropping repeated URLs.

    Motivation vs Logic
    Motivation: when one wording returns ten strong hits, appending the variants one after another would hide every
    other wording's hits behind it.
    Logic: take position 0 from each list in turn, then position 1, and so on, so every wording contributes early. The
    first occurrence of a normalised URL wins.
    """
    out: list[Hit] = []
    seen: set[str] = set()
    for position in range(max((len(lst) for lst in lists), default=0)):
        for lst in lists:
            if position >= len(lst):
                continue
            hit = lst[position]
            key = normalise_url(hit.url)
            if key not in seen:
                seen.add(key)
                out.append(hit)
    return out


def registry_hits(query: str) -> list[Hit]:
    """The primary pages for the query's series family, as Hits labelled ``source="registry"``.

    Motivation vs Logic
    Motivation: when search finds nothing usable, the planner still needs the official pages to read. They are useful,
    but they are not search results, and the caller must be able to tell them apart.
    Logic: the family comes from the query. Each registry page becomes a Hit whose source and backend are "registry",
    so a caller that filters on ``source`` never presents one as a search hit. The list is empty when no family matches.
    """
    hits: list[Hit] = []
    for page in registry_pages(detect_family(query)):
        url = normalise_url(page["url"])
        tier, score = credibility(url)
        hits.append(Hit(title=page["name"], url=url, snippet=page.get("what", ""), date=None,
                        source=REGISTRY_SOURCE, backend=REGISTRY_SOURCE, rank=len(hits) + 1, tier=tier, score=score))
    return hits
