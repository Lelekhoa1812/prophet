"""Source credibility tiers and ordering of search hits: deduplicate, cap per domain, prefer primary sources and recent items.

Motivation vs Logic
Motivation: a forecast brief is only as good as its sources. A central bank, a statistics office or a major wire should
outweigh a content farm, and the same article should not fill five of the eight slots. The agent should see the most
trustworthy and most recent evidence first, with the reason visible.
Logic: a small, explicit domain table gives a tier from 1 (primary or official) to 4 (low quality), checked in that order:
primary suffixes and domains, then low-quality domains and URL patterns, then established outlets, then everything else.
Hits are deduplicated by normalised URL, capped at two per domain, and ordered by tier, then by date (newest first; undated
last), then by the search rank. The tier table is a judgement list for this project, not an authority ranking; it is
meant to be edited as the team learns which sources matter.
"""
from __future__ import annotations

import re
import time
from dataclasses import replace
from datetime import datetime, timezone
from urllib.parse import urlsplit

from prophet_agent.research.search import Hit, normalise_url

TIER1_DOMAINS = {
    "federalreserve.gov", "ecb.europa.eu", "bankofengland.co.uk", "boj.or.jp", "rba.gov.au", "bis.org",
    "imf.org", "worldbank.org", "oecd.org", "eia.gov", "fred.stlouisfed.org", "bls.gov", "bea.gov", "census.gov",
    "ons.gov.uk", "abs.gov.au", "opec.org", "iea.org", "sec.gov", "arxiv.org",
}
TIER1_SUFFIXES = (".gov", ".gov.au", ".gov.uk", ".europa.eu", ".edu")
TIER2_DOMAINS = {
    "reuters.com", "bloomberg.com", "ft.com", "wsj.com", "apnews.com", "bbc.co.uk", "economist.com", "cnbc.com",
    "nature.com", "science.org",
}
TIER4_DOMAINS = {"medium.com", "tumblr.com", "wordpress.com", "ehow.com", "answers.com", "hubpages.com",
                 "ezinearticles.com", "brainly.com"}
TIER4_HOST_MARKERS = ("blogspot.",)
TIER4_PATH = re.compile(r"[/-]best-|/top-?10\b|top-10|affiliate|[?&]aff=", re.I)
SCORE = {1: 100, 2: 80, 3: 55, 4: 20}
MAX_PER_DOMAIN = 2


def _host(url: str) -> str:
    host = (urlsplit(url).hostname or "").lower()
    return host[4:] if host.startswith("www.") else host


def _matches(host: str, domain: str) -> bool:
    return host == domain or host.endswith("." + domain)


def credibility(url: str) -> tuple[int, int]:
    """(tier, score) for a URL. Tier 1 is the most trusted, 4 the least; score is a 0 to 100 guide."""
    host = _host(url)
    parts = urlsplit(url)
    path_and_query = f"{parts.path}?{parts.query}" if parts.query else parts.path
    if any(_matches(host, d) for d in TIER1_DOMAINS) or host.endswith(TIER1_SUFFIXES):
        tier = 1
    elif any(_matches(host, d) for d in TIER4_DOMAINS) or any(m in host for m in TIER4_HOST_MARKERS) \
            or TIER4_PATH.search(path_and_query):
        tier = 4
    elif any(_matches(host, d) for d in TIER2_DOMAINS):
        tier = 2
    else:
        tier = 3
    return tier, SCORE[tier]


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


def rank_hits(hits: list[Hit], *, recency_days: float | None = None, now: float | None = None,
              max_per_domain: int = MAX_PER_DOMAIN) -> list[Hit]:
    """Deduplicate, cap per domain, drop hits older than ``recency_days`` (undated hits are kept), and order them.
    Each returned hit carries its tier and score."""
    now_ts = time.time() if now is None else now
    cutoff = now_ts - recency_days * 86400 if recency_days is not None else None
    scored = []
    for hit in hits:
        tier, score = credibility(hit.url)
        when = parse_date(hit.date)
        if cutoff is not None and when is not None and when.timestamp() < cutoff:
            continue
        ts = when.timestamp() if when else 0.0
        scored.append((tier, 0 if when else 1, -ts, hit.rank or 10_000, replace(hit, tier=tier, score=score)))
    scored.sort(key=lambda row: row[:4])
    seen: set[str] = set()
    per_domain: dict[str, int] = {}
    ranked: list[Hit] = []
    for *_, hit in scored:
        key = normalise_url(hit.url)
        if key in seen:
            continue
        host = _host(hit.url)
        if per_domain.get(host, 0) >= max_per_domain:
            continue
        seen.add(key)
        per_domain[host] = per_domain.get(host, 0) + 1
        ranked.append(replace(hit, url=key))
    return ranked
