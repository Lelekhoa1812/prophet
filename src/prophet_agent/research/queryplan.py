"""Query variants for data discovery: a small, deterministic, rule-based set of wordings for one search.

Motivation vs Logic
Motivation: one wording rarely finds a data file. "NVDA historical stock price CSV download" returned only registry
pages in a live mission, while a shorter "NVDA stock price" or a site-targeted form may reach a price table. Search
engines also differ in how they read long queries and "site:" operators. The variants must stay few, predictable and
testable, because each one costs a live call per backend.
Logic: the query is split into an entity (leading words that name the subject), data-measure words, file words
(csv, json, download), history words (historical, daily) and stop words. Variants are built in priority order:
the query as given; a short form (the content words, at most five); the short form with a file word; the short form
with "historical data"; then site-targeted forms ("site:host short") for up to two hosts. The hosts come from the
opendata registry when a registry source shares a word with the query, and from a small domain table (finance, sport,
science, economics, energy) otherwise. The list is de-duplicated and capped at ``MAX_VARIANTS``. Queries that do not
ask for data get only the first two variants.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from urllib.parse import urlsplit

MAX_VARIANTS = 6
MAX_SHORT_WORDS = 5
MAX_SITE_HOSTS = 2

TOKEN = re.compile(r"[A-Za-z0-9][A-Za-z0-9&^./_-]*")
STOP = frozenset({
    "a", "an", "and", "the", "of", "for", "in", "on", "to", "from", "with", "by", "at", "or", "is", "are", "what",
    "latest", "current", "recent", "new", "free", "get", "show", "me", "please", "best", "site",
})
FILE_WORDS = frozenset({"csv", "json", "download", "xlsx", "excel", "file", "api", "zip", "txt"})
HISTORY_WORDS = frozenset({"historical", "historic", "history", "daily", "monthly", "weekly", "annual", "yearly",
                           "data", "dataset", "series", "timeseries", "time"})
MEASURE_WORDS = frozenset({
    "price", "prices", "close", "closes", "closing", "rate", "rates", "index", "table", "standings", "results",
    "fixtures", "demand", "storage", "spot", "cpi", "unemployment", "inflation", "anomaly", "temperature", "co2",
    "gdp", "output", "volume", "yield", "return", "returns", "level", "levels", "statistics", "stats",
})
DATA_WORDS = FILE_WORDS | HISTORY_WORDS | MEASURE_WORDS | frozenset({"stock", "quote", "record", "points", "scores"})

# Common public hosts for a domain, used when the query names the domain but no registry source matches.
DOMAIN_HOSTS = {
    "finance": (("price", "prices", "close", "closes", "closing", "stock", "share", "shares", "spot", "crude", "oil",
                 "wti", "brent", "nvda", "s&p", "equity", "ticker"), ("stooq.com", "nasdaq.com")),
    "sport": (("table", "standings", "results", "fixtures", "league", "season", "premier", "nba", "afl", "nrl"),
              ("wikipedia.org",)),
    "science": (("co2", "temperature", "anomaly", "climate", "emissions", "mauna", "sea", "ice"), ("noaa.gov",)),
    "economics": (("cpi", "inflation", "unemployment", "gdp", "rate", "rates", "wages", "consumer", "ecb"),
                  ("fred.stlouisfed.org", "abs.gov.au")),
    "energy": (("electricity", "demand", "gas", "storage", "power", "nem", "aemo", "wholesale", "eia"),
               ("eia.gov", "aemo.com.au")),
}


@dataclass(frozen=True)
class QueryPlan:
    query: str
    variants: tuple[str, ...]
    data_seeking: bool
    entity: str
    short: str
    site_hosts: tuple[str, ...] = field(default=())


def _tokens(text: str) -> list[str]:
    return TOKEN.findall(text or "")


def is_data_seeking(query: str) -> bool:
    """True when the query names a data word (a file, a history or a measure word such as price, table or CPI)."""
    words = {t.lower() for t in _tokens(query)}
    return bool(words & DATA_WORDS)


def _entity(tokens: list[str]) -> str:
    """Leading words before the first measure, file or history word: the subject of the query."""
    entity = []
    for tok in tokens:
        low = tok.lower()
        if low in MEASURE_WORDS or low in FILE_WORDS or low in HISTORY_WORDS or low in STOP:
            break
        entity.append(tok)
    return " ".join(entity)


def _short(tokens: list[str]) -> str:
    """The content words, dropping stop, file and history words, at most ``MAX_SHORT_WORDS`` words."""
    kept = [t for t in tokens if t.lower() not in STOP and t.lower() not in FILE_WORDS
            and t.lower() not in HISTORY_WORDS]
    return " ".join(kept[:MAX_SHORT_WORDS])


def _measure(tokens: list[str]) -> str:
    return " ".join(t for t in tokens if t.lower() in MEASURE_WORDS)


def _registry_hosts(terms: set[str]) -> list[str]:
    """Hosts of opendata registry sources whose title or key shares a word with the query (read only)."""
    from prophet_agent.opendata.sources import SOURCES

    hosts: list[str] = []
    for source in SOURCES.values():
        words = {w for w in re.split(r"[^a-z0-9]+", f"{source.key} {source.title}".lower()) if len(w) >= 3}
        if not words & terms:
            continue
        host = (urlsplit(source.url_template).hostname or "").lower()
        if host.startswith("www."):
            host = host[4:]
        if host and host not in hosts:
            hosts.append(host)
    return hosts


def _domain_hosts(low_words: set[str]) -> list[str]:
    hosts: list[str] = []
    for triggers, domain_hosts in DOMAIN_HOSTS.values():
        if low_words & set(triggers):
            hosts.extend(h for h in domain_hosts if h not in hosts)
    return hosts


def plan_query(query: str) -> QueryPlan:
    """The query variants for data discovery, in priority order, capped at ``MAX_VARIANTS``."""
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty string")
    text = " ".join(query.split())
    tokens = _tokens(text)
    low_words = {t.lower() for t in tokens}
    data = is_data_seeking(text)
    short = _short(tokens) or text
    entity = _entity(tokens) or short
    measure = _measure(tokens)

    candidates: list[str] = [text, short]
    site_hosts: list[str] = []
    if data:
        file_word = "json" if low_words & {"json", "api"} else "csv"
        candidates.append(f"{short} {file_word} download" if "download" in low_words else f"{short} {file_word}")
        candidates.append(f"{entity} {measure} historical data".replace("  ", " ").strip())
        # Domain hosts first: they name the subject. Registry hosts match only on subject words, not on generic data
        # words such as "data", "table" or "index", which would pull in unrelated agencies.
        hosts = _domain_hosts(low_words) + _registry_hosts(low_words - DATA_WORDS - STOP)
        site_hosts = list(dict.fromkeys(hosts))[:MAX_SITE_HOSTS]
        candidates.extend(f"site:{host} {short}" for host in site_hosts)

    variants: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = " ".join(candidate.lower().split())
        if key and key not in seen:
            seen.add(key)
            variants.append(" ".join(candidate.split()))
    return QueryPlan(query=text, variants=tuple(variants[:MAX_VARIANTS]), data_seeking=data, entity=entity,
                     short=short, site_hosts=tuple(site_hosts))
