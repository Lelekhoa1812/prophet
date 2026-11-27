"""Search the open-data catalogues for a series whose source and codes are not known.

Motivation vs Logic
Motivation: in the VN Unemployment notebook (2026-11-27) the planner guessed series codes it had never seen (two FRED ids, an
OECD dataflow for a non-member), was refused or answered 404 each time, and spent 28 steps before stopping. Codes are looked
up, not guessed: a principal analyst opens the catalogue first.
Logic: DBnomics indexes the catalogues of 80+ providers (IMF, ILO, OECD, Eurostat, World Bank, central banks, national
statistics offices) behind one keyless search. With a query alone, the datasets matching it are listed; with a dataset as
well, the series inside it matching the query are listed, each with the exact download_open_data call that fetches it. An
ILO dataset also names the ILOSTAT dataflow, since ILOSTAT itself is fresher than the mirror. Every request goes through
fetch.download (host and size rules), and nothing found here is a figure: it is where figures can be fetched from.
"""
from __future__ import annotations

import json
import re
from typing import Any
from urllib.parse import quote, urlencode

from prophet_agent.opendata import fetch

API = "https://api.db.nomics.world/v22"
DATASET_REF = re.compile(r"([A-Za-z0-9_-]{1,40})/([A-Za-z0-9_.:@+-]{1,160})")
MAX_QUERY = 200
MAX_LIMIT = 25
_CACHE: dict[str, dict] = {}


def _get(url: str) -> dict:
    if url in _CACHE:
        return _CACHE[url]
    fetched = fetch.download(url, max_bytes=4 * 1024 * 1024, timeout_s=30)
    data = json.loads(fetched["bytes"].decode("utf-8"))
    if not isinstance(data, dict):
        raise ValueError("the catalogue answered with something other than a JSON object")
    _CACHE[url] = data
    return data


def _datasets(query: str, limit: int) -> list[dict]:
    data = _get(f"{API}/search?{urlencode({'q': query, 'limit': limit}, quote_via=quote)}")
    found = []
    for doc in ((data.get("results") or {}).get("docs") or [])[:limit]:
        provider, code = doc.get("provider_code"), doc.get("code")
        if not provider or not code:
            continue
        item = {"provider": provider, "dataset": code, "name": str(doc.get("name", ""))[:160],
                "series_count": doc.get("nb_series"),
                "next": {"tool": "search_open_data", "args": {"query": query, "dataset": f"{provider}/{code}"}}}
        if provider == "ILO":
            item["direct"] = {"source": "ilostat_sdmx", "dataflow": f"DF_{code}",
                              "note": "ILOSTAT itself is fresher than the mirror; its key is country.frequency.measure.sex.age"}
        found.append(item)
    return found


def _frequency(doc: dict) -> str | None:
    freq = doc.get("@frequency")
    if freq:
        return str(freq)
    dims = doc.get("dimensions") or {}
    for name in ("FREQ", "freq", "frequency"):
        if dims.get(name):
            return str(dims[name])
    return None


AGE_TOTAL = "AGE_YTHADULT_YGE15"
FREQ_WORDS = {"monthly": "M", "month": "M", "quarterly": "Q", "quarter": "Q", "annual": "A", "yearly": "A", "year": "A"}


def ilostat_call(dataset: str, series: str) -> dict | None:
    """The ilostat_sdmx download for an ILO series in DBnomics' code order (ref_area.source.classif1.sex.frequency), or None.
    Round 20 (VNUNEMP, 2026-11-27): the mirror's copy stopped at 2023 while ILOSTAT had 2024; the planner took the mirror."""
    parts = series.split(".")
    pieces = dataset.split("_")
    if len(parts) != 5 or len(pieces) < 3 or parts[4] not in ("A", "Q", "M"):
        return None
    ref, _source, classif, sex, freq = parts
    if classif.startswith("AGE_") and classif.endswith("_TOTAL"):
        # ILOSTAT's SDMX service names the all-ages total only as 15+ (a band total answers 404); the mirror's band
        # totals equal it (Viet Nam 2023-Q1 to Q3: 1.545, 1.579, 1.674 in both, checked 2026-11-27).
        classif = AGE_TOTAL
    measure = "_".join(pieces[:2] + pieces[-1:])
    return {"tool": "download_open_data", "args": {"source": "ilostat_sdmx", "params": {
        "dataflow": f"DF_{dataset}", "key": f"{ref}.{freq}.{measure}.{sex}.{classif}"}}}


def _rank(doc: dict, query: str) -> tuple:
    """Totals before breakdowns, the frequency the query names first: the series an analyst wants is rarely the first the
    mirror lists (round 20: ten searches before the 15+ total turned up)."""
    code = str(doc.get("series_code", ""))
    want = {f for w, f in FREQ_WORDS.items() if re.search(rf"\b{w}\b", query.lower())}
    freq = code.rsplit(".", 1)[-1] if "." in code else ""
    parts = code.split(".")
    sex_total = int("SEX_T" in parts)
    age_total = int(any(p == AGE_TOTAL or (p.startswith("AGE_") and p.endswith("_TOTAL")) for p in parts))
    return (0 if not want or freq in want else 1, -(sex_total + age_total), -int(AGE_TOTAL in parts))


def _series(provider: str, dataset: str, query: str, limit: int) -> list[dict]:
    url = f"{API}/series/{provider}/{dataset}?{urlencode({'q': _place(query), 'limit': 100, 'observations': 0}, quote_via=quote)}"
    data = _get(url)
    found = []
    docs = [d for d in ((data.get("series") or {}).get("docs") or []) if isinstance(d, dict)]
    for doc in sorted(docs, key=lambda d: _rank(d, query))[:limit]:
        code = doc.get("series_code")
        if not code:
            continue
        found.append({"series": code, "name": str(doc.get("series_name", ""))[:200], "frequency": _frequency(doc),
                      "mirror_indexed_at": doc.get("indexed_at"),
                      "download": {"tool": "download_open_data",
                                   "args": {"source": "dbnomics",
                                            "params": {"provider": provider, "dataset": dataset, "series": code}}}})
        if provider == "ILO" and (direct := ilostat_call(dataset, code)):
            found[-1]["direct_download"] = direct
            found[-1]["note"] = "prefer direct_download: ILOSTAT itself is newer than the mirror"
    return found


MEASURE_WORDS = re.compile(r"\b(unemployment|employment|rate|monthly|quarterly|annual|yearly|seasonally|adjusted|total|sex|"
                           r"labour|labor|force|data|series|percent|age)\b", re.IGNORECASE)


def _place(query: str) -> str:
    """Inside one dataset the series are named by place and breakdown, so the measure's words only narrow the match away
    (round 20: 'Viet Nam monthly unemployment rate seasonally adjusted' matched nothing that 'Viet Nam' matched)."""
    rest = " ".join(MEASURE_WORDS.sub(" ", query).split())
    return rest or query


def search(query: Any, dataset: Any = None, limit: Any = 10) -> dict:
    """Datasets matching the query, or, with dataset as PROVIDER/DATASET, the series inside it that match the query."""
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be text naming the measure and the place, for example 'Viet Nam unemployment rate'")
    query = query.strip()[:MAX_QUERY]
    if isinstance(limit, bool) or not isinstance(limit, int):
        limit = 10
    limit = max(1, min(MAX_LIMIT, limit))
    if dataset is None:
        found = _datasets(query, limit)
        out: dict = {"query": query, "datasets": found}
        if not found:
            out["note"] = ("no dataset matched; use fewer words (the measure and the country's UN name, such as 'Viet Nam' "
                           "or 'Korea, Republic of'), or the measure alone")
        else:
            out["note"] = "pick a dataset and call search_open_data again with dataset set to its provider/dataset"
        return out
    match = DATASET_REF.fullmatch(dataset) if isinstance(dataset, str) else None
    if not match:
        raise ValueError("dataset must be PROVIDER/DATASET as listed by a search, for example ILO/UNE_DEAP_SEX_AGE_RT")
    provider, code = match.groups()
    found = _series(provider, code, query, limit)
    out = {"query": query, "dataset": f"{provider}/{code}", "series": found}
    out["note"] = ("each series carries the exact download call; the frequency is in the code or the name (A annual, Q "
                   "quarterly, M monthly)" if found else
                   "no series matched; use the country's name alone as the query, or check the dataset")
    return out
