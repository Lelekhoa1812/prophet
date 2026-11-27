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


def _series(provider: str, dataset: str, query: str, limit: int) -> list[dict]:
    url = f"{API}/series/{provider}/{dataset}?{urlencode({'q': query, 'limit': limit, 'observations': 0}, quote_via=quote)}"
    data = _get(url)
    found = []
    for doc in ((data.get("series") or {}).get("docs") or [])[:limit]:
        code = doc.get("series_code")
        if not code:
            continue
        found.append({"series": code, "name": str(doc.get("series_name", ""))[:200], "frequency": _frequency(doc),
                      "mirror_indexed_at": doc.get("indexed_at"),
                      "download": {"tool": "download_open_data",
                                   "args": {"source": "dbnomics",
                                            "params": {"provider": provider, "dataset": dataset, "series": code}}}})
    return found


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
