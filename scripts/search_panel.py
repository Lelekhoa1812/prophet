"""Live data-discovery search panel: 20 fixed queries, recall of relevant non-registry hits.

Motivation vs Logic
Motivation: the owner decided there is no keyed search API, so the search cascade has to find data pages with ddgs alone.
A change to query planning or backend handling needs a measured before and after figure, not a feeling. The panel is
fixed so that runs are comparable.
Logic: each query is run once, live, in one of two modes. ``single`` is the original behaviour (the query as given, one
cascade). ``discover`` runs the query variants and merges them. A query counts as recalled when at least one non-registry
hit has a main entity alias and a data term in its URL or title. Registry entries never count. The check is a heuristic
string match, so the figure is reported as measured recall under that rule, not as a relevance score.
Output: JSON under data/search_panel/ (git-ignored). It only calls the search engines.

Usage: PYTHONPATH=src .venv/bin/python scripts/search_panel.py --mode single|discover [--label text]
"""
from __future__ import annotations

import argparse
import json
import sys
import time

from prophet_agent.config import DATA
from prophet_agent.research.search import web_search

REGISTRY_SOURCE = "registry"

# (id, query, main entity aliases, data terms). An alias or a term must appear in the hit URL or title (lower case).
PANEL = [
    ("nvda_csv", "NVDA historical stock price CSV download", ["nvda", "nvidia"],
     ["price", "history", "historical", "csv", "download", "quote", "data"]),
    ("nvda_daily", "NVDA daily stock price history", ["nvda", "nvidia"],
     ["price", "history", "historical", "daily", "data", "csv", "quote"]),
    ("spx_close", "S&P 500 daily closing prices historical data", ["s&p 500", "s&p500", "sp500", "^gspc", "gspc", "s-p-500"],
     ["price", "history", "historical", "close", "closing", "csv", "data"]),
    ("wti_spot", "WTI crude oil spot price history", ["wti", "crude oil", "crude-oil"],
     ["price", "history", "spot", "csv", "data", "daily"]),
    ("epl_table", "Premier League 2026-27 table", ["premier league", "premier-league", "premierleague"],
     ["table", "standings", "points", "league"]),
    ("epl_results", "Premier League 2026-27 results", ["premier league", "premier-league", "premierleague"],
     ["results", "fixtures", "scores", "match"]),
    ("nba_standings", "NBA 2026-27 standings", ["nba"], ["standings", "table", "record", "wins"]),
    ("nba_wins", "NBA standings by team wins and losses", ["nba"], ["standings", "wins", "losses", "table", "record"]),
    ("co2_daily", "Mauna Loa daily CO2 data", ["mauna loa", "mauna-loa", "maunaloa", "co2"],
     ["co2", "daily", "data", "csv", "record"]),
    ("co2_download", "Mauna Loa CO2 daily record download", ["mauna loa", "mauna-loa", "maunaloa", "co2"],
     ["co2", "daily", "record", "download", "csv", "data"]),
    ("temp_monthly", "global temperature anomaly monthly data", ["temperature"],
     ["anomaly", "monthly", "data", "csv", "global"]),
    ("temp_dataset", "global surface temperature anomaly dataset csv", ["temperature"],
     ["anomaly", "dataset", "csv", "data"]),
    ("cpi_monthly", "US CPI monthly index data", ["cpi", "consumer price", "consumer-price"],
     ["cpi", "index", "monthly", "data", "csv"]),
    ("cpi_csv", "US consumer price index CSV download", ["cpi", "consumer price", "consumer-price"],
     ["index", "csv", "download", "data"]),
    ("au_unemp", "Australian unemployment rate monthly data", ["unemployment"],
     ["unemployment", "rate", "monthly", "data", "csv", "labour", "labor"]),
    ("ecb_rates", "ECB deposit facility rate history", ["ecb", "deposit facility", "deposit-facility", "interest rate"],
     ["rate", "history", "data", "csv", "statistics"]),
    ("au_demand", "Australian electricity demand half-hourly data", ["electricity", "demand"],
     ["demand", "data", "csv", "half-hourly", "nem", "aemo"]),
    ("aemo_download", "AEMO NEM electricity demand data download", ["aemo", "nem", "electricity"],
     ["demand", "data", "download", "csv"]),
    ("eia_storage", "EIA natural gas storage weekly data", ["natural gas", "natural-gas", "natgas", "gas storage", "gas-storage"],
     ["storage", "weekly", "data", "csv", "gas"]),
    ("eia_storage_csv", "EIA weekly natural gas storage report csv", ["natural gas", "natural-gas", "natgas", "gas storage"],
     ["storage", "weekly", "report", "csv", "data"]),
]


def is_relevant(url: str, title: str, entities: list[str], terms: list[str]) -> bool:
    """True when the URL or title carries one entity alias and one data term."""
    text = f"{url} {title}".lower()
    return any(e in text for e in entities) and any(t in text for t in terms)


def run_query(mode: str, query: str, *, max_results: int = 10):
    if mode == "discover":
        from prophet_agent.research.search import discover   # added with the variant search
        return discover(query, max_results=max_results, use_cache=False, backoff_s=0.5)
    return web_search(query, max_results=max_results, use_cache=False, backoff_s=0.5)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--mode", choices=["single", "discover"], required=True)
    parser.add_argument("--label", default="")
    args = parser.parse_args(argv)

    rows = []
    started = time.time()
    for qid, query, entities, terms in PANEL:
        t0 = time.time()
        result = run_query(args.mode, query)
        search_hits = [h for h in result if h.source != REGISTRY_SOURCE]
        registry_hits = [h for h in result if h.source == REGISTRY_SOURCE]
        matched = [h.url for h in search_hits if is_relevant(h.url, h.title, entities, terms)]
        rows.append({
            "id": qid, "query": query, "recalled": bool(matched), "relevant_urls": matched[:3],
            "search_hits": len(search_hits), "registry_hits": len(registry_hits),
            "backend": result.backend, "diagnostics": result.diagnostics[-2:], "seconds": round(time.time() - t0, 1),
        })
        print(f"{qid:16s} recalled={bool(matched)!s:5s} hits={len(search_hits):2d} backend={result.backend}", flush=True)

    recalled = sum(1 for r in rows if r["recalled"])
    summary = {"mode": args.mode, "label": args.label, "queries": len(rows), "recalled": recalled,
               "started_at": started, "total_seconds": round(time.time() - started, 1), "rows": rows,
               "note": "measured: live ddgs run; relevance is a URL/title string match (entity alias and data term)"}
    out_dir = DATA / "search_panel"
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%dT%H%M%S", time.localtime(started))
    name = f"{args.mode}{'_' + args.label if args.label else ''}_{stamp}.json"
    path = out_dir / name
    path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"recalled {recalled} of {len(rows)} (mode={args.mode})")
    print(f"written: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
