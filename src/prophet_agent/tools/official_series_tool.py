"""official_series: series that official bodies publish only as release pages and workbooks, assembled and saved with dates.

Motivation vs Logic
Motivation (Mission 10, M10-4): VNUNEMP missions forecast from an ILOSTAT series that ends in 2024-Q4, because the statistics
office's quarterly rate is spread over release pages and workbooks that no generic download reads. The forecast then started
21 months before the run date.
Logic: each named series has a builder in opendata (vn_unemployment_quarterly: opendata.vn_labour). The values, each with the
address it came from, are saved as a dated series in a sandbox run (prophet_lib.save_series), so the forecast, backtest and
driver tools read it as source = {run_id, file, name}. Figures are as published; nothing is interpolated here.
"""
from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

from prophet_agent.opendata import market
from prophet_agent.opendata import vn_labour
from prophet_agent.runtime.runner import Tool
from prophet_agent.sandbox.runner import run_python

SERIES = {"vn_unemployment_quarterly": "Vietnam, unemployment rate of the working-age labour force, quarterly, per cent "
                                       "(statistics office releases)"}
OUT = "out/series_official.json"
SAVE_CODE = '''
import json
import pandas as pd
import prophet_lib as pl

doc = json.load(open("in/official.json", encoding="utf-8"))
s = pd.Series(doc["values"], index=pd.to_datetime(doc["dates"]))
pl.save_series({{doc["name"]: s}}, "{out}")
print(json.dumps({{"points": len(s), "from": str(s.index[0].date()), "to": str(s.index[-1].date())}}))
'''


def _fetch(url: str) -> bytes:
    return market._get(url, timeout_s=45.0, resolver=None)[0]


def _save(name: str, dates: list[str], values: list[float]) -> dict:
    doc = {"name": name, "dates": dates, "values": values}
    res = run_python(SAVE_CODE.format(out=OUT), timeout_s=60, mem_mb=1024,
                     files={"in/official.json": json.dumps(doc).encode("utf-8")})
    if not res.ok:
        raise ValueError("the series could not be saved: " + (res.stderr.strip()[-300:] or "timed out"))
    return {"run_id": res.run_id, "file": OUT, "name": name}


def _run(args: dict, _key: str, get: Callable[[str], bytes] | None = None,
         save: Callable[..., dict] | None = None) -> dict[str, Any]:
    extra = sorted(set(args) - {"series"})
    if extra:
        raise ValueError(f"unknown arguments {extra}; allowed: ['series']")
    name = args.get("series")
    if name not in SERIES:
        raise ValueError(f"series must be one of {sorted(SERIES)}")
    built = vn_labour.quarterly_unemployment(get or _fetch)
    rows = built["series"]
    source = (save or _save)(name, [r["date"] for r in rows], [r["value"] for r in rows])
    return {"series": name, "label": SERIES[name], "source": source, "grain": "quarterly", "first": built["first"],
            "last": built["last"], "points": built["points"], "gaps": built["gaps"], "values": rows,
            "releases": built["releases"], "failures": built["failures"], "measure": built["measure"],
            "provenance": built["source_label"], "figure_label": "real staging data, as published"}


def observe(result: dict) -> dict:
    vals = result.get("values", [])
    tail = [f"{r['period']}: {r['value']}" for r in vals[-8:]]
    return {"series": result.get("label"), "source": result.get("source"), "grain": result.get("grain"),
            "span": f"{result.get('first')} to {result.get('last')} ({result.get('points')} quarters)",
            "gaps": result.get("gaps"), "latest": tail, "measure": result.get("measure"),
            "provenance": result.get("provenance"), "figure_label": result.get("figure_label"),
            "release_addresses": sorted({r["source"] for r in vals[-4:]}), "failures": result.get("failures", [])[:3],
            "next": ("this is the published grain (quarterly) up to the latest release: backtest and forecast source = this "
                     "source in quarters; a monthly path, when asked for, is derived from the quarterly forecast and must be "
                     "labelled as derived, never as published")}


def official_series_tool(get: Callable[[str], bytes] | None = None, save: Callable[..., dict] | None = None) -> Tool:
    return Tool("official_series", lambda args, key: _run(args, key, get, save), idempotent=True,
                doc="official_series: an official series published only as release pages and workbooks, saved with dates",
                title="Official series", activity="Assembling an official series from its releases", observe=observe)


OFFICIAL_SERIES_SPEC = {
    "name": "official_series",
    "title": "Official series",
    "activity": "Assembling an official series from its releases",
    "description": "Assemble an official series that is published only as release pages and workbooks, with every value's "
                   "release address, and save it with its dates as a series file (source = {run_id, file, name}). "
                   "vn_unemployment_quarterly: Vietnam's working-age unemployment rate from the statistics office's quarterly "
                   "labour releases, from 2020-Q1 to the latest quarter.",
    "args": {"series": f"one of {sorted(SERIES)}"},
}
