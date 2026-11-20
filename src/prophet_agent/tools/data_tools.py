"""Tools that look at the person's data and test a forecast before it is trusted.

Motivation vs Logic
Motivation: before forecasting, the agent should know what the file is (layout, units, gaps, codes, formats), and after
forecasting it should know whether the model beat a simple rule on this data.
Logic: ``inspect_data`` runs prophet_lib.profile in the sandbox, so a corrupt or huge file cannot hurt the server, and
reports facts without repairing anything. ``backtest_series`` runs a rolling-origin backtest of TimesFM against simple
baselines on one series and reports the measured difference.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from prophet_agent.forecast.rolling import rolling_backtest
from prophet_agent.runtime.runner import Tool
from prophet_agent.sandbox.runner import run_python
from prophet_agent.tools.series_io import SOURCE_HELP, VALUES_HELP, resolve_series

PROFILE_CODE = '''
import json, sys
import prophet_lib as pl
out = pl.profile("in/" + {name!r}, sheet={sheet!r})
print(json.dumps(out))
'''


def make_inspect(datasets: dict[str, Path]):
    def _run(args: dict, _key: str) -> dict[str, Any]:
        extra = sorted(set(args) - {"file", "sheet"})
        if extra:
            raise ValueError(f"unknown arguments {extra}; allowed: file, sheet")
        names = sorted(datasets)
        if not names:
            raise ValueError("no files were uploaded for this request")
        name = args.get("file") or (names[0] if len(names) == 1 else None)
        if name not in datasets:
            raise ValueError(f"file must be one of {names}")
        res = run_python(PROFILE_CODE.format(name=name, sheet=args.get("sheet", 0)), timeout_s=60, mem_mb=1536,
                         inputs={name: datasets[name]})
        if not res.ok:
            raise RuntimeError("the file could not be profiled: " + (res.stderr[-400:] or "timed out"))
        import json
        return {"file": name, "profile": json.loads(res.stdout), "run_id": res.run_id}
    return _run


def inspect_data_tool(datasets: dict[str, Path] | None = None) -> Tool:
    return Tool("inspect_data", make_inspect(dict(datasets or {})), idempotent=True)


INSPECT_DATA_SPEC = {
    "name": "inspect_data",
    "title": "Data check",
    "activity": "Reading and checking your data",
    "description": "Profile an uploaded file: layout (wide or long), columns and types, number and date formats, missing "
                   "values and codes that look like them, gaps and duplicate timestamps, and ambiguous day/month dates. "
                   "Reports facts; nothing is repaired. Use it first on any uploaded file.",
    "args": {"file": "name of an uploaded file (optional when there is only one)", "sheet": "spreadsheet sheet name or number (optional)"},
}


def _backtest(args: dict, _key: str) -> dict[str, Any]:
    extra = sorted(set(args) - {"values", "source", "horizon", "n_origins", "season"})
    if extra:
        raise ValueError(f"unknown arguments {extra}")
    horizon = args.get("horizon")
    if not isinstance(horizon, int) or isinstance(horizon, bool) or horizon < 1:
        raise ValueError("horizon must be a positive integer")
    named = resolve_series(args)
    if len(named) != 1:
        raise ValueError(f"backtest one series at a time; choose with source.name from {[n for n, _ in named][:20]}")
    n_origins = args.get("n_origins", 8)
    if not isinstance(n_origins, int) or not 2 <= n_origins <= 30:
        raise ValueError("n_origins must be an integer from 2 to 30")
    season = args.get("season")
    if season is not None and (not isinstance(season, int) or not 1 <= season <= 1000):
        raise ValueError("season must be an integer from 1 to 1000 (1 means none)")
    return rolling_backtest(named[0][1], horizon, n_origins=n_origins, season=season)


def backtest_tool() -> Tool:
    return Tool("backtest_series", _backtest, idempotent=True)


BACKTEST_SPEC = {
    "name": "backtest_series",
    "title": "Accuracy check",
    "activity": "Checking how well it would have forecast the past",
    "description": "Rolling-origin backtest on ONE series: hides the last windows, forecasts them with TimesFM and with "
                   "simple rules (repeat last value, same point last season, average slope, recent mean), and reports "
                   "MASE and WAPE per method plus how often the true values fell inside the 10 to 90 percent range. "
                   "Run it on the series you intend to forecast, with the horizon you intend to use. If TimesFM does "
                   "not beat the best simple rule, say so in the answer rather than presenting the forecast as reliable.",
    "args": {"values": VALUES_HELP, "source": SOURCE_HELP, "horizon": "integer periods ahead",
             "n_origins": "windows to test, 2 to 30 (default 8)",
             "season": "length of the repeating cycle in periods, 1 for none (default: detected from the data)"},
}
