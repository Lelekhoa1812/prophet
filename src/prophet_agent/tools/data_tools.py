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

from prophet_agent.forecast.control import CONTROL_ARGS, CONTROL_HELP, build_control
from prophet_agent.forecast.rolling import rolling_backtest
from prophet_agent.runtime.runner import Tool
from prophet_agent.sandbox.runner import run_python
from prophet_agent.tools.series_io import SOURCE_HELP, VALUES_HELP, resolve_series, source_covariates

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


def _col_line(c: dict) -> str:
    bits = [c["kind"]]
    for k in ("first", "last", "typical_step", "missing_periods", "min", "max", "mean", "zero_share", "missing", "unique"):
        if k in c and c[k] not in (None, 0, 0.0):
            v = c[k]
            bits.append(f"{k}={str(v)[:19] if k in ('first', 'last') else v}")
    for k in ("sentinel_candidate", "decimal_comma_values", "unparsed_values", "ambiguous_day_month", "looks_like_year"):
        if k in c:
            bits.append(f"{k}={c[k]}")
    return f"{c['name']}: " + ", ".join(bits) + f"; e.g. {c.get('sample', [])[:2]}"


def observe_inspect(result: dict) -> dict:
    p = result["profile"]
    return {"file": result["file"], "rows": p["rows"], "read": {k: v for k, v in p["read"].items() if k != "file"},
            "columns": [_col_line(c) for c in p["columns"][:40]], "issues": p["issues"], "layout_hints": p["layout_hints"],
            "analysis_run_id": result["run_id"]}


def inspect_data_tool(datasets: dict[str, Path] | None = None) -> Tool:
    return Tool("inspect_data", make_inspect(dict(datasets or {})), idempotent=True, observe=observe_inspect)


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
    extra = sorted(set(args) - {"values", "source", "horizon", "n_origins", "season", "combine"} - CONTROL_ARGS)
    if extra:
        raise ValueError(f"unknown arguments {extra}")
    horizon = args.get("horizon")
    if not isinstance(horizon, int) or isinstance(horizon, bool) or horizon < 1:
        raise ValueError("horizon must be a positive integer")
    named = resolve_series(args)
    if len(named) != 1:
        raise ValueError(f"backtest one series at a time; choose with source.name from {[n for n, _ in named][:20]}")
    n_origins = args.get("n_origins", 12)
    if not isinstance(n_origins, int) or not 2 <= n_origins <= 30:
        raise ValueError("n_origins must be an integer from 2 to 30")
    combine = args.get("combine", False)
    if not isinstance(combine, bool):
        raise ValueError("combine must be true or false")
    season = args.get("season")
    if season is not None and (not isinstance(season, int) or not 1 <= season <= 1000):
        raise ValueError("season must be an integer from 1 to 1000 (1 means none)")
    series = named[0][1]
    control = build_control(args, len(series), horizon, source_covariates(args))
    return rolling_backtest(series, horizon, n_origins=n_origins, season=season, control=control, combine_forecasts=combine)


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
                   "not beat the best simple rule, say so in the answer rather than presenting the forecast as reliable. Accepts the "
                   "same control arguments as timesfm_forecast (history window, ignored points, covariates, znorm, "
                   "symmetric averaging); then it also scores default TimesFM, so you can see whether the configuration "
                   "helped before using it. Future covariates must cover the history plus the horizon. The result gives "
                   "verdict (timesfm_better, baseline_better or no_clear_difference) from a paired 90 percent interval of "
                   "the MASE gain over the best simple rule, and wins per window. With combine true it also scores a blend "
                   "of TimesFM and the earlier-best rule, weighted on earlier windows only.",
    "args": {"values": VALUES_HELP, "source": SOURCE_HELP, "horizon": "integer periods ahead",
             "n_origins": "windows to test, 2 to 30 (default 12; fewer if the series is too short)",
             "season": "length of the repeating cycle in periods, 1 for none (default: chosen from the data by one shared rule)",
             "combine": "true to also score the TimesFM plus best-earlier-rule blend (default false)",
             **CONTROL_HELP},
}
