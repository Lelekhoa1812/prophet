"""Tools that look at the person's data and test a forecast before it is trusted.

Motivation vs Logic
Motivation: before forecasting, the agent should know what the file is (layout, units, gaps, codes, formats), and after
forecasting it should know whether the model beat a simple rule on this data.
Logic: ``inspect_data`` runs prophet_lib.profile in the sandbox, so a corrupt or huge file cannot hurt the server, and
reports facts without repairing anything. ``backtest_series`` runs a rolling-origin backtest of TimesFM against simple
baselines on one series and reports the measured difference.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from prophet_agent.forecast.control import CONTROL_ARGS, CONTROL_HELP, build_control
from prophet_agent.forecast.rolling import rolling_backtest
from prophet_agent.runtime.runner import Tool
from prophet_agent.sandbox.runner import run_python
from prophet_agent.tools.series_io import SOURCE_HELP, VALUES_HELP, resolve_series, source_covariates

PROFILE_CODE = '''
import json
import prophet_lib as pl
out = pl.profile("in/" + {name!r}, sheet={sheet!r})
with open("out/profile.json", "w", encoding="utf-8") as f:
    json.dump(out, f)
'''
PROFILE_FILE = "out/profile.json"


def make_inspect(datasets: dict[str, Path]):
    def _run(args: dict, _key: str) -> dict[str, Any]:
        extra = sorted(set(args) - {"file", "sheet"})
        if extra:
            raise ValueError(f"unknown arguments {extra}; allowed: file, sheet")
        names = sorted(datasets)
        if not names:
            raise ValueError("no files were uploaded for this request")
        raw = args.get("file")
        if raw is not None and not isinstance(raw, str):
            # A list or dict is unhashable, so the lookup below raised a bare TypeError (round 6m).
            raise ValueError(f"file must be one of {names}, given as text")
        name = raw or (names[0] if len(names) == 1 else None)
        if name not in datasets:
            raise ValueError(f"file must be one of {names}")
        res = run_python(PROFILE_CODE.format(name=name, sheet=args.get("sheet", 0)), timeout_s=60, mem_mb=1536,
                         inputs={name: datasets[name]})
        if not res.ok:
            raise RuntimeError("the file could not be profiled: " + (res.stderr[-400:] or "timed out"))
        # The profile is read from its file, not from stdout: stdout is clipped to a head and a tail, and a clipped JSON
        # text failed with a parser error that did not name the cause (round 6l, item 8).
        path = Path(res.workdir) / PROFILE_FILE
        if not path.is_file():
            raise RuntimeError("the file was profiled but the profile was not written; check the file and try again")
        return {"file": name, "profile": json.loads(path.read_text(encoding="utf-8")), "run_id": res.run_id}
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
    out = {"file": result["file"], "rows": p["rows"], "read": {k: v for k, v in p["read"].items() if k != "file"},
           "columns": [_col_line(c) for c in p["columns"][:40]], "issues": p["issues"], "layout_hints": p["layout_hints"],
           "analysis_run_id": result["run_id"]}
    if len(p["columns"]) > 40:
        # Round 6m: the column cut was silent; a wide file looked as if it had only 40 columns.
        out["note"] = f"{len(p['columns'])} columns; the first 40 shown. The full profile is kept in the ledger"
    return out


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


MAX_HORIZON = 512           # the limit of timesfm_forecast and calibrate_intervals
MAX_BACKTEST_SERIES = 100   # as forecast_report's panel cap (round 5: electricity had 60 clients)


class _RecordingEngine:
    """Wraps the TimesFM engine and keeps the forecasts it returns, so the price comparison scores the same TimesFM points that
    the backtest scored. Every other attribute is passed through unchanged.

    Motivation vs Logic
    Motivation: the price comparison must put TimesFM on the same origins with the same points, and rolling_backtest does not
    return its per-origin forecasts.
    Logic: rolling_backtest calls forecast_batch once (all origins, default settings) and forecast_controlled once per origin
    when a control is set. Both results are kept as returned, so nothing is re-run and nothing is re-shaped.
    """

    def __init__(self, inner: Any):
        self._inner = inner
        self.batch: Any = None
        self.controlled: list[Any] = []

    def forecast_batch(self, histories: Any, horizon: int) -> Any:
        self.batch = self._inner.forecast_batch(histories, horizon)
        return self.batch

    def forecast_controlled(self, history: Any, horizon: int, control: Any) -> Any:
        run = self._inner.forecast_controlled(history, horizon, control)
        self.controlled.append(run)
        return run

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


def _rolling_with_price(series: Any, horizon: int, n_origins: int, season: int | None, control: Any, combine: bool,
                        price: bool | None) -> dict[str, Any]:
    """rolling_backtest, plus the price baselines on the same origins when the series is price-like (or price is forced).

    Motivation vs Logic
    Motivation: a price series that is compared only with "repeat the last value" can look like a model win on a rising stretch.
    Logic: detect_price_series decides (or the caller forces it). For a price, TimesFM's per-origin points are recorded from the
    backtest itself and scored beside the random walk, drift, EWMA and GARCH baselines on the same origins, and the result
    states which method was best and whether any beat the random walk. A non-price series gets the unchanged backtest.
    """
    from prophet_agent.forecast.price import detect_price_series, price_comparison

    x = np.asarray(series, dtype=float)
    if not detect_price_series(x, force=price)["is_price"]:
        return rolling_backtest(series, horizon, n_origins=n_origins, season=season, control=control,
                                combine_forecasts=combine)
    if not (np.isfinite(x).all() and (x > 0).all()):
        raise ValueError("price baselines need positive, finite values; pass price false to backtest this series as it is")
    from prophet_agent.forecast.timesfm import TimesFMEngine
    rec = _RecordingEngine(TimesFMEngine.get())
    out = rolling_backtest(series, horizon, n_origins=n_origins, season=season, control=control, combine_forecasts=combine,
                           engine=rec)
    k = int(out["origins"])
    timesfm = None
    if "timesfm_median" not in out["models"]:      # lumpy demand swaps TimesFM's point for expected volume: no price comparison
        if rec.controlled and len(rec.controlled) == k:
            timesfm = (np.stack([np.asarray(r.point) for r in rec.controlled]),
                       np.stack([np.asarray(r.q) for r in rec.controlled]))
        elif rec.batch is not None:
            timesfm = (np.asarray(rec.batch.point)[:k], np.asarray(rec.batch.q)[:k])
    out["price"] = price_comparison(x, horizon, k, season=int(out["season_used"]), timesfm=timesfm)
    return out


def _backtest(args: dict, _key: str) -> dict[str, Any]:
    extra = sorted(set(args) - {"values", "source", "horizon", "n_origins", "season", "combine", "price"} - CONTROL_ARGS)
    if extra:
        raise ValueError(f"unknown arguments {extra}")
    price = args.get("price")
    if price is not None and not isinstance(price, bool):
        raise ValueError("price must be true or false")
    horizon = args.get("horizon")
    if not isinstance(horizon, int) or isinstance(horizon, bool) or horizon < 1:
        raise ValueError("horizon must be a positive integer")
    if horizon > MAX_HORIZON:
        # The backtest forecasts this horizon at every window, so an unbounded one spends the model on steps timesfm_forecast
        # would refuse (round 6m: the same limit, 512, as the forecast tools).
        raise ValueError(f"horizon must be at most {MAX_HORIZON}, the limit of timesfm_forecast; a longer outlook is not backtested")
    named = resolve_series(args)
    if len(named) > MAX_BACKTEST_SERIES:
        raise ValueError(f"{len(named)} series in one call; at most {MAX_BACKTEST_SERIES}. Choose with source.name, or split the file")
    n_origins = args.get("n_origins", 12)
    if not isinstance(n_origins, int) or not 2 <= n_origins <= 30:
        raise ValueError("n_origins must be an integer from 2 to 30")
    combine = args.get("combine", False)
    if not isinstance(combine, bool):
        raise ValueError("combine must be true or false")
    season = args.get("season")
    if season is not None and (not isinstance(season, int) or not 1 <= season <= 1000):
        raise ValueError("season must be an integer from 1 to 1000 (1 means none)")
    if len(named) == 1:
        series = named[0][1]
        control = build_control(args, len(series), horizon, source_covariates(args))
        return _rolling_with_price(series, horizon, n_origins, season, control, combine, price)
    # Several named series (a panel saved with pl.save_panel): the same backtest on each, in one call, with a summary table.
    # Live, a 12-station mission spent its whole budget on 23 one-series calls and forecast only 8 stations.
    if any(k in args for k in CONTROL_ARGS if args.get(k) not in (None, [], False)):
        raise ValueError("control options apply to one series; backtest a panel with the defaults, then tune one series at a time")
    per: dict[str, Any] = {}
    for name, series in named:
        try:
            per[name] = _rolling_with_price(series, horizon, n_origins, season, None, combine, price)
        except ValueError as exc:
            per[name] = {"error": str(exc)[:200]}
    rows = {n: {"timesfm_mase": (r.get("models") or {}).get("timesfm", {}).get("mase"), "best_baseline": r.get("best_baseline"),
                "baseline_mase": (r.get("models") or {}).get(r.get("best_baseline") or "", {}).get("mase"),
                "verdict": r.get("verdict"), "coverage_80": r.get("timesfm_range_coverage_80"),
                **({"price_best_method": r["price"]["best_method"],
                    "price_beat_random_walk": r["price"]["beat_random_walk"]} if "price" in r else {})} if "error" not in r else r
            for n, r in per.items()}
    verdicts = [r.get("verdict") for r in rows.values() if r.get("verdict")]
    return {"series": per, "summary": rows, "horizon": horizon, "n_series": len(named),
            "verdict_counts": {v: verdicts.count(v) for v in sorted(set(verdicts))},
            "reading": "One backtest per series on the same settings. Choose the centre line per series from its own verdict."}


def observe_backtest(result: dict) -> dict:
    """A panel backtest shows its summary table to the planner, not every series' full result (the ledger keeps those)."""
    if "series" not in result:
        return result
    # Round 6m: the per-series results were dropped with no note; the summary table is what the planner sees.
    out = {k: v for k, v in result.items() if k != "series"}
    out["note"] = (f"the full result of each of the {len(result['series'])} series is kept in the ledger; this view shows the "
                   "summary table only")
    return out


def backtest_tool() -> Tool:
    return Tool("backtest_series", _backtest, idempotent=True, observe=observe_backtest)


BACKTEST_SPEC = {
    "name": "backtest_series",
    "title": "Accuracy check",
    "activity": "Checking how well it would have forecast the past",
    "description": "Rolling-origin backtest on one series, or on every series of a panel file (pl.save_panel, up to 100) in one call with a summary table: hides the last windows, forecasts them with TimesFM and with "
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
             "price": ("true or false to force or skip the price baselines (random walk, drift, EWMA and GARCH volatility, "
                       "scored on the same origins); default: detected from the values (positive, persistent, daily-like)"),
             **CONTROL_HELP},
}
