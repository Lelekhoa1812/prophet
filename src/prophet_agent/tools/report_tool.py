"""One tool that forecasts a series with TimesFM 3 and builds the chart, spreadsheet, document and PDF pack.

Motivation vs Logic
Motivation: the user should get the forecast as files they can open, not only as numbers in the chat.
Logic: the forecast comes from the same TimesFM call that the timesfm_forecast tool uses, so the files and the numbers
cannot drift apart. The pack is built inside the sandbox, and the result carries the manifest path so the files can be
verified later. The tool is idempotent: it writes only to a fresh run folder with no external effect.
"""
from __future__ import annotations

from typing import Any

from prophet_agent.artifacts import build_forecast_pack
from prophet_agent.forecast.control import CONTROL_ARGS, CONTROL_HELP
from prophet_agent.runtime.runner import Tool
from prophet_agent.tools.series_io import SOURCE_HELP, VALUES_HELP, resolve_series, run_dates, source_covariates
from prophet_agent.tools.timesfm_tool import _run as timesfm_run

MAX_TITLE = 120


def _future_dates(dates: list[str], n: int) -> list[str] | None:
    """The next ``n`` dates, continuing the history's own rhythm (daily, weekly, month-end ...). None when the dates cannot
    be read or the history has no regular step, so a chart never invents a calendar."""
    import pandas as pd
    try:
        idx = pd.DatetimeIndex(pd.to_datetime(dates))
    except (ValueError, TypeError):
        return None
    if len(idx) < 3 or not idx.is_monotonic_increasing:
        return None
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")            # pandas is changing the return type of infer_freq; the string is what is used here
        freq = pd.infer_freq(idx[-min(len(idx), 90):])
    if freq in ("B", "C") or (freq is None and _weekdays_only(idx)):
        # A trading-day series: skip the holidays of the exchange calendar its history follows, not just weekends.
        from prophet_agent.forecast.calendar import infer_calendar, next_trading_days
        cal = infer_calendar(t.date() for t in idx)
        nxt = pd.DatetimeIndex([pd.Timestamp(d) for d in next_trading_days(idx[-1].date(), n, cal["calendar"])])
    elif freq:
        nxt = pd.date_range(idx[-1], periods=n + 1, freq=freq)[1:]
    else:
        gaps = idx.to_series().diff().dropna()
        step = gaps.median()
        if step <= pd.Timedelta(0) or (gaps != step).mean() > 0.1:
            return None
        nxt = pd.DatetimeIndex([idx[-1] + step * (k + 1) for k in range(n)])
    return [t.strftime("%Y-%m-%d %H:%M") if (t.hour or t.minute) else t.strftime("%Y-%m-%d") for t in nxt]


def _weekdays_only(idx) -> bool:
    """True when a daily history has no weekend dates and its steps are one to four days (weekends and holidays)."""
    if len(idx) < 20 or (idx.hour != 0).any():
        return False
    gaps = idx.to_series().diff().dropna().dt.days
    return bool((idx.day_of_week < 5).all() and gaps.between(1, 4).all())


def _event_spans(args: dict, n: int, horizon: int) -> list[dict]:
    """Known-ahead drivers that are on/off flags, as runs of positions over history plus horizon, to shade on the chart."""
    pool = source_covariates(args)
    out = []
    for name in args.get("future_covariates") or []:
        v = pool.get(name)
        if not isinstance(v, list) or len(v) != n + horizon or not set(v) <= {0, 0.0, 1, 1.0}:
            continue
        spans, start = [], None
        for i, f in enumerate(v + [0]):
            if f and start is None:
                start = i
            elif not f and start is not None:
                spans.append([start, i])
                start = None
        if spans:
            out.append({"name": str(name), "spans": spans})
    return out
ALLOWED_ARGS = {"values", "source", "horizon", "title", "calibrate", "centre"} | CONTROL_ARGS
CENTRES = ("timesfm", "naive", "drift")


def _method_note(config: dict, centre: dict | None = None) -> str:
    """One line in the document that says what produced the numbers, so the file is honest on its own."""
    if centre:
        rule = {"naive": "the last value repeated", "drift": "the average historical change extended"}[centre["rule"]]
        return (f"Centre line: {rule}, because the backtest found it more accurate than TimesFM; the range is TimesFM 3's "
                "band, recentred on it.")
    c = config.get("control")
    if not c:
        return "TimesFM 3, default settings, no external drivers."
    parts = []
    if c.get("ignored_points"):
        parts.append(f"{c['ignored_points']} history points ignored")
    if c.get("future_covariates"):
        parts.append("known-ahead drivers: " + ", ".join(c["future_covariates"]))
    if c.get("past_covariates"):
        parts.append("past drivers: " + ", ".join(c["past_covariates"]))
    if c.get("context_points"):
        parts.append(f"last {c['context_points']} points only")
    if c.get("znorm"):
        parts.append("standardised")
    if c.get("symmetric_averaging"):
        parts.append("symmetric averaging")
    return "TimesFM 3: " + "; ".join(parts) + "."


def _health_note(health: dict | None) -> str:
    """A sentence for the document when the forecast is flat or the check found a cause, so the file explains itself."""
    if not health:
        return ""
    v = health.get("verdict")
    if v == "flat_expected":
        return (" The median is close to flat because the history shows no repeating pattern to carry forward; read the shaded "
                "interval, not the line, for how far values can move.")
    if v == "flat_unexpected":
        return " Warning: the history repeats on a cycle that this forecast does not reproduce: " + "; ".join(health.get("causes", [])[:2]) + "."
    if v == "check" and health.get("causes"):
        return " Check: " + "; ".join(health["causes"][:2]) + "."
    return ""


def _calibrated(fc: dict, values: list[float], args: dict) -> dict:
    """The forecast with its band replaced by the calibrated one when the held-out test says that is closer to 80 percent.
    The point forecast is never changed. When calibration cannot run (too short a series) the raw band stays and says so."""
    from prophet_agent.forecast.calibrate import calibrate
    from prophet_agent.forecast.control import build_control
    from prophet_agent.forecast.timesfm import TimesFMEngine
    from prophet_agent.tools.series_io import source_covariates
    horizon = len(fc["point"])
    try:
        control = build_control(args, len(values), horizon, source_covariates(args))
        if control.future or control.past:
            raise ValueError("calibration with covariates is not supported")
        res = calibrate(values, horizon, engine=TimesFMEngine.get(), control=control)
    except ValueError as exc:
        return {**fc, "interval_calibration": {"applied": False, "reason": str(exc)[:200]}}
    info = {"applied": res["recommend"] == "use_calibrated", "raw_heldout_coverage": res["heldout_coverage_raw"],
            "calibrated_heldout_coverage": res["heldout_coverage_calibrated"], "scales": res["scales_by_horizon_third"],
            "reason": res["reason"]}
    if not info["applied"]:
        return {**fc, "interval_calibration": info}
    return {**fc, "quantiles": res["calibrated"]["quantiles"], "interval_calibration": info}


def _centre(fc: dict, values: list[float], which: str) -> dict:
    """The forecast with its centre line replaced by a simple rule (the band stays TimesFM's, shifted to the new centre so the
    band keeps its width around it). Used when the backtest said the rule was more accurate. Records what was done."""
    import numpy as np
    h = len(fc["point"])
    last = float(values[-1])
    if which == "naive":
        new = np.full(h, last)
    else:   # drift: the average change over the history, extended
        slope = (float(values[-1]) - float(values[0])) / max(len(values) - 1, 1)
        new = last + slope * np.arange(1, h + 1)
    old = np.asarray(fc["point"], dtype=float)
    shift = new - old
    q = {k: [round(float(v) + float(d), 4) for v, d in zip(vals, shift)] for k, vals in fc["quantiles"].items()}
    return {**fc, "point": [round(float(v), 4) for v in new], "quantiles": q, "timesfm_point": fc["point"],
            "centre": {"rule": which, "reason": "the backtest found this simple rule more accurate than TimesFM's centre line; "
                       "the band is TimesFM's, recentred on the rule"}}


def _run(args: dict, key: str) -> dict[str, Any]:
    extra = sorted(set(args) - ALLOWED_ARGS)
    if extra:
        raise ValueError(f"unknown arguments {extra}; allowed: values or source, horizon, title, "
                         f"and the control options {sorted(CONTROL_ARGS)}")
    named = resolve_series(args)
    if len(named) != 1:
        raise ValueError(f"the pack is for one series; choose one with source.name from {[n for n, _ in named][:20]}")
    title = str(args.get("title", "Forecast"))[:MAX_TITLE] or "Forecast"
    centre = args.get("centre", "timesfm")
    if centre not in CENTRES:
        raise ValueError(f"centre must be one of {list(CENTRES)}")
    fc = timesfm_run({k: v for k, v in args.items() if k not in ("title", "calibrate", "centre")}, key)
    if "series" in fc:
        fc = next(iter(fc["series"].values()))
    values = [float(v) for v in named[0][1]]
    if args.get("calibrate") is True:
        fc = _calibrated(fc, values, args)
    if centre != "timesfm":
        fc = _centre(fc, values, centre)
    hist_dates = run_dates(args, named[0][0])
    if hist_dates is not None and len(hist_dates) != len(values):
        hist_dates = None
    ahead = _future_dates(hist_dates, len(fc["point"])) if hist_dates else None
    q = fc["quantiles"]
    pack = build_forecast_pack(history=values, point=fc["point"], q10=q["0.1"], q90=q["0.9"], title=title,
                               note=_method_note(fc["config"], fc.get("centre")) + _health_note(fc.get("health")), dates=hist_dates if ahead else None, future_dates=ahead,
                               events=_event_spans(args, len(values), len(fc["point"])),
                               ignored=args.get("ignore_ranges"), q30=q.get("0.3"), q70=q.get("0.7"))
    if not pack.ok:
        raise RuntimeError(f"artefact build failed: {pack.stderr[-500:]}")
    return {
        "forecast": fc,
        "run_id": pack.run_id,
        "manifest": f"{pack.workdir}/manifest.json",
        "files": [{"path": o.path, "sha256": o.sha256, "bytes": o.bytes} for o in pack.outputs],
    }


def observe(result: dict) -> dict:
    from prophet_agent.tools.timesfm_tool import _compact
    # The run id is what proves where the files are: without it the planner cannot cite or re-open the workbook and asks
    # where it went (seen live, as "the forecast workbook could not be found").
    return {"forecast": _compact(result["forecast"]), "run_id": result["run_id"],
            "files": [f["path"] for f in result["files"]], "manifest": result["manifest"],
            "range_vs_centre_pct": range_vs_centre(result["forecast"]),
            "deliverables": "built and listed in the manifest; no further build is needed"}


def range_vs_centre(fc: dict) -> dict | None:
    """How far the 10-90 percent band reaches below and above the centre line on the first and last step, in percent of the
    centre. Answers quote these ("day 30: +3.2% / -3.5%"); live, the grounding check flagged them because no tool had
    printed them."""
    try:
        point, lo, hi = fc["point"], fc["quantiles"]["0.1"], fc["quantiles"]["0.9"]
        out = {}
        for name, i in (("first_step", 0), ("last_step", len(point) - 1)):
            c = float(point[i])
            if c == 0:
                return None
            out[name] = {"below_pct": round((float(lo[i]) - c) / abs(c) * 100, 1),
                         "above_pct": round((float(hi[i]) - c) / abs(c) * 100, 1)}
        return out
    except (KeyError, IndexError, TypeError, ValueError):
        return None


def forecast_report_tool() -> Tool:
    return Tool("forecast_report", _run, idempotent=True, observe=observe)


FORECAST_REPORT_SPEC = {
    "name": "forecast_report",
    "title": "Forecast pack",
    "activity": "Building the chart, workbook and report",
    "description": "Forecast ONE series with TimesFM 3 and produce the deliverables a person opens: a chart (PNG, SVG), a "
                   "spreadsheet (XLSX), a report (DOCX) and a PDF, listed in a sha256 manifest. Use it for the final "
                   "forecast the person asked for, after any checking is done. It takes the same control arguments as "
                   "timesfm_forecast: pass exactly the configuration that won your backtest, or the pack will show the default forecast. "
                   "Backtest-only settings (season, n_origins) belong to backtest_series and are refused here.",
    "args": {"values": VALUES_HELP, "source": SOURCE_HELP, "horizon": "integer, 1 to 512", "title": "short title for the chart and report",
             "centre": "optional 'timesfm' (default), 'naive' or 'drift': when backtest_series says a simple rule was more accurate (verdict baseline_better), chart that rule as the centre line with TimesFM's band around it",
             "calibrate": "optional true: replace the 10-90 percent band by the calibrated one when a held-out test on past windows shows it covers closer to 80 percent (the point forecast never changes)",
             **CONTROL_HELP},
}
