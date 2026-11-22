"""TimesFM 3 as a runtime tool: one or more series in, quantiles and the recorded configuration out.

Motivation vs Logic
Motivation: a forecast is only useful if the reader can see which model, context and horizon produced it.
Logic: the tool is pure (same input, same output), so it is marked idempotent and a crashed call can be re-run safely.
Every result carries the checkpoint, the context actually used, the horizon and the quantile levels, so the planner,
the report and the written answer can see which model and settings produced the numbers. The series comes from ``values`` or by reference from an analysis run (see series_io). The model is loaded
lazily on the first call, so importing the runtime never loads MLX.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from prophet_agent.forecast.control import CONTROL_ARGS, CONTROL_HELP, Control, build_control
from prophet_agent.forecast.health import assess
from prophet_agent.forecast.timesfm import CHECKPOINT, TimesFMEngine
from prophet_agent.forecast.tsb import INTERMITTENT_ADI, expected_volume, is_intermittent, tsb_band  # noqa: F401
from prophet_agent.runtime.runner import Tool
from prophet_agent.tools.series_io import SOURCE_HELP, VALUES_HELP, resolve_series, run_dates, source_covariates

QUANTILE_LEVELS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
MAX_HORIZON = 512
# season and centre are accepted and ignored: the planner passes them on from backtest_series and forecast_report, as it
# does for calibrate_intervals (live WTI rep2: a refused "season" cost a call of a 20-call allowance).
ALLOWED_ARGS = {"values", "source", "horizon", "season", "centre"} | CONTROL_ARGS
SHOWN = 30      # points of a forecast shown to the planner before it is thinned
FULL_SHOWN = 120   # a forecast up to this many periods is shown in full: thinning hid hours the planner needed (48 h live)
# One definition of intermittent demand, shared with the backtest and calibration (forecast/tsb.py): the backtest must score
# the forecast this tool delivers (A6-01). That definition also refuses series that go below zero (a price, a balance).


IGNORED_REASON = {
    "season": "ignored by timesfm_forecast: TimesFM reads the cycle from the history and takes no season here; "
              "backtest_series and control_lab take one, and health.period gives the measured cycle",
    "centre": "ignored by timesfm_forecast: this point is TimesFM's median; forecast_report applies a rule centre",
}


def ignored_arguments(args: dict) -> dict[str, str]:
    """The arguments accepted for call compatibility but not applied, with the reason (A6-11). Empty when none was given.

    Motivation vs Logic
    Motivation: season and centre are accepted so a call is not refused (a refused season cost a live call), but nothing
    said they had no effect, so the planner could believe a season had been applied.
    Logic: name each one given, with its reason, in the recorded configuration and so in the observation.
    """
    return {k: v for k, v in IGNORED_REASON.items() if k in args}


def _one(series: np.ndarray, point: np.ndarray, q: np.ndarray, horizon: int, engine: Any, control: Control | None = None,
         crossings: int | None = None) -> dict[str, Any]:
    cfg_extra = {"control": control.describe()} if control is not None and not control.is_default else {}
    median = [round(float(v), 4) for v in point]
    lumpy = is_intermittent(series)
    # For intermittent demand the median is zero or near it, so the point is the expected demand (mean under TSB).
    # The TimesFM median stays visible as median_point.
    shown = [round(float(v), 4) for v in expected_volume(series, horizon)] if lumpy else median
    if lumpy:
        # The band must bracket the point shown. TimesFM's band sits on its median, near zero for lumpy demand, so the
        # point (the mean under TSB) fell outside it (audit item 14). The TSB band is the one the backtest scores, widened
        # so it always contains the expected volume (tsb_band, A6-02).
        q = tsb_band(np.asarray(series, dtype=float), horizon)
    ctx_used = min(len(series), (control.context_points if control is not None and control.context_points else engine.global_context))
    if lumpy:
        health = {"verdict": "flat_expected", "reason": "intermittent demand: the point is the expected demand, a level by design; "
                                                        "the information is in the interval and the probability of demand"}
    else:
        health = assess(series, point, q[:, 0], q[:, 8], context_used=ctx_used,
                        context_points=control.context_points if control is not None else None)
    return {
        "point": shown,
        "health": health,
        **({"median_point": median} if lumpy else {}),
        "quantiles": {f"{lvl:.1f}": [round(float(v), 4) for v in q[:, i]] for i, lvl in enumerate(QUANTILE_LEVELS)},
        "config": {
            "checkpoint": engine.checkpoint or CHECKPOINT,
            "backend": "mlx",
            "global_context": engine.global_context,
            "context_used": ctx_used,
            "horizon": horizon,
            "quantile_levels": QUANTILE_LEVELS,
            "history_points": len(series),
            "point_basis": "expected demand (mean under TSB)" if lumpy else "median",
            **cfg_extra,
            # A6-12: how many adjacent quantile pairs TimesFM returned out of order (sorted afterwards); absent if not reported
            **({"quantile_crossings": crossings} if crossings is not None else {}),
        },
    }


def _engine_crossings(engine: Any) -> list[int] | None:
    """The engine's crossing counts for its last call, one per series, or None when the engine does not report them."""
    reader = getattr(engine, "last_quantile_crossings", None)
    return reader() if callable(reader) else None


def _run(args: dict, _key: str) -> dict[str, Any]:
    # The checkpoint is fixed to TimesFM 3. Unknown keys are refused, so a planner cannot pass a different model and
    # believe it was used.
    extra = sorted(set(args) - ALLOWED_ARGS)
    if extra:
        raise ValueError(f"unknown arguments {extra}; the model is fixed to TimesFM 3; allowed: values or source, horizon, "
                         f"and the control options {sorted(CONTROL_ARGS)}")
    horizon = args.get("horizon")
    if not isinstance(horizon, int) or isinstance(horizon, bool) or not 1 <= horizon <= MAX_HORIZON:
        raise ValueError(f"horizon must be an integer from 1 to {MAX_HORIZON}")
    named = resolve_series(args)
    engine = TimesFMEngine.get()
    pool = source_covariates(args)
    controls = []
    for name, s in named:
        c = build_control(args, len(s), horizon, pool)
        if len(named) > 1 and (c.ignore or c.past or c.future):
            raise ValueError("ignore_ranges and covariates belong to one series; forecast one series (source.name) at a time")
        controls.append(c)
    crossings: list[int | None] = [None] * len(named)
    if all(c.is_default for c in controls):
        fc = engine.forecast_batch([s for _, s in named], horizon)
        pairs = [(fc.point[i], fc.q[i]) for i in range(len(named))]
        counts = _engine_crossings(engine)
        if counts is not None and len(counts) == len(named):
            crossings = list(counts)
    else:
        pairs = []
        for i, ((_, s), c) in enumerate(zip(named, controls)):
            f = engine.forecast_controlled(s, horizon, c)
            pairs.append((f.point, f.q))
            counts = _engine_crossings(engine)           # read straight after this call, before the next one
            if counts:
                crossings[i] = counts[0]
    results = {name: _one(s, pairs[i][0], pairs[i][1], horizon, engine, controls[i], crossings[i])
               for i, (name, s) in enumerate(named)}
    ignored = ignored_arguments(args)
    for name, r in results.items():
        if ignored:
            r["config"]["ignored_arguments"] = ignored
        step = _typical_step(run_dates(args, name))
        if step is not None:
            r["config"]["series_step"] = step      # the series' own step, so a request in weeks or months can be converted
    if len(results) == 1 and ("values" in args or "series" in results):
        return next(iter(results.values()))        # one unnamed series keeps the flat shape
    return {"series": results}


def _typical_step(dates: list[str] | None) -> float | None:
    """The median gap between consecutive dates, in days. None when there are fewer than three dates or they do not parse.

    Motivation vs Logic
    Motivation: "next 6 weeks" means 6 periods of a weekly series and 42 of a daily one. The forecast record has to say which
    series it forecast, or the horizon check cannot convert the request, and it can only warn.
    Logic: the median of the positive gaps, so one missing date or a duplicate does not move the step. Dates come from the
    run that holds the series; a values-only series has none and the record keeps no step.
    """
    if not dates or len(dates) < 3:
        return None
    try:
        t = np.array([np.datetime64(str(d).strip().replace(" ", "T")[:19]) for d in dates], dtype="datetime64[s]")
    except (ValueError, TypeError):
        return None
    gaps = np.diff(t).astype("timedelta64[s]").astype(float) / 86400.0
    gaps = gaps[gaps > 0]
    if gaps.size == 0:
        return None
    return round(float(np.median(gaps)), 4)


def _thin(v: list[float]) -> list[float]:
    if len(v) <= FULL_SHOWN:
        return [round(x, 3) for x in v]
    idx = np.linspace(0, len(v) - 1, SHOWN).round().astype(int)
    return [round(v[i], 3) for i in idx]


def _compact(r: dict) -> dict:
    q = r["quantiles"]
    out = {"config": {k: v for k, v in r["config"].items() if k != "quantile_levels"},
           "health": r.get("health"),
           "point": _thin(r["point"]), "q10": _thin(q["0.1"]), "q90": _thin(q["0.9"])}
    if len(r["point"]) > FULL_SHOWN:
        out["note"] = f"{len(r['point'])} periods; every ~{len(r['point']) // SHOWN}th shown"
    return out


def observe(result: dict) -> dict:
    if "series" in result:
        items = list(result["series"].items())
        out = {"series_count": len(items), "series": {k: _compact(v) for k, v in items[:5]}}
        if len(items) > 5:
            out["not_shown"] = len(items) - 5
        return out
    return _compact(result)


def timesfm_tool() -> Tool:
    return Tool("timesfm_forecast", _run, idempotent=True, observe=observe)


TIMESFM_TOOL_SPEC = {
    "name": "timesfm_forecast",
    "title": "Forecast",
    "activity": "Forecasting the series",
    "description": "Forecast one or more series with TimesFM 3 (a pretrained time-series model, used as it is). Returns "
                   "point and 10 to 90 percent quantile forecasts and the configuration used. Forecasts are floored at zero "
                   "only when the history never went below zero. For intermittent demand (sales in fewer than about three in four periods) the point is the expected demand (the mean under TSB) and the TimesFM median is returned as median_point. Pass the series as given; clean or reshape it in an "
                   "analysis step first when it needs that. Every result carries a health verdict: ok, check, flat_expected "
                   "(a flat median is the honest forecast; present the interval) or flat_unexpected (the history has a "
                   "cycle the forecast lost; read causes and try, fix the input, forecast again). The optional control arguments change how the model reads "
                   "the series (history window, points to ignore, known drivers such as promotions or holidays). They "
                   "are choices, not improvements: run backtest_series with the same arguments and keep one only if it "
                   "beats the default.",
    "args": {"values": VALUES_HELP, "source": SOURCE_HELP, "horizon": "integer, 1 to 512, in the series' own periods",
             **CONTROL_HELP},
}
