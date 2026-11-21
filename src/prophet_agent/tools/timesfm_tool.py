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
from prophet_agent.forecast.tsb import expected_volume
from prophet_agent.runtime.runner import Tool
from prophet_agent.tools.series_io import SOURCE_HELP, VALUES_HELP, resolve_series, source_covariates

QUANTILE_LEVELS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
MAX_HORIZON = 512
ALLOWED_ARGS = {"values", "source", "horizon"} | CONTROL_ARGS
SHOWN = 30      # points of a forecast shown to the planner before it is thinned
FULL_SHOWN = 120   # a forecast up to this many periods is shown in full: thinning hid hours the planner needed (48 h live)
INTERMITTENT_ADI = 1.32   # Syntetos-Boylan cut-off: an average gap above this between periods with demand is intermittent


def is_intermittent(series: Any) -> bool:
    """True when demand is intermittent: the average gap between periods with demand is above INTERMITTENT_ADI."""
    s = np.nan_to_num(np.asarray(series, dtype=float))
    sold = int(np.count_nonzero(s > 0))
    return sold > 0 and len(s) / sold > INTERMITTENT_ADI


def _one(series: np.ndarray, point: np.ndarray, q: np.ndarray, horizon: int, engine: Any, control: Control | None = None) -> dict[str, Any]:
    cfg_extra = {"control": control.describe()} if control is not None and not control.is_default else {}
    median = [round(float(v), 4) for v in point]
    lumpy = is_intermittent(series)
    # For intermittent demand the median is zero or near it, so the point is the expected demand (mean under TSB).
    # The TimesFM median stays visible as median_point.
    shown = [round(float(v), 4) for v in expected_volume(series, horizon)] if lumpy else median
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
        },
    }


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
    if all(c.is_default for c in controls):
        fc = engine.forecast_batch([s for _, s in named], horizon)
        pairs = [(fc.point[i], fc.q[i]) for i in range(len(named))]
    else:
        pairs = []
        for (_, s), c in zip(named, controls):
            f = engine.forecast_controlled(s, horizon, c)
            pairs.append((f.point, f.q))
    results = {name: _one(s, pairs[i][0], pairs[i][1], horizon, engine, controls[i]) for i, (name, s) in enumerate(named)}
    if len(results) == 1 and ("values" in args or "series" in results):
        return next(iter(results.values()))        # one unnamed series keeps the flat shape
    return {"series": results}


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
