"""TimesFM 3 as a runtime tool: one or more series in, quantiles and the recorded configuration out.

Motivation vs Logic
Motivation: a forecast is only useful if the reader can see which model, context and horizon produced it.
Logic: the tool is pure (same input, same output), so it is marked idempotent and a crashed call can be re-run safely.
Every result carries the checkpoint, the context actually used, the horizon and the quantile levels. The series comes
from ``values`` or by reference from an analysis run (see series_io). The licence note lives in NOTICE.md and the
README, not in results, so it never reaches the console or an answer. The model is loaded lazily on the first call, so
importing the runtime never loads MLX.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from prophet_agent.forecast.timesfm import CHECKPOINT, TimesFMEngine
from prophet_agent.runtime.runner import Tool
from prophet_agent.tools.series_io import SOURCE_HELP, VALUES_HELP, resolve_series

QUANTILE_LEVELS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
LICENCE_NOTE = ("TimesFM 3.0 weights: TimesFM Non-Commercial License v1.0. Research use only; "
                "not for commercial decisions, client deliverables or paid services.")
MAX_HORIZON = 512
ALLOWED_ARGS = {"values", "source", "horizon"}
SHOWN = 30      # points of a forecast shown to the planner before it is thinned


def _one(series: np.ndarray, point: np.ndarray, q: np.ndarray, horizon: int, engine: Any) -> dict[str, Any]:
    return {
        "point": [round(float(v), 4) for v in point],
        "quantiles": {f"{lvl:.1f}": [round(float(v), 4) for v in q[:, i]] for i, lvl in enumerate(QUANTILE_LEVELS)},
        "config": {
            "checkpoint": engine.checkpoint or CHECKPOINT,
            "backend": "mlx",
            "global_context": engine.global_context,
            "context_used": min(len(series), engine.global_context),
            "horizon": horizon,
            "quantile_levels": QUANTILE_LEVELS,
            "history_points": len(series),
        },
    }


def _run(args: dict, _key: str) -> dict[str, Any]:
    # The checkpoint is fixed to TimesFM 3. Unknown keys are refused, so a planner cannot pass a different model and
    # believe it was used.
    extra = sorted(set(args) - ALLOWED_ARGS)
    if extra:
        raise ValueError(f"unknown arguments {extra}; the model is fixed to TimesFM 3; allowed: values or source, horizon")
    horizon = args.get("horizon")
    if not isinstance(horizon, int) or isinstance(horizon, bool) or not 1 <= horizon <= MAX_HORIZON:
        raise ValueError(f"horizon must be an integer from 1 to {MAX_HORIZON}")
    named = resolve_series(args)
    engine = TimesFMEngine.get()
    fc = engine.forecast_batch([s for _, s in named], horizon)
    results = {name: _one(s, fc.point[i], fc.q[i], horizon, engine) for i, (name, s) in enumerate(named)}
    if len(results) == 1 and ("values" in args or "series" in results):
        return next(iter(results.values()))        # one unnamed series keeps the flat shape
    return {"series": results}


def _thin(v: list[float]) -> list[float]:
    if len(v) <= SHOWN:
        return [round(x, 3) for x in v]
    idx = np.linspace(0, len(v) - 1, SHOWN).round().astype(int)
    return [round(v[i], 3) for i in idx]


def _compact(r: dict) -> dict:
    q = r["quantiles"]
    out = {"config": {k: v for k, v in r["config"].items() if k != "quantile_levels"},
           "point": _thin(r["point"]), "q10": _thin(q["0.1"]), "q90": _thin(q["0.9"])}
    if len(r["point"]) > SHOWN:
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
                   "only when the history never went below zero. Pass the series as given; clean or reshape it in an "
                   "analysis step first when it needs that.",
    "args": {"values": VALUES_HELP, "source": SOURCE_HELP, "horizon": "integer, 1 to 512, in the series' own periods"},
}
