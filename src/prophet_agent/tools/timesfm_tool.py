"""TimesFM 3 as a runtime tool: one series in, quantiles and the recorded configuration out.

Motivation vs Logic
Motivation: a forecast is only useful if the reader can see which model, context and horizon produced it. The agent
must also never present a forecast as a commercial result: the weights are non-commercial.
Logic: the tool is pure (same input, same output), so it is marked idempotent and a crashed call can be re-run safely.
Every result carries the checkpoint, the context actually used, the horizon and the quantile levels. The licence
note lives in NOTICE.md and the README, not in results, so it never reaches the console or an answer. The model is loaded lazily on the first call, so importing the runtime never loads MLX.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from prophet_agent.forecast.timesfm import CHECKPOINT, TimesFMEngine
from prophet_agent.runtime.runner import Tool

QUANTILE_LEVELS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
LICENCE_NOTE = ("TimesFM 3.0 weights: TimesFM Non-Commercial License v1.0. Research use only; "
                "not for commercial decisions, client deliverables or paid services.")
MAX_HORIZON = 512
MAX_SERIES = 100_000


ALLOWED_ARGS = {"values", "horizon"}


def _run(args: dict, _key: str) -> dict[str, Any]:
    # The checkpoint is fixed to TimesFM 3. Unknown keys are refused, so a planner cannot pass a different model and
    # believe it was used.
    extra = sorted(set(args) - ALLOWED_ARGS)
    if extra:
        raise ValueError(f"unknown arguments {extra}; the model is fixed to TimesFM 3; allowed: values, horizon")
    values = args.get("values")
    horizon = args.get("horizon")
    if not isinstance(values, list) or not values or len(values) > MAX_SERIES:
        raise ValueError(f"values must be a non-empty list of at most {MAX_SERIES} numbers")
    if not isinstance(horizon, int) or not 1 <= horizon <= MAX_HORIZON:
        raise ValueError(f"horizon must be an integer from 1 to {MAX_HORIZON}")
    series = np.asarray(values, dtype=np.float64)
    engine = TimesFMEngine.get()
    fc = engine.forecast(series, horizon)
    ctx_used = min(len(series), engine.global_context)
    q = fc.q
    return {
        "point": [round(float(v), 4) for v in fc.point],
        "quantiles": {f"{lvl:.1f}": [round(float(v), 4) for v in q[:, i]] for i, lvl in enumerate(QUANTILE_LEVELS)},
        "config": {
            "checkpoint": engine.checkpoint or CHECKPOINT,
            "backend": "mlx",
            "global_context": engine.global_context,
            "context_used": ctx_used,
            "horizon": horizon,
            "quantile_levels": QUANTILE_LEVELS,
            "history_points": len(series),
        },
    }


def timesfm_tool() -> Tool:
    return Tool("timesfm_forecast", _run, idempotent=True)


TIMESFM_TOOL_SPEC = {
    "name": "timesfm_forecast",
    "title": "Forecast",
    "activity": "Forecasting the series",
    "description": "Forecast a univariate series with TimesFM 3. Returns point and quantile forecasts and the "
                   "exact configuration used. The model is fixed: pass "
                   "only the two arguments below.",
    "args": {"values": "list of numbers, oldest first", "horizon": "integer, 1 to 512"},
}
