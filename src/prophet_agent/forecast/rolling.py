"""Rolling-origin backtest for one series: does TimesFM beat simple rules on this data?

Motivation vs Logic
Motivation: a forecast is only worth trusting if it beats a rule anyone could apply. The agent needs that answer for the
series in front of it, not for a benchmark, and needs it before it spends effort on charts and write-ups.
Logic: cut the series at several past origins (non-overlapping windows of the forecast horizon, newest last), forecast
each window using only data before it, and score against what happened. Baselines: repeat the last value, repeat the
value one season back, extend the average slope, and the mean of the last season. Scores are MASE (error scaled by how
hard the series is to predict naively), WAPE, and for TimesFM the coverage of its 10 to 90 percent range. The result
reports the measured difference and its limits; it applies no pass or fail bar.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from prophet_agent.sandbox.prophet_lib import find_season

MIN_HISTORY = 8


def _snaive(x: np.ndarray, h: int, m: int) -> np.ndarray:
    if m < 2 or len(x) < m:
        return np.full(h, x[-1])
    return np.array([x[len(x) - m + (i % m)] for i in range(h)])


def _drift(x: np.ndarray, h: int) -> np.ndarray:
    if len(x) < 2:
        return np.full(h, x[-1])
    slope = (x[-1] - x[0]) / (len(x) - 1)
    return x[-1] + slope * np.arange(1, h + 1)


def _mean_last(x: np.ndarray, h: int, m: int) -> np.ndarray:
    return np.full(h, x[-max(m, 4):].mean())


def pick_season(x: np.ndarray, given: int | None) -> int:
    if given is not None:
        return max(1, int(given))
    res = find_season(x)
    lag, strength = res.get("strongest_lag"), res.get("strength")
    return int(lag) if lag and strength is not None and strength >= 0.3 else 1


def rolling_backtest(values: list[float] | np.ndarray, horizon: int, *, n_origins: int = 8, season: int | None = None,
                     engine: Any = None) -> dict:
    x = np.asarray(values, dtype=float)
    n = len(x)
    if not np.isfinite(x).all():
        raise ValueError("series contains NaN or infinite values; fill or drop them first")
    m = pick_season(x, season)
    min_hist = max(MIN_HISTORY, 2 * m if m > 1 else MIN_HISTORY)
    k = min(int(n_origins), (n - min_hist) // horizon)
    if k < 2:
        raise ValueError(f"series of {n} points is too short to backtest a horizon of {horizon} "
                         f"(needs at least {min_hist + 2 * horizon})")
    origins = [n - horizon * (k - j) for j in range(k)]           # oldest origin first, last window ends at n
    if engine is None:
        from prophet_agent.forecast.timesfm import TimesFMEngine
        engine = TimesFMEngine.get()
    fc = engine.forecast_batch([x[:o] for o in origins], horizon)

    names = ["timesfm", "naive", "seasonal_naive", "drift", "mean_last_season"]
    errs: dict[str, list[float]] = {nm: [] for nm in names}
    act_sum, abs_err = 0.0, {nm: 0.0 for nm in names}
    covered, pin = [], []
    qs = np.arange(1, 10) / 10
    for j, o in enumerate(origins):
        hist, actual = x[:o], x[o:o + horizon]
        preds = {"timesfm": fc.point[j], "naive": np.full(horizon, hist[-1]), "seasonal_naive": _snaive(hist, horizon, m),
                 "drift": _drift(hist, horizon), "mean_last_season": _mean_last(hist, horizon, m)}
        scale = np.mean(np.abs(hist[m:] - hist[:-m])) if len(hist) > m else np.mean(np.abs(np.diff(hist)))
        scale = scale if scale > 1e-12 else np.nan
        for nm, p in preds.items():
            e = np.abs(actual - p)
            errs[nm].append(float(e.mean() / scale) if scale == scale else float("nan"))
            abs_err[nm] += float(e.sum())
        act_sum += float(np.abs(actual).sum())
        q = fc.q[j]
        covered.append(float(((actual >= q[:, 0]) & (actual <= q[:, 8])).mean()))
        d = actual[:, None] - q
        pin.append(float(np.mean(np.maximum(qs * d, (qs - 1) * d)) / scale) if scale == scale else float("nan"))

    models = {nm: {"mase": round(float(np.nanmean(v)), 4) if len(v) and not np.all(np.isnan(v)) else None,
                   "wape": round(abs_err[nm] / act_sum, 4) if act_sum > 0 else None} for nm, v in errs.items()}
    base = {nm: v["mase"] for nm, v in models.items() if nm != "timesfm" and v["mase"] is not None}
    out: dict[str, Any] = {"points": n, "horizon": horizon, "origins": k, "season_used": m,
                           "models": models,
                           "timesfm_range_coverage_80": round(float(np.mean(covered)), 3),
                           "timesfm_pinball_scaled": round(float(np.nanmean(pin)), 4) if not np.all(np.isnan(pin)) else None}
    if base and models["timesfm"]["mase"] is not None:
        best = min(base, key=base.get)
        gain = (base[best] - models["timesfm"]["mase"]) / base[best] * 100 if base[best] else None
        out["best_baseline"] = best
        out["timesfm_vs_best_baseline_pct"] = round(gain, 1) if gain is not None else None
        out["reading"] = (
            "Lower MASE is better; 1.0 means as accurate as repeating the value one season (or step) back. "
            f"Range coverage should sit near 0.8. Measured over {k} windows of {horizon} periods; with this few windows a "
            "difference of a few percent is not distinguishable from noise.")
    return out
