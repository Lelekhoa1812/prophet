"""Honest baselines with empirical bands from their own past errors. Ported from src/lab/engine/baselines.py.

The `run` name dispatcher from the legacy file is not ported. Callers use the three functions directly.
"""
from __future__ import annotations

import numpy as np

Z = np.array([-1.2816, -0.8416, -0.5244, -0.2533, 0.0, 0.2533, 0.5244, 0.8416, 1.2816])


def _band(point: np.ndarray, resid_sd: float) -> np.ndarray:
    """point (h,) -> quantiles (h, 9), widening with sqrt of lead, floored at zero."""
    h = np.arange(1, len(point) + 1)[:, None]
    q = point[:, None] + Z[None, :] * resid_sd * np.sqrt(h)
    return np.clip(q, 0, None)


def moving_average(series: np.ndarray, h: int, window: int) -> tuple[np.ndarray, np.ndarray]:
    s = np.asarray(series, dtype=float)
    w = min(window, len(s))
    level = s[-w:].mean() if w else 0.0
    # residual sd from rolling one-step errors of the same rule
    if len(s) > w + 4:
        roll = np.convolve(s, np.ones(w) / w, mode="valid")[:-1]
        sd = float(np.std(s[w:] - roll))
    else:
        sd = float(np.std(s)) if len(s) > 1 else 0.0
    p = np.full(h, level)
    return p, _band(p, sd)


def seasonal_naive(series: np.ndarray, h: int, season: int, k: int = 4) -> tuple[np.ndarray, np.ndarray]:
    """Mean of the last k same-phase values, e.g. the last 4 same weekdays for daily data with season=7."""
    s = np.asarray(series, dtype=float)
    if len(s) < season:
        return moving_average(s, h, season)
    p = np.zeros(h)
    for i in range(h):
        idx = len(s) + i - season * np.arange(1, k + 1)
        idx = idx[idx >= 0]
        idx = idx[idx < len(s)]
        p[i] = s[idx].mean() if len(idx) else s[-season:].mean()
    resid = s[season:] - s[:-season]
    sd = float(np.std(resid) / np.sqrt(2)) if len(resid) > 2 else float(np.std(s))
    return p, _band(p, sd)


def seasonal_year(series: np.ndarray, h: int, season: int = 52) -> tuple[np.ndarray, np.ndarray]:
    """Same period last season, scaled by the recent level ratio. Needs more than one season of history."""
    s = np.asarray(series, dtype=float)
    if len(s) < season + h + 8:
        return moving_average(s, h, 8)
    recent = s[-8:].mean()
    ly = s[-season - 8:-season].mean()
    ratio = recent / ly if ly > 0 else 1.0
    ratio = float(np.clip(ratio, 0.5, 2.0))
    p = np.array([s[len(s) + i - season] for i in range(h)]) * ratio
    sd = float(np.std(s[-26:] - s[-26 - season:-season] * ratio)) if len(s) > season + 26 else float(np.std(s))
    return p, _band(p, sd)
