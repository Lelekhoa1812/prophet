"""Forecast accuracy metrics. Ported from src/lab/engine/metrics.py. All take arrays shaped (series, horizon)."""
from __future__ import annotations

import numpy as np

QS = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])


def wape(actual: np.ndarray, pred: np.ndarray) -> float:
    d = np.abs(actual).sum()
    return float(np.abs(actual - pred).sum() / d) if d > 0 else float("nan")


def bias(actual: np.ndarray, pred: np.ndarray) -> float:
    d = np.abs(actual).sum()
    return float((pred - actual).sum() / d) if d > 0 else float("nan")


def pinball(actual: np.ndarray, q: np.ndarray) -> float:
    """Mean pinball loss across the nine quantiles, normalised by mean demand (lower is better)."""
    a = actual[..., None]
    diff = a - q
    loss = np.maximum(QS * diff, (QS - 1) * diff)
    d = np.abs(actual).mean()
    return float(loss.mean() / d) if d > 0 else float("nan")


def coverage(actual: np.ndarray, q: np.ndarray, lo: int = 0, hi: int = 8) -> float:
    """Share of actuals inside the band from quantile index lo to hi (default 10 to 90, nominal 80%)."""
    return float(((actual >= q[..., lo]) & (actual <= q[..., hi])).mean())


def mase(actual: np.ndarray, pred: np.ndarray, hist_scale: np.ndarray) -> float:
    s = np.where(hist_scale > 0, hist_scale, np.nan)
    return float(np.nanmean(np.abs(actual - pred).mean(axis=1) / s))
