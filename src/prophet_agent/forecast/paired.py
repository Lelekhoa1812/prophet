"""Paired bootstrap of a relative error gain, shared by the rolling backtest and the control lab.

Motivation vs Logic
Motivation: "TimesFM beats the best rule" is only worth saying when the gap survives resampling of the origins. Two places
need the same arithmetic, and two copies would drift apart.
Logic: take the per-origin (or per-fold) errors of a reference and a candidate, difference them pair by pair, and resample
the differences with a seeded generator, so the same inputs always give the same interval. The gain is the mean difference
relative to the reference's mean error: positive means the candidate has the lower error.
"""
from __future__ import annotations

import math

import numpy as np


def bootstrap_gain(base: np.ndarray, cand: np.ndarray, seed: int, confidence: float, resamples: int) -> dict[str, float]:
    """Relative mean gain of ``cand`` over ``base`` with a seeded percentile bootstrap interval at ``confidence``.

    Returns gain, lo and hi as fractions (0.05 means 5 percent lower error). NaN values when the reference mean is not
    positive, because a relative gain is then undefined. Pairs with a NaN on either side are dropped first.
    """
    base = np.asarray(base, dtype=float)
    cand = np.asarray(cand, dtype=float)
    keep = np.isfinite(base) & np.isfinite(cand)
    diffs = (base - cand)[keep]
    mean_base = float(base[keep].mean()) if keep.any() else math.nan
    if diffs.size == 0 or not mean_base > 0:
        return {"gain": math.nan, "lo": math.nan, "hi": math.nan}
    rng = np.random.default_rng(seed)
    picks = rng.integers(0, len(diffs), size=(resamples, len(diffs)))
    means = diffs[picks].mean(axis=1)
    tail = (1 - confidence) / 2 * 100
    lo, hi = np.percentile(means, [tail, 100 - tail]) / mean_base
    return {"gain": float(diffs.mean()) / mean_base, "lo": float(lo), "hi": float(hi)}
