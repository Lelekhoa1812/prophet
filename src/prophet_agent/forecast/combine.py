"""Inverse-MSE combination of two point forecasts, with weights fitted on held-out errors for each horizon third.

Motivation vs Logic
Motivation: when TimesFM and a simple rule each win on different series or different horizons, a weighted blend can beat
both. Weights fitted on the same errors they are scored on would flatter the blend, so the weights must come only from
errors the blend has not been scored on.
Logic: for each horizon third, a model's weight is its inverse mean squared error, normalised across the two models. A model
with half the error variance gets twice the weight. With no earlier errors at all the weights are equal (0.5 each), which
is the honest default before any evidence exists. The caller decides which errors count as earlier; this module only
computes the weights and the blend.
"""
from __future__ import annotations

from typing import Any

import numpy as np

_EPS = 1e-12


def _thirds(horizon: int) -> list[np.ndarray]:
    edges = np.linspace(0, horizon, min(3, horizon) + 1).round().astype(int)
    return [np.arange(edges[i], edges[i + 1]) for i in range(len(edges) - 1) if edges[i + 1] > edges[i]]


def combine(point_a: Any, point_b: Any, errors_a: Any, errors_b: Any) -> dict[str, Any]:
    """Blend ``point_a`` and ``point_b`` (shape (horizon,)) with inverse-MSE weights per horizon third.

    ``errors_a`` and ``errors_b`` are held-out errors shaped (origins, horizon), actual minus forecast, from origins
    earlier than the one being forecast. Zero rows (or None) give equal weights.
    Returns {"point", "weights_a", "weights_b"} where the weights are lists with one value per horizon third.
    """
    a = np.asarray(point_a, dtype=float).reshape(-1)
    b = np.asarray(point_b, dtype=float).reshape(-1)
    if a.shape != b.shape:
        raise ValueError("the two point forecasts must cover the same horizon")
    h = len(a)
    ea = np.zeros((0, h)) if errors_a is None else np.asarray(errors_a, dtype=float).reshape(-1, h)
    eb = np.zeros((0, h)) if errors_b is None else np.asarray(errors_b, dtype=float).reshape(-1, h)
    w_a = np.full(h, 0.5)
    w_b = np.full(h, 0.5)
    weights_a: list[float] = []
    weights_b: list[float] = []
    have = len(ea) > 0 and len(eb) > 0
    for idx in _thirds(h):
        if have:
            mse_a = float(np.mean(ea[:, idx] ** 2)) + _EPS
            mse_b = float(np.mean(eb[:, idx] ** 2)) + _EPS
            wa = (1 / mse_a) / (1 / mse_a + 1 / mse_b)
        else:
            wa = 0.5
        w_a[idx], w_b[idx] = wa, 1 - wa
        weights_a.append(round(float(wa), 4))
        weights_b.append(round(1 - float(wa), 4))
    return {"point": w_a * a + w_b * b, "weights_a": weights_a, "weights_b": weights_b}
