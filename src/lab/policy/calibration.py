"""Calibrate an agent's stated chance against how often its alerts really proved useful.

Motivation vs Logic
Motivation: the shortfall agent states a 97% chance of running short, yet alerts at that level were useful 56% of the
time, because the model's chance is 'demand exceeds position' while a useful alert needs a costly stockout. Pricing false
alarms with the stated chance understates their cost, and showing it as a fact misleads.
Logic: fit a monotone (non-decreasing) map from stated chance to observed usefulness with pool-adjacent-violators on the
client's own history, per agent. The map prices false alarms and is shown to the operator as 'alerts like this were useful
x% of the time'. With too little history it returns the stated chance unchanged.
"""
from __future__ import annotations

import numpy as np

MIN_POINTS = 300


def fit(p: np.ndarray, useful: np.ndarray, bins: int = 20) -> dict | None:
    """Return {"x": [...], "y": [...]} knots of the calibrated map, or None when there is too little evidence."""
    p = np.asarray(p, dtype=float); y = np.asarray(useful, dtype=float)
    ok = np.isfinite(p)
    p, y = p[ok], y[ok]
    if len(p) < MIN_POINTS:
        return None
    order = np.argsort(p)
    p, y = p[order], y[order]
    edges = np.array_split(np.arange(len(p)), bins)
    xs = np.array([p[i].mean() for i in edges if len(i)])
    ys = np.array([y[i].mean() for i in edges if len(i)])
    ws = np.array([len(i) for i in edges if len(i)], dtype=float)
    # pool adjacent violators
    vals, wts, grp = list(ys), list(ws), [[k] for k in range(len(ys))]
    k = 0
    while k < len(vals) - 1:
        if vals[k] > vals[k + 1] + 1e-12:
            tot = wts[k] + wts[k + 1]
            vals[k] = (vals[k] * wts[k] + vals[k + 1] * wts[k + 1]) / tot
            wts[k] = tot; grp[k] += grp[k + 1]
            del vals[k + 1], wts[k + 1], grp[k + 1]
            k = max(k - 1, 0)
        else:
            k += 1
    out = np.empty(len(ys))
    for v, g in zip(vals, grp):
        out[g] = v
    return {"x": [round(float(v), 4) for v in xs], "y": [round(float(v), 4) for v in out]}


def apply(cal: dict | None, p: float | None) -> float | None:
    if p is None or not cal:
        return p
    return float(np.interp(p, cal["x"], cal["y"]))
