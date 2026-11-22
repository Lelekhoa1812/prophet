"""Is this series a random walk? A variance-ratio test, so a price-like series is recognised before anyone hopes to beat naive.

Motivation vs Logic
Motivation: on prices, FX and rates the honest centre line is usually the last value: their changes are close to unpredictable,
and a foundation model trained on smoother series tends to lose to "repeat the last value" there (live: WTI and EUR/JPY both
ended baseline_better). Knowing that before the forecast changes what is worth trying (the range, not the line) and what the
answer should say.
Logic: Lo and MacKinlay (1988) compare the variance of q-step changes with q times the variance of one-step changes. Under a
random walk the ratio is 1; below 1 the changes revert, above 1 they persist. The z statistic is the heteroskedasticity-robust
one (volatility clusters in markets), and several horizons are judged jointly by Chow and Denning (1993): the largest |z|
against a Sidak-adjusted critical value, so testing four horizons does not inflate false alarms. Changes are log returns when
every value is positive, plain differences otherwise.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy.stats import norm

HORIZONS = (2, 5, 10, 20)
MIN_POINTS = 100        # fewer changes than this and the test has too little power to say anything
ALPHA = 0.05


def _vr_z(r: np.ndarray, q: int) -> tuple[float, float]:
    """Variance ratio VR(q) and its heteroskedasticity-robust z statistic (Lo and MacKinlay 1988, overlapping sums)."""
    n = len(r)
    mu = r.mean()
    d = r - mu
    var1 = float(d @ d) / (n - 1)
    sums = np.convolve(r, np.ones(q), mode="valid") - q * mu      # overlapping q-step changes, demeaned
    m = q * (n - q + 1) * (1 - q / n)
    varq = float(sums @ sums) / m
    vr = varq / var1
    d2 = d * d
    denom = float(d2.sum()) ** 2
    theta = sum((2 * (q - j) / q) ** 2 * float(d2[j:] @ d2[:-j]) / denom for j in range(1, q))
    return vr, (vr - 1) / math.sqrt(theta) if theta > 0 else 0.0


def random_walk_test(values: Any, horizons: tuple[int, ...] = HORIZONS) -> dict[str, Any] | None:
    """Variance ratios at several horizons and a joint verdict:
    ``consistent_with_random_walk`` (no horizon departs from 1 after adjustment), ``mean_reverting`` (the strongest
    departure is below 1) or ``persistent`` (above 1). None when the series is too short or constant."""
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < MIN_POINTS + 1:
        return None
    logs = bool((x > 0).all())
    r = np.diff(np.log(x)) if logs else np.diff(x)
    if np.allclose(r, r[0]):
        return None
    qs = [q for q in horizons if q < len(r) // 5]
    if not qs:
        return None
    rows = {}
    for q in qs:
        vr, z = _vr_z(r, q)
        rows[q] = {"variance_ratio": round(vr, 3), "z": round(z, 2)}
    crit = float(norm.ppf(1 - (1 - (1 - ALPHA) ** (1 / len(qs))) / 2))     # Chow-Denning with a Sidak adjustment
    q_star = max(rows, key=lambda q: abs(rows[q]["z"]))
    z_star = rows[q_star]["z"]
    if abs(z_star) < crit:
        verdict = "consistent_with_random_walk"
        reading = ("Its changes behave like a random walk: no tested horizon shows reversal or persistence beyond chance, so "
                   "the last value is the honest centre line and the range carries the forecast.")
    elif rows[q_star]["variance_ratio"] < 1:
        verdict = "mean_reverting"
        reading = (f"Its changes tend to reverse (variance ratio {rows[q_star]['variance_ratio']} at {q_star} steps), so a "
                   "model can improve on repeating the last value.")
    else:
        verdict = "persistent"
        reading = (f"Its changes tend to continue (variance ratio {rows[q_star]['variance_ratio']} at {q_star} steps), so "
                   "momentum or trend carries information a model can use.")
    return {"verdict": verdict, "changes": "log returns" if logs else "differences", "n_changes": int(len(r)),
            "by_horizon": {str(q): v for q, v in rows.items()}, "max_abs_z": abs(z_star), "critical_value": round(crit, 2),
            "test": "Lo-MacKinlay heteroskedasticity-robust variance ratio, Chow-Denning joint test at 5 percent",
            "reading": reading}
