"""Inverse-MSE combination of two point forecasts, with weights fitted on held-out errors for each horizon third, and an
equal-weight blend that is adopted only on a backtest win.

Motivation vs Logic
Motivation: when TimesFM and a simple rule each win on different series or different horizons, a weighted blend can beat
both. Weights fitted on the same errors they are scored on would flatter the blend, so the weights must come only from
errors the blend has not been scored on.
Logic: for each horizon third, a model's weight is its inverse mean squared error, normalised across the two models. A model
with half the error variance gets twice the weight. With no earlier errors at all the weights are equal (0.5 each), which
is the honest default before any evidence exists. The caller decides which errors count as earlier; this module only
computes the weights and the blend.
The equal-weight blend fits nothing: it is the plain mean of the two points. Choosing it as a centre is a selection on the
backtest windows (see adopt_equal_blend), so its result is labelled as selected, not as an unbiased test.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from prophet_agent.forecast.metrics import paired_signflip
from prophet_agent.forecast.paired import bootstrap_gain

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


def equal_blend(point_a: Any, point_b: Any) -> np.ndarray:
    """The plain mean of two point forecasts of the same horizon, with no weights and no errors.

    Motivation vs Logic
    Motivation: an equal blend needs no fitting, so it cannot be tuned on the windows it is scored on. It is the centre line
    the report can offer when TimesFM and the best simple rule each miss in different directions.
    Logic: (a + b) / 2 element by element. Nothing else enters, so the same two points always give the same blend.
    """
    a = np.asarray(point_a, dtype=float).reshape(-1)
    b = np.asarray(point_b, dtype=float).reshape(-1)
    if a.shape != b.shape:
        raise ValueError("the two point forecasts must cover the same horizon")
    return (a + b) / 2.0


def _beats(base: np.ndarray, cand: np.ndarray, seed: int, confidence: float, resamples: int, win_share: float) -> dict[str, Any]:
    """Whether ``cand`` beats ``base`` on the same origins: a one-sided sign-flip test rejects equal error at the level
    implied by ``confidence`` in the candidate's favour, and ``cand`` has the lower error on at least ``win_share`` of the
    origins.

    Motivation vs Logic
    Motivation: a lower mean error on a few origins is not a win. The verdict needs the same rule the rolling backtest uses
    for TimesFM against a baseline, so a blend and a forecast are judged by one standard. The percentile bootstrap held
    its level badly at 6 origins (audit A2-06), so the decision uses the exact sign-flip test.
    Logic: keep the origins where both errors are finite, count how many the candidate wins, and take the one-sided
    sign-flip p-value. The bootstrap interval is kept as a description. Both conditions must hold.
    """
    keep = np.isfinite(base) & np.isfinite(cand)
    n = int(keep.sum())
    wins = int((cand[keep] < base[keep]).sum())
    g = bootstrap_gain(base, cand, seed, confidence, resamples)
    st = paired_signflip(base, cand, seed=seed)
    alpha = (1 - confidence) / 2
    beats = bool(n and st["p_cand_better"] <= alpha and wins >= win_share * n)
    interval = [round(g["lo"] * 100, 2), round(g["hi"] * 100, 2)] if np.isfinite(g["lo"]) else None
    return {"beats": beats, "gain_pct": round(g["gain"] * 100, 2) if np.isfinite(g["gain"]) else None,
            "interval_pct": interval, "wins": wins, "of": n, "test": st["method"],
            "p_one_sided": round(st["p_cand_better"], 4) if n else None}


def adopt_equal_blend(err_timesfm: Any, err_blend: Any, err_rule: Any, *, seed: int, confidence: float, resamples: int,
                      win_share: float) -> dict[str, Any]:
    """Adopt the equal blend as the centre only if it beats TimesFM's centre AND the simple rule on the same origins.

    Motivation vs Logic
    Motivation: the blend is an option the planner can take, and taking it on a hunch would be a claim the data does not
    make. Adoption is therefore a decision made on the backtest, and the decision is itself a selection on those windows.
    Logic: each per-origin error (MASE) array is compared with the blend's on the same origins through _beats. The blend is
    adopted only when both comparisons pass. A tie, a loss to either, or too few origins means it is not adopted. The
    result says the adoption was selected on these windows, so it is never read as an unbiased figure.
    """
    t = np.asarray(err_timesfm, dtype=float).reshape(-1)
    b = np.asarray(err_blend, dtype=float).reshape(-1)
    r = np.asarray(err_rule, dtype=float).reshape(-1)
    if not (t.shape == b.shape == r.shape):
        raise ValueError("the errors must be given for the same origins")
    kw = dict(seed=seed, confidence=confidence, resamples=resamples, win_share=win_share)
    vs_t = _beats(t, b, **kw)
    vs_r = _beats(r, b, **kw)
    return {"adopt": bool(vs_t["beats"] and vs_r["beats"]), "vs_timesfm": vs_t, "vs_rule": vs_r}
