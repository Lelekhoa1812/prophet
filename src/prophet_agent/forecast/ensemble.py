"""Score every candidate forecast on the same rolling origins and combine them with weights earned on the backtest.

Motivation vs Logic
Motivation (owner decision 2026-11-28, Mission 10): the centre line has to be earned. A ledger audit of twelve panel missions
found every delivered centre line came from the target's own history, chosen by one comparison, while the market path, the
adopted drivers and the structural models never competed. Forecast-combination evidence (M4 and M5 competitions, central
bank evaluations) favours a combination of several models over a single winner picked on a short backtest, and favours
weights pulled towards equal unless the evidence for unequal weights is long.
Logic: a candidate is a function from an origin (the number of values known) and a horizon to nine quantiles per step (10 to
90 per cent). It may read anything known at that origin: its own history, a market curve, a driver. Every candidate is
forecast from the same origins, and each forecast is scored against what happened by the weighted interval score (WIS, a
quantile approximation of CRPS), the absolute error of its median, and whether the 10 to 90 per cent band held the actual.
Weights for a forecast made at an origin come only from forecasts whose whole window had been observed by that origin, so no
weight sees the errors it is judged on. A candidate's raw weight is its inverse mean WIS in each horizon third; the weight
used is lam * raw + (1 - lam) / N with lam = k / (k + KAPPA), where k is the number of origins behind it: with few origins the
combination is close to equal weights. The combined quantiles are the weighted average of the candidates' quantiles
(Vincentisation), sorted. The combination is then scored on the later origins exactly as each candidate was, beside equal
weights and the best single candidate (whose choice is made on the same origins, so its score flatters it). When the
combination's band holds fewer actuals than 80 per cent on those origins, its band is widened by split conformal scales per
horizon third, fitted on those origins (never narrowed), and the widening is reported. The forecast from the last origin
uses weights from every complete window.
Every figure here is measured on the values given; none is assumed.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from prophet_agent.forecast.calibrate import apply_scales, fit_scales
from prophet_agent.forecast.metrics import QS, wis

KAPPA = 8.0                  # origins at which the backtest weights and equal weights count the same
MIN_WEIGHT_ORIGINS = 3       # complete windows needed before weights move away from equal
TARGET_COVERAGE = 0.8
COVERAGE_TOLERANCE = 0.05    # a band this far under 80 per cent on the scored origins is widened
KINDS = ("rule", "model", "market", "structural", "driver")


@dataclass(frozen=True)
class Candidate:
    name: str
    kind: str                                         # one of KINDS
    predict: Callable[[int, int], np.ndarray]         # (origin, horizon) -> quantiles shaped (horizon, 9)
    note: str = ""


def _thirds(h: int) -> list[np.ndarray]:
    edges = np.linspace(0, h, min(3, h) + 1).round().astype(int)
    return [np.arange(edges[i], edges[i + 1]) for i in range(len(edges) - 1) if edges[i + 1] > edges[i]]


def origins_for(n: int, horizon: int, n_origins: int, step: int, min_train: int) -> list[int]:
    """Origins (values known) whose whole window lies inside the series, newest last, ``step`` apart."""
    last = n - horizon
    out = [last - step * k for k in range(n_origins)]
    return sorted(o for o in out if o >= min_train)


def _clean(q: Any, h: int) -> np.ndarray:
    a = np.asarray(q, dtype=float)
    if a.shape != (h, len(QS)) or not np.isfinite(a).all():
        raise ValueError(f"a candidate must return finite quantiles shaped ({h}, {len(QS)}), got {a.shape}")
    return np.sort(a, axis=1)


def weights(wis_rows: np.ndarray, n_cands: int, h: int) -> tuple[np.ndarray, float]:
    """Weights (n_cands, h) from past WIS rows shaped (origins, n_cands, h); NaN marks a candidate that failed there.
    Returns the weights and lam, the share given to the backtest over equal weights."""
    k = wis_rows.shape[0]
    equal = np.full((n_cands, h), 1.0 / n_cands)
    if k < MIN_WEIGHT_ORIGINS:
        return equal, 0.0
    lam = k / (k + KAPPA)
    w = np.empty((n_cands, h))
    for idx in _thirds(h):
        m = np.nanmean(wis_rows[:, :, idx], axis=(0, 2))
        m = np.where(np.isfinite(m) & (m > 0), m, np.nan)
        inv = np.where(np.isfinite(m), 1.0 / np.where(np.isfinite(m), m, 1.0), 0.0)
        raw = inv / inv.sum() if inv.sum() > 0 else np.full(n_cands, 1.0 / n_cands)
        w[:, idx] = (lam * raw + (1 - lam) / n_cands)[:, None]
    return w / w.sum(axis=0, keepdims=True), float(lam)


def combine(qs: np.ndarray, w: np.ndarray, ok: np.ndarray | None = None) -> np.ndarray:
    """Weighted average of quantiles qs (n_cands, h, 9) with weights w (n_cands, h); candidates not ok are left out."""
    w = np.array(w, dtype=float)
    if ok is not None:
        w = w * np.asarray(ok, dtype=float)[:, None]
    w = w / np.maximum(w.sum(axis=0, keepdims=True), 1e-12)
    out = np.einsum("ch,chq->hq", w, np.nan_to_num(qs))
    return np.sort(out, axis=1)


def _score(actual: np.ndarray, q: np.ndarray, scale: float) -> dict[str, Any]:
    s = wis(actual, q)
    return {"wis": s, "ae": np.abs(actual - q[:, 4]) / scale,
            "inside": ((actual >= q[:, 0]) & (actual <= q[:, 8])).astype(float)}


def _summary(rows: list[dict]) -> dict[str, float]:
    if not rows:
        return {"wis": float("nan"), "mase": float("nan"), "coverage": float("nan"), "origins": 0}
    return {"wis": float(np.mean([r["wis"].mean() for r in rows])), "mase": float(np.mean([r["ae"].mean() for r in rows])),
            "coverage": float(np.mean([r["inside"].mean() for r in rows])), "origins": len(rows)}


def run(values: Any, horizon: int, candidates: list[Candidate], *, n_origins: int = 12, step: int | None = None,
        min_train: int = 24, season: int = 1) -> dict[str, Any]:
    """Score the candidates, combine them, and forecast from the end of ``values``.

    ``season`` sets the MASE scale (the mean absolute change over that many periods in the history before the first origin).
    Refuses fewer than two candidates or too short a series. A candidate that fails at an origin is left out there and its
    failure is counted; one that fails at the final origin is left out of the forecast.
    """
    y = np.asarray(values, dtype=float)
    if len(candidates) < 2:
        raise ValueError("combination needs at least two candidates")
    names = [c.name for c in candidates]
    if len(set(names)) != len(names):
        raise ValueError("candidate names must be unique")
    for c in candidates:
        if c.kind not in KINDS:
            raise ValueError(f"candidate kind must be one of {KINDS}")
    h = int(horizon)
    step = max(1, h // 4) if step is None else int(step)
    orig = origins_for(len(y), h, n_origins, step, min_train)
    if len(orig) < MIN_WEIGHT_ORIGINS + 2:
        raise ValueError(f"the series is too short to score {h} steps ahead: {len(orig)} origins fit, "
                         f"{MIN_WEIGHT_ORIGINS + 2} are needed (shorten the horizon or the step)")
    hist0 = y[:orig[0]]
    m = max(1, int(season))
    scale = float(np.mean(np.abs(hist0[m:] - hist0[:-m]))) if len(hist0) > m else 1.0
    scale = scale if np.isfinite(scale) and scale > 0 else 1.0
    C = len(candidates)
    Q = np.full((len(orig), C, h, len(QS)), np.nan)
    failures: dict[str, list[str]] = {n: [] for n in names}
    for i, t in enumerate(orig):
        for j, c in enumerate(candidates):
            try:
                Q[i, j] = _clean(c.predict(t, h), h)
            except Exception as exc:     # a candidate's own failure is recorded, never fatal to the others
                failures[c.name].append(f"origin {t}: {str(exc)[:120]}")
    actual = np.stack([y[t:t + h] for t in orig])
    W = np.stack([wis(actual[i][None, :], Q[i]) for i in range(len(orig))])   # (origins, C, h); NaN where a candidate failed
    per_cand = {n: [] for n in names}
    for i in range(len(orig)):
        for j, n in enumerate(names):
            if np.isfinite(Q[i, j]).all():
                per_cand[n].append(_score(actual[i], Q[i, j], scale))
    # Combination at each origin from windows complete before it.
    comb_rows, eq_rows = [], []
    for i, t in enumerate(orig):
        done = [k for k, s in enumerate(orig) if s + h <= t]
        ok = np.isfinite(Q[i]).all(axis=(1, 2))
        if len(done) < MIN_WEIGHT_ORIGINS or not ok.any():
            continue
        w, _ = weights(W[done], C, h)
        qc = combine(Q[i], w, ok)
        qe = combine(Q[i], np.full((C, h), 1.0 / C), ok)
        comb_rows.append((i, qc))
        eq_rows.append((i, qe))
    combo = [_score(actual[i], q, scale) for i, q in comb_rows]
    equal = [_score(actual[i], q, scale) for i, q in eq_rows]
    scored_idx = [i for i, _ in comb_rows]
    table = []
    for n in names:
        table.append({"name": n, "kind": candidates[names.index(n)].kind, **_summary(per_cand[n]),
                      "failed_origins": len(failures[n])})
    on_same = {n: _summary([_score(actual[i], Q[i, names.index(n)], scale) for i in scored_idx
                            if np.isfinite(Q[i, names.index(n)]).all()]) for n in names}
    best = min((n for n in names if on_same[n]["origins"]), key=lambda n: on_same[n]["wis"], default=None)
    # Band repair on the scored origins (split conformal, never narrower).
    combo_cov = _summary(combo)["coverage"]
    scales, widened = None, False
    if combo and combo_cov < TARGET_COVERAGE - COVERAGE_TOLERANCE:
        acts = np.stack([actual[i] for i, _ in comb_rows])
        qq = np.stack([q for _, q in comb_rows])
        scales = [float(s) for s in fit_scales(acts, qq)]
        widened = any(s > 1.0 for s in scales)
    # Forecast from the end of the series.
    T = len(y)
    final_q = np.full((C, h, len(QS)), np.nan)
    final_fail = []
    for j, c in enumerate(candidates):
        try:
            final_q[j] = _clean(c.predict(T, h), h)
        except Exception as exc:
            final_fail.append({"name": c.name, "error": str(exc)[:200]})
    ok = np.isfinite(final_q).all(axis=(1, 2))
    if not ok.any():
        raise ValueError("no candidate could forecast from the end of the series: "
                         + "; ".join(f"{f['name']}: {f['error']}" for f in final_fail))
    done = list(range(len(orig)))
    w_final, lam_final = weights(W[done], C, h)
    w_used = w_final * ok[:, None]
    w_used = w_used / w_used.sum(axis=0, keepdims=True)
    q_final = combine(final_q, w_final, ok)
    if widened and scales:
        q_final = apply_scales(q_final, scales)
    cs, es = _summary(combo), _summary(equal)
    return {
        "horizon": h, "origins": [int(o) for o in orig], "step": step, "scored_origins": len(combo),
        "mase_scale": scale, "candidates": table,
        "combination": {**cs, "vs_equal_wis_pct": _gain(es["wis"], cs["wis"]),
                        "vs_best_single_wis_pct": _gain(on_same[best]["wis"], cs["wis"]) if best else None,
                        "best_single": best, "best_single_on_same_origins": on_same.get(best)},
        "equal_weights": es,
        "weights": [{"name": n, "kind": candidates[j].kind, "weight": round(float(w_used[j].mean()), 4),
                     "by_third": [round(float(w_used[j, idx].mean()), 4) for idx in _thirds(h)]}
                    for j, n in enumerate(names)],
        "shrinkage": {"lam": round(lam_final, 4), "kappa": KAPPA, "complete_windows": len(done)},
        "band": {"coverage_scored": cs["coverage"], "target": TARGET_COVERAGE, "widened": widened,
                 "scales": scales},
        "forecast": {"quantiles": q_final.tolist(), "median": q_final[:, 4].tolist(), "p10": q_final[:, 0].tolist(),
                     "p90": q_final[:, 8].tolist()},
        "members": {n: final_q[j].tolist() for j, n in enumerate(names) if ok[j]},
        "failures": {n: f[:3] for n, f in failures.items() if f}, "final_failures": final_fail,
        "method": ("each candidate forecast from the same origins; scored by WIS (a CRPS approximation from the nine "
                   "quantiles), median absolute error over the history's mean absolute change (MASE) and 10-90 band "
                   "coverage; weights are inverse mean WIS per horizon third from windows complete before each origin, "
                   f"shrunk towards equal weights by k/(k+{KAPPA:g}); quantiles are averaged with those weights"),
    }


def _gain(base: float, new: float) -> float | None:
    if not (np.isfinite(base) and np.isfinite(new)) or base <= 0:
        return None
    return round(100.0 * (base - new) / base, 2)
