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
weight sees the errors it is judged on. In each horizon third a candidate's raw weight is its inverse mean WIS on those
windows (or, with method "stack", the point of the simplex that minimises the past WIS of the weighted quantile average). The
weight used is lam * raw + (1 - lam) / N with lam = k / (k + KAPPA), where k is the number of complete windows, so a short
backtest stays close to equal weights. Quantiles are averaged with those weights (Vincentisation). The combination is then
scored on the later origins exactly as each candidate was, beside equal weights, beside the honest selection rule (at each
origin, the candidate with the best past score) and beside the best single candidate (chosen on the same windows, so its
score flatters it); each candidate is reported on those same windows. A price or an index is scored relative to its level
at each origin (``relative``), so the recent, higher-priced windows do not dominate. The forecast from the last origin uses
weights from every complete window. Its band is then checked against the combination's past errors laid on today's median:
when its own widths would have held fewer than 75 per cent of them, each tail is widened by a one-sided split conformal scale
per horizon third (never narrowed, at most threefold; misses above widen the upper tail only). The old combined bands are not
used for this, because members whose bands come from past errors already carry the old misses (the cash rate's 2022 rises
widened a 24-month band to 0 to 20 per cent that way).
Measured (2026-11-28, real staging data, monthly; candidates: the rules, the market forward or path, the structural models):
gold (COMEX front month 2008-09 to 2026-10, with GVZ) at 12 and 3 months, the S&P 500 (1990-04 to 2026-10, with VIX) at 24
and 6 months, Australian unemployment (ABS, 200 origins from 2008) and the cash rate (RBA F1, origins from 2016) at 24 months.
Mean WIS of the combination against equal weights: stacked +1.5, +0.3, -1.4, -0.6, +2.4, +0.5 per cent (positive is worse);
inverse-score +0.2, 0.0, +0.04, 0.0, -0.7, +0.3; trimmed (within 10 per cent of the best) +5.7, -0.2, +0.2, +0.2, +3.4, +4.5.
Equal weights are hard to beat (the forecast-combination puzzle); inverse-score weights shrunk towards equal match them and
still move away from a candidate that keeps failing, so they are the default. Stacked weights beat the best-so-far selection
by 0.9 to 5.7 per cent but lose to equal weights on four of six. KAPPA 8, 24, 64 and 200 gave stacking -4.3, -2.9, -1.5 and
-0.4 per cent against equal weights on gold at 12 months.
Every figure here is measured on the values given; none is assumed.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from prophet_agent.forecast.metrics import QS, wis

KAPPA = 64.0                 # origins at which the backtest weights and equal weights count the same (measured, see docstring)
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
    prepare: Callable[[list[int], int], None] | None = None   # told every origin first (a model can batch its calls)


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


def weights(wis_rows: np.ndarray, n_cands: int, h: int, *, q_past: np.ndarray | None = None,
            a_past: np.ndarray | None = None, norms: np.ndarray | None = None,
            kappa: float = KAPPA, method: str = "stack") -> tuple[np.ndarray, float]:
    """Weights (n_cands, h) from past windows, and lam, the share given to the backtest over equal weights.

    With ``q_past`` (origins, n_cands, h, 9) and ``a_past`` (origins, h), the weights in each horizon third are stacked: the
    point of the simplex that minimises the mean WIS of the combined quantiles on those windows (each window's WIS divided by
    its norm), found by a softmax parametrisation. Without them (or with too few complete windows) they are inverse mean
    WIS from ``wis_rows`` (origins, n_cands, h). Inverse-score weights stay close to equal when the scores differ by ten per
    cent, which is the usual gap between forecasts of a market price, so stacking is the default. Either way the result is
    lam * raw + (1 - lam) / N with lam = k / (k + KAPPA). NaN marks a candidate that failed at an origin."""
    k = wis_rows.shape[0]
    equal = np.full((n_cands, h), 1.0 / n_cands)
    if k < MIN_WEIGHT_ORIGINS:
        return equal, 0.0
    lam = k / (k + kappa)
    w = np.empty((n_cands, h))
    use = None
    if q_past is not None and a_past is not None:
        okc = np.isfinite(q_past).all(axis=(2, 3))                       # (origins, candidates)
        use = okc.sum(axis=1) >= 2                                        # windows with at least two candidates
        if use.sum() < MIN_WEIGHT_ORIGINS:
            use = None
    for idx in _thirds(h):
        if method == "trimmed":
            m = np.nanmean(wis_rows[:, :, idx], axis=(0, 2))
            keep = np.isfinite(m) & (m <= np.nanmin(m) * (1 + TRIM))
            raw = keep / keep.sum() if keep.any() else np.full(n_cands, 1.0 / n_cands)
            w[:, idx] = raw[:, None]
            continue
        if use is not None:
            raw = _stack(q_past[use][:, :, idx], a_past[use][:, idx],
                         None if norms is None else np.asarray(norms)[use])
        else:
            m = np.nanmean(wis_rows[:, :, idx], axis=(0, 2))
            m = np.where(np.isfinite(m) & (m > 0), m, np.nan)
            inv = np.where(np.isfinite(m), 1.0 / np.where(np.isfinite(m), m, 1.0), 0.0)
            raw = inv / inv.sum() if inv.sum() > 0 else np.full(n_cands, 1.0 / n_cands)
        w[:, idx] = (lam * raw + (1 - lam) / n_cands)[:, None]
    return w / w.sum(axis=0, keepdims=True), float(lam)


def _weights_for(method, W, C, h, Q, A, norm, kappa) -> tuple[np.ndarray, float]:
    if method == "equal":
        return np.full((C, h), 1.0 / C), 0.0
    if method == "stack":
        return weights(W, C, h, q_past=Q, a_past=A, norms=norm, kappa=kappa)
    return weights(W, C, h, kappa=kappa, method=method)


def _stack(q: np.ndarray, a: np.ndarray, norms: np.ndarray | None) -> np.ndarray:
    """Simplex weights minimising the mean (normed) WIS of the weighted quantile average over the windows given. A candidate
    that did not forecast in a window is left out of that window's average, and the others' weights are renormalised."""
    from scipy import optimize
    C = q.shape[1]
    d = np.ones(len(a)) if norms is None else np.maximum(np.abs(norms), 1e-12)
    ok = np.isfinite(q).all(axis=(2, 3)).astype(float)                   # (windows, candidates)
    q0 = np.nan_to_num(q)

    def loss(z):
        e = np.exp(z - z.max())
        w = (e / e.sum())[None, :] * ok
        w = w / np.maximum(w.sum(axis=1, keepdims=True), 1e-12)
        comb = np.sort(np.einsum("kc,kchq->khq", w, q0), axis=-1)
        return float(np.mean(wis(a, comb) / d[:, None]))

    best = optimize.minimize(loss, np.zeros(C), method="Nelder-Mead",
                             options={"xatol": 1e-3, "fatol": 1e-7, "maxiter": 200 * C})
    e = np.exp(best.x - best.x.max())
    return e / e.sum()


def combine(qs: np.ndarray, w: np.ndarray, ok: np.ndarray | None = None) -> np.ndarray:
    """Weighted average of quantiles qs (n_cands, h, 9) with weights w (n_cands, h); candidates not ok are left out."""
    w = np.array(w, dtype=float)
    if ok is not None:
        w = w * np.asarray(ok, dtype=float)[:, None]
    w = w / np.maximum(w.sum(axis=0, keepdims=True), 1e-12)
    out = np.einsum("ch,chq->hq", w, np.nan_to_num(qs))
    return np.sort(out, axis=1)


def _score(actual: np.ndarray, q: np.ndarray, scale: float, norm: float = 1.0) -> dict[str, Any]:
    s = wis(actual, q) / norm
    return {"wis": s, "ae": np.abs(actual - q[:, 4]) / norm / scale,
            "inside": ((actual >= q[:, 0]) & (actual <= q[:, 8])).astype(float)}


def _summary(rows: list[dict]) -> dict[str, float]:
    if not rows:
        return {"wis": float("nan"), "mase": float("nan"), "coverage": float("nan"), "origins": 0}
    return {"wis": float(np.mean([r["wis"].mean() for r in rows])), "mase": float(np.mean([r["ae"].mean() for r in rows])),
            "coverage": float(np.mean([r["inside"].mean() for r in rows])), "origins": len(rows)}


def run(values: Any, horizon: int, candidates: list[Candidate], *, n_origins: int = 12, step: int | None = None,
        min_train: int = 24, season: int = 1, relative: bool = False, method: str = "inverse",
        kappa: float = KAPPA) -> dict[str, Any]:
    """Score the candidates, combine them, and forecast from the end of ``values``.

    ``season`` sets the MASE scale (the mean absolute change over that many periods in the history before the first origin).
    ``relative`` scores each window relative to the last value known at its origin (for a price or an index, whose level
    moves by multiples over the history, so recent windows would otherwise dominate); WIS is then a share of the level.
    ``method`` is "inverse" (inverse mean WIS, the default), "stack" (simplex weights that minimise past WIS), "trimmed"
    (equal weights over the candidates within TRIM of the best) or "equal".
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
    if method not in ("stack", "inverse", "trimmed", "equal"):
        raise ValueError("method must be 'stack', 'inverse', 'trimmed' or 'equal'")
    hist0 = y[:orig[0]]
    m = max(1, int(season))
    diffs = np.abs(hist0[m:] - hist0[:-m])
    if relative:
        diffs = diffs / np.maximum(np.abs(hist0[:-m]), 1e-12)
    scale = float(np.mean(diffs)) if len(hist0) > m else 1.0
    scale = scale if np.isfinite(scale) and scale > 0 else 1.0
    C = len(candidates)
    for c in candidates:
        if c.prepare is not None:
            try:
                c.prepare(list(orig) + [len(y)], h)
            except Exception:            # a failed preparation shows up as failed forecasts below
                pass
    Q = np.full((len(orig), C, h, len(QS)), np.nan)
    failures: dict[str, list[str]] = {n: [] for n in names}
    for i, t in enumerate(orig):
        for j, c in enumerate(candidates):
            try:
                Q[i, j] = _clean(c.predict(t, h), h)
            except Exception as exc:     # a candidate's own failure is recorded, never fatal to the others
                failures[c.name].append(f"origin {t}: {str(exc)[:120]}")
    actual = np.stack([y[t:t + h] for t in orig])
    norm = np.array([abs(y[t - 1]) if relative else 1.0 for t in orig])
    norm = np.where(norm > 0, norm, 1.0)
    W = np.stack([wis(actual[i][None, :], Q[i]) / norm[i] for i in range(len(orig))])   # (origins, C, h); NaN = failed
    per_cand = {n: [] for n in names}
    for i in range(len(orig)):
        for j, n in enumerate(names):
            if np.isfinite(Q[i, j]).all():
                per_cand[n].append(_score(actual[i], Q[i, j], scale, norm[i]))
    stack = method == "stack"
    refit = max(1, len(orig) // 40)          # stacked weights are refitted every few origins on long backtests
    cached: tuple[int, np.ndarray] | None = None
    # Combination at each origin from windows complete before it.
    comb_rows, eq_rows, pick_rows = [], [], []
    for i, t in enumerate(orig):
        done = [k for k, s in enumerate(orig) if s + h <= t]
        ok = np.isfinite(Q[i]).all(axis=(1, 2))
        if len(done) < MIN_WEIGHT_ORIGINS or not ok.any():
            continue
        if cached is None or not stack or i - cached[0] >= refit:
            w, _ = _weights_for(method, W[done], C, h, Q[done], actual[done], norm[done], kappa)
            cached = (i, w)
        w = cached[1]
        qc = combine(Q[i], w, ok)
        qe = combine(Q[i], np.full((C, h), 1.0 / C), ok)
        comb_rows.append((i, qc))
        eq_rows.append((i, qe))
        past = np.nanmean(W[done], axis=(0, 2))
        past = np.where(ok & np.isfinite(past), past, np.inf)
        pick_rows.append((i, Q[i, int(np.argmin(past))]))
    combo = [_score(actual[i], q, scale, norm[i]) for i, q in comb_rows]
    equal = [_score(actual[i], q, scale, norm[i]) for i, q in eq_rows]
    picked = [_score(actual[i], q, scale, norm[i]) for i, q in pick_rows]
    scored_idx = [i for i, _ in comb_rows]
    on_same = {n: _summary([_score(actual[i], Q[i, names.index(n)], scale, norm[i]) for i in scored_idx
                            if np.isfinite(Q[i, names.index(n)]).all()]) for n in names}
    # Each candidate is reported on the windows the combination was scored on, so the figures compare; its score over every
    # origin is kept beside it.
    table = [{"name": n, "kind": candidates[names.index(n)].kind, **on_same[n], "all_origins": _summary(per_cand[n]),
              "failed_origins": len(failures[n]), "note": candidates[names.index(n)].note} for n in names]
    best = min((n for n in names if on_same[n]["origins"]), key=lambda n: on_same[n]["wis"], default=None)
    combo_cov = _summary(combo)["coverage"]
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
    w_final, lam_final = _weights_for(method, W[done], C, h, Q[done], actual[done], norm[done], kappa)
    w_used = w_final * ok[:, None]
    w_used = w_used / w_used.sum(axis=0, keepdims=True)
    q_final = combine(final_q, w_final, ok)
    # Band repair (split conformal, never narrower): would the final band's own widths have held the combination's past
    # errors? Members' bands built from past errors already carry the old misses, so the old combined bands' coverage would
    # count them twice (the cash rate's 2022 rises widened a 24-month band to 0-20 per cent that way).
    held, scales, widened = None, None, False
    if combo:
        lv = abs(y[-1]) if relative and y[-1] != 0 else 1.0
        errs = np.stack([(actual[i] - q[:, 4]) / norm[i] for i, q in comb_rows])            # (windows, h)
        med = q_final[:, 4] / lv
        cur = np.stack([med + errs[j] for j in range(len(errs))])                          # past errors on today's median
        held = float(np.mean((cur >= q_final[:, 0] / lv) & (cur <= q_final[:, 8] / lv)))
        if held < TARGET_COVERAGE - COVERAGE_TOLERANCE:
            lo_s, hi_s = side_scales(cur, np.broadcast_to(q_final / lv, (len(errs),) + q_final.shape))
            scales = {"lower": [round(x, 3) for x in lo_s], "upper": [round(x, 3) for x in hi_s]}
            widened = any(x > 1.0 for x in lo_s + hi_s)
            if widened:
                q_final = apply_side_scales(q_final, lo_s, hi_s)
    cs, es, ps = _summary(combo), _summary(equal), _summary(picked)
    return {
        "horizon": h, "origins": [int(o) for o in orig], "step": step, "scored_origins": len(combo),
        "mase_scale": scale, "candidates": table,
        "combination": {**cs, "vs_equal_wis_pct": _gain(es["wis"], cs["wis"]),
                        "vs_best_so_far_wis_pct": _gain(ps["wis"], cs["wis"]),
                        "vs_best_single_wis_pct": _gain(on_same[best]["wis"], cs["wis"]) if best else None,
                        "best_single": best, "best_single_on_same_origins": on_same.get(best)},
        "equal_weights": es,
        "best_so_far": {**ps, "rule": "at each origin, the candidate with the lowest mean WIS on the windows complete before it"},
        "weights": [{"name": n, "kind": candidates[j].kind, "weight": round(float(w_used[j].mean()), 4),
                     "by_third": [round(float(w_used[j, idx].mean()), 4) for idx in _thirds(h)]}
                    for j, n in enumerate(names)],
        "shrinkage": {"lam": round(lam_final, 4), "kappa": kappa, "complete_windows": len(done), "method": method,
                      "relative": relative},
        "band": {"coverage_scored": cs["coverage"], "target": TARGET_COVERAGE, "widened": widened, "scales": scales,
                 "final_band_held_past_errors": None if held is None else round(held, 3)},
        "forecast": {"quantiles": q_final.tolist(), "median": q_final[:, 4].tolist(), "p10": q_final[:, 0].tolist(),
                     "p90": q_final[:, 8].tolist()},
        "members": {n: final_q[j].tolist() for j, n in enumerate(names) if ok[j]},
        "failures": {n: f[:3] for n, f in failures.items() if f}, "final_failures": final_fail,
        "method": ("each candidate forecast from the same origins; scored by WIS (a CRPS approximation from the nine "
                   "quantiles), median absolute error over the history's mean absolute change (MASE) and 10-90 band "
                   "coverage; weights per horizon third from windows complete before each origin ("
                   + ("stacked: the simplex weights minimising past WIS" if stack else "inverse mean WIS")
                   + f"), shrunk towards equal weights by k/(k+{kappa:g}); quantiles are averaged with those weights"
                   + ("; each window scored relative to the level at its origin" if relative else "")),
    }


TRIM = 0.10                  # "trimmed": equal weights over the candidates within this share of the best past mean WIS
MAX_SIDE_SCALE = 3.0         # a tail is never widened more than threefold; a regime the past windows never saw is not fixed by it


def side_scales(actual: np.ndarray, q: np.ndarray) -> tuple[list[float], list[float]]:
    """One-sided split conformal scales per horizon third: the lower tail's distance from the median is multiplied by the
    finite-sample 90 per cent quantile of (median - actual) / (median - q10) over the windows, the upper tail likewise with
    (actual - median) / (q90 - median); each clipped to [1, MAX_SIDE_SCALE]. Misses on one side widen that side only."""
    med = q[..., 4]
    lo = np.maximum(med - q[..., 0], 1e-9)
    hi = np.maximum(q[..., 8] - med, 1e-9)
    r_lo, r_hi = (med - actual) / lo, (actual - med) / hi
    out_lo, out_hi = [], []
    for idx in _thirds(actual.shape[1]):
        for r, out in ((r_lo, out_lo), (r_hi, out_hi)):
            cell = np.sort(r[:, idx].ravel())
            n = cell.size
            k = min(n, int(np.ceil((n + 1) * 0.9)))
            out.append(float(np.clip(cell[k - 1], 1.0, MAX_SIDE_SCALE)) if n else 1.0)
    return out_lo, out_hi


def apply_side_scales(q: np.ndarray, lo: list[float], hi: list[float]) -> np.ndarray:
    out = q.copy()
    med = q[:, 4:5]
    for idx, a, b in zip(_thirds(q.shape[0]), lo, hi):
        below = q[idx] < med[idx]
        out[idx] = np.where(below, med[idx] + (q[idx] - med[idx]) * a, med[idx] + (q[idx] - med[idx]) * b)
    return np.sort(out, axis=1)


def _gain(base: float, new: float) -> float | None:
    if not (np.isfinite(base) and np.isfinite(new)) or base <= 0:
        return None
    return round(100.0 * (base - new) / base, 2)
