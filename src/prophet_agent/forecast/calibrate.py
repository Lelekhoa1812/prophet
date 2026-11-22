"""Interval calibration: make the 10 to 90 percent band cover what it claims, measured on origins the band never saw.

Motivation vs Logic
Motivation: TimesFM's quantile bands are not guaranteed to cover the stated share. Live runs showed 72 percent coverage for
an 80 percent band on quake counts and 82 percent on CO2. A band that is too narrow makes every decision that uses it
over-confident, and no amount of point-forecast tuning fixes that. Split conformal prediction repairs the band from the model's own
past errors without touching its point forecast.
Logic: forecast several past, non-overlapping windows (newest last) using only data before each, and score every actual by how far
out it fell, as a multiple of the model's own half-band: r = (median - y) / (median - q10) below the median, (y - median) /
(q90 - median) above it. The scale s for a horizon bucket is the conformal quantile of r that reaches the target
coverage, with the finite-sample rank ceil((n + 1) * target). Every quantile's distance from the median is multiplied by s (never
below 1: a band is only widened, so a lucky sample cannot make a forecast look more certain). The first two thirds of the
origins fit the scales and the last third tests them, so the result reports raw and calibrated coverage on origins that were
not used to fit; the scales used for the forecast then come from all origins. Windows share a series, so they are not
independent, and conformal coverage assumes errors that look like the past: a regime change breaks it. Both are reported.

Decision rule (changed from a fixed 0.02 gain): use the calibrated band when the raw held-out coverage is more than 0.05
from target AND the calibrated held-out coverage is closer to target than the raw one. The fixed gain could not accept a
band that moved from 0.69 to 0.71 against a target of 0.80, which is the case that matters most. If the held-out coverage
is still more than 0.05 under target after the horizon-third scales, a second pass refits at a higher level (see the
``second_pass`` field). Scales never go below 1, so the band is never narrowed.

Adaptive conformal (added as a candidate): Adaptive Conformal Inference (Gibbs and Candes, 2021) moves the miscoverage level
alpha after each fit window: alpha rises when the band covered that window and falls when it missed, so a band that has
gone stale after a regime change is widened without waiting for a full refit. The DtACI extension (Gibbs and Candes, 2022)
runs several learning rates side by side and reports the one with the lowest recent pinball loss. Its scales are fitted on
the fit windows only and applied as fixed scales to the held-out windows, so the held-out windows never move the alpha path.
It is chosen only when it lands at least GAP_STEP closer to target than the method already chosen, and it is never narrower
than raw.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

from prophet_agent.forecast.rolling import MIN_HISTORY, backtest_season
from prophet_agent.forecast.tsb import expected_volume, is_intermittent, tsb_band

TARGET = 0.8                # coverage of the 10 to 90 percent band
GAP_STEP = 0.05             # a raw band this far from target is worth fixing; the same gap is the second-pass trigger
MAX_LEVEL = 0.99            # the highest level the second pass may fit at
BUCKETS = 3                 # horizon thirds get their own scale: error grows with the horizon
MIN_ORIGINS = 6
MAX_SCALE = 5.0             # a band more than five times wider is a model that has failed, not a band to publish
Q_LO, Q_MID, Q_HI = 0, 4, 8
ACI_GAMMAS = (0.005, 0.01, 0.05)  # learning rates run in parallel (DtACI, Gibbs and Candes 2022): slow, medium, fast
ACI_LOOKBACK = 4            # the expert used is the one with the lowest pinball loss over this many recent windows
ALPHA_LO, ALPHA_HI = 0.01, 0.5    # the miscoverage level alpha is clipped to this range, so the scale stays sensible


def _ratios(actual: np.ndarray, q: np.ndarray) -> np.ndarray:
    """How far each actual fell, in units of the model's half-band on the side it fell on. Shape of ``actual``."""
    med = q[..., Q_MID]
    lo = np.maximum(med - q[..., Q_LO], 1e-9 * max(float(np.abs(med).max()), 1.0))
    hi = np.maximum(q[..., Q_HI] - med, 1e-9 * max(float(np.abs(med).max()), 1.0))
    return np.where(actual < med, (med - actual) / lo, (actual - med) / hi)


def _buckets(horizon: int, buckets: int) -> list[np.ndarray]:
    edges = np.linspace(0, horizon, min(buckets, horizon) + 1).round().astype(int)
    return [np.arange(edges[i], edges[i + 1]) for i in range(len(edges) - 1) if edges[i + 1] > edges[i]]


def fit_scales(actual: np.ndarray, q: np.ndarray, target: float = TARGET, buckets: int = BUCKETS) -> list[float]:
    """One scale per horizon bucket from actuals (origins, horizon) and quantile forecasts (origins, horizon, 9)."""
    r = _ratios(actual, q)
    out = []
    for idx in _buckets(actual.shape[1], buckets):
        cell = np.sort(r[:, idx].ravel())
        n = cell.size
        k = min(n, math.ceil((n + 1) * target))
        out.append(float(min(max(cell[k - 1], 1.0), MAX_SCALE)))
    return out


def apply_scales(q: np.ndarray, scales: list[float]) -> np.ndarray:
    """Scale each quantile's distance from the median, by horizon bucket. ``q`` has shape (horizon, 9)."""
    out = q.copy()
    med = q[:, Q_MID:Q_MID + 1]
    for idx, s in zip(_buckets(q.shape[0], len(scales)), scales):
        out[idx] = med[idx] + (q[idx] - med[idx]) * s
    if (q >= 0).all():
        out = np.clip(out, 0, None)
    return out


def use_calibrated(raw_gap: float, cal_gap: float, max_scale: float, width_ratio: float) -> bool:
    """The decision rule: the raw band is more than GAP_STEP from target, the calibrated one is closer, it actually widened,
    and it did not blow the band up past MAX_SCALE."""
    return bool(raw_gap > GAP_STEP and cal_gap < raw_gap and max_scale > 1.0 and width_ratio <= MAX_SCALE)


EWMA_LAMBDA = 0.94          # RiskMetrics daily decay for the volatility of one-step changes
LEVELS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
Z90 = 1.2815515655446004    # standard normal quantile at 0.9


def ewma_sigma(hist: np.ndarray, lam: float = EWMA_LAMBDA) -> float:
    """Exponentially weighted volatility of one-step changes, using only ``hist`` (no look-ahead)."""
    d = np.diff(np.asarray(hist, dtype=float))
    if d.size == 0:
        return 0.0
    var = float(np.mean(d[: min(30, d.size)] ** 2))
    for v in d:
        var = lam * var + (1 - lam) * float(v) * float(v)
    return math.sqrt(max(var, 1e-18))


def _vol_scores(actual: np.ndarray, q: np.ndarray, sig: np.ndarray) -> np.ndarray:
    """|actual - median| in units of sigma * sqrt(step): the error a volatility-aware band has to cover."""
    steps = np.sqrt(np.arange(1, actual.shape[1] + 1))[None, :]
    return np.abs(actual - q[..., Q_MID]) / (sig[:, None] * steps)


def fit_vol_scales(actual: np.ndarray, q: np.ndarray, sig: np.ndarray, target: float = TARGET,
                   buckets: int = BUCKETS) -> list[float]:
    r = _vol_scores(actual, q, sig)
    out = []
    for idx in _buckets(actual.shape[1], buckets):
        cell = np.sort(r[:, idx].ravel())
        k = min(cell.size, math.ceil((cell.size + 1) * target))
        out.append(float(cell[k - 1]))
    return out


def vol_band(q: np.ndarray, sigma: float, scales: list[float]) -> np.ndarray:
    """Quantiles centred on the model's median with a half-width of scale * sigma * sqrt(step) at the 10/90 levels, and the
    inner levels spaced as a normal distribution would space them. Used only when it covers closer to target on held-out
    windows: a series whose volatility clusters (prices, rates) needs a band that follows the recent volatility."""
    out = q.copy()
    med = q[:, Q_MID]
    steps = np.sqrt(np.arange(1, q.shape[0] + 1))
    from statistics import NormalDist
    z = [NormalDist().inv_cdf(l) / Z90 for l in LEVELS]
    for idx, s in zip(_buckets(q.shape[0], len(scales)), scales):
        half = s * sigma * steps[idx]
        for j, zj in enumerate(z):
            out[idx, j] = med[idx] + zj * half
    if (q >= 0).all():
        out = np.clip(out, 0, None)
    return out


def _conformal_scale(pool: np.ndarray, alpha: float) -> float:
    """The score quantile at level 1 - alpha, with the finite-sample rank, clipped to [1, MAX_SCALE]. No past scores: the raw band."""
    n = pool.size
    if n == 0:
        return 1.0
    k = min(n, math.ceil((n + 1) * (1 - alpha)))
    return float(min(max(np.sort(pool)[k - 1], 1.0), MAX_SCALE))


def aci_scales(scores_by_window: Any, target: float = TARGET, gammas: tuple = ACI_GAMMAS, start: float | None = None,
               buckets: int = BUCKETS, lookback: int = ACI_LOOKBACK) -> dict:
    """Adaptive conformal scales (Gibbs and Candes 2021, DtACI in 2022), one per horizon bucket.

    ``scores_by_window`` has shape (windows, horizon), in time order: each entry is a normalised score (see ``_ratios``), so
    a scale of 1 is the raw band edge. For each bucket and each learning rate gamma, the miscoverage level starts at ``start``
    (default 1 - target) and, after each window, moves by gamma * (alpha_target - err), where err is the share of that window's
    steps that fell outside the band at the current level. The band for a window uses the conformal quantile at level
    1 - alpha of the scores from earlier windows only, so no window is scored with its own score. Each learning rate runs
    in parallel, and the one predicting each window is the one with the lowest pinball loss over the last ``lookback`` windows.
    The returned scale is the conformal quantile of all windows at the final alpha of the expert that would be used next.
    Alpha is clipped to [ALPHA_LO, ALPHA_HI]; scales are clipped to [1, MAX_SCALE].

    Returns {"scales": one per bucket, "alpha_final": the final alpha per bucket, "alpha_path": the alpha used for each window
    per bucket}.
    """
    s = np.asarray(scores_by_window, dtype=float)
    if s.ndim != 2 or s.shape[0] < 1:
        raise ValueError("aci_scales needs scores shaped (windows, horizon) with at least one window")
    alpha_target = 1 - target
    a0 = alpha_target if start is None else float(start)
    scales, alpha_final, paths = [], [], []
    for idx in _buckets(s.shape[1], buckets):
        sc = s[:, idx]
        n_win, g = sc.shape[0], len(gammas)
        alpha = np.full(g, a0)
        loss = np.zeros((n_win, g))
        path = []
        for t in range(n_win):
            pool = sc[:t].ravel()
            pred = np.array([_conformal_scale(pool, a) for a in alpha])
            recent = loss[max(0, t - lookback):t].sum(axis=0) if t > 0 else np.zeros(g)
            best = int(np.argmin(recent))
            path.append(float(alpha[best]))
            y = sc[t]
            for i in range(g):
                d = y - pred[i]
                loss[t, i] = float(np.mean(np.where(d >= 0, target * d, (target - 1) * d)))
                err = float(np.mean(y > pred[i]))
                alpha[i] = float(np.clip(alpha[i] + gammas[i] * (alpha_target - err), ALPHA_LO, ALPHA_HI))
        best = int(np.argmin(loss[max(0, n_win - lookback):].sum(axis=0)))
        scales.append(_conformal_scale(sc.ravel(), float(alpha[best])))
        alpha_final.append(float(alpha[best]))
        paths.append(path)
    return {"scales": scales, "alpha_final": alpha_final, "alpha_path": paths}


def wilson(hits: int, n: int, conf: float = 0.9) -> list[float]:
    """Wilson score interval for a coverage share; with few windows a coverage of 0.7 is not proof the band is too narrow."""
    if n <= 0:
        return [0.0, 1.0]
    return _wilson_share(hits / n, n, conf)


def _wilson_share(p: float, n: int, conf: float = 0.9) -> list[float]:
    """Wilson interval for a share p observed over n independent trials. ``calibrate`` passes the number of held-out WINDOWS
    as n, not the number of points: the points inside one window share a forecast and a series, so they are not independent
    (audit item 16)."""
    if n <= 0:
        return [0.0, 1.0]
    from statistics import NormalDist
    zq = NormalDist().inv_cdf(0.5 + conf / 2)
    den = 1 + zq * zq / n
    c = (p + zq * zq / (2 * n)) / den
    h = zq * math.sqrt(p * (1 - p) / n + zq * zq / (4 * n * n)) / den
    return [round(max(0.0, c - h), 3), round(min(1.0, c + h), 3)]


def _coverage(actual: np.ndarray, q: np.ndarray) -> float:
    return float(((actual >= q[..., Q_LO]) & (actual <= q[..., Q_HI])).mean())


def _width(q: np.ndarray) -> float:
    return float((q[..., Q_HI] - q[..., Q_LO]).mean())


def _intermittent_result(x: np.ndarray, horizon: int, target: float) -> dict:
    """The delivered object for intermittent demand, reported as it is: TSB's expected volume and TSB's band.

    Motivation vs Logic
    Motivation: for lumpy demand the delivered band is TSB's. Calibrating it as if it were TimesFM's, around a near-zero
    median, put a TimesFM band back over the point (audit A6-02).
    Logic: no TimesFM forecast is made and nothing is refitted; the point and band are the ones the tool delivers, so the
    band brackets the point at every step and the recommendation is keep_raw.
    """
    point = expected_volume(x, horizon)
    band = tsb_band(x, horizon)
    return {
        "horizon": horizon, "target_coverage": target, "origins": 0, "fitted_on": 0, "tested_on": 0,
        "scales_by_horizon_third": [], "heldout_coverage_raw": None, "heldout_coverage_calibrated": None,
        "raw_gap": None, "calibrated_gap": None, "target": target, "scale_level": None, "second_pass": None,
        "width_ratio": 1.0, "method": "tsb_delivered_band", "ewma_volatility": None, "heldout_coverage_adaptive": None,
        "heldout_points": 0, "heldout_windows": 0, "heldout_raw_wilson90": None, "heldout_calibrated_wilson90": None,
        "selection": {"held_out_used_for_choice": False, "optimistic": False},
        "recommend": "keep_raw",
        "reason": ("intermittent demand: the delivered band is TSB's, not TimesFM's, and calibration of TimesFM's band is not "
                   "defined for it, so the band is kept as delivered"),
        "point": [round(float(v), 4) for v in point],
        "raw": {"q10": [round(float(v), 4) for v in band[:, 0]], "q90": [round(float(v), 4) for v in band[:, 8]]},
        "calibrated": {"q10": [round(float(v), 4) for v in band[:, 0]], "q90": [round(float(v), 4) for v in band[:, 8]],
                       "quantiles": {f"{lvl:.1f}": [round(float(v), 4) for v in band[:, i]]
                                     for i, lvl in enumerate([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])}},
        "caveats": ["Intermittent demand: the point is the expected volume (TSB) and the band is TSB's; neither is TimesFM's, "
                    "so no TimesFM calibration was run.",
                    "The band is widened, where needed, to contain the expected volume; its coverage is therefore not the nominal 80 percent."],
    }


def calibrate(values: Any, horizon: int, *, engine: Any, control: Any = None, n_origins: int = 12,
              target: float = TARGET) -> dict:
    """Calibrate the band of a TimesFM forecast of ``values``. Raises ValueError with the fix when the series is too short.
    For intermittent demand the delivered TSB object is returned unchanged (see _intermittent_result)."""
    x = np.asarray(values, dtype=float)
    if not np.isfinite(x).all():
        raise ValueError("series contains NaN or infinite values; fill or drop them first")
    if is_intermittent(x):
        return _intermittent_result(x, horizon, target)
    n = len(x)
    m, _ = backtest_season(x, horizon, n_origins, None)     # measured before the first window (audit A2-01)
    min_hist = max(MIN_HISTORY, 2 * m if m > 1 else MIN_HISTORY, 3 * horizon)
    k = min(int(n_origins), (n - min_hist) // horizon)
    if k < MIN_ORIGINS:
        raise ValueError(f"{n} points leave {max(k, 0)} past windows of {horizon}; calibration needs at least {MIN_ORIGINS}. "
                         "Use a shorter horizon, fewer origins, or keep the raw band and say it is uncalibrated.")
    origins = [n - horizon * (k - i) for i in range(k)]
    acts, qs, sigs = [], [], []
    for o in origins:
        hist = x[:o]
        sigs.append(ewma_sigma(hist))
        f = engine.forecast_controlled(hist, horizon, control) if control is not None and not control.is_default \
            else engine.forecast(hist, horizon)
        qs.append(np.asarray(f.q, dtype=float))
        acts.append(x[o:o + horizon])
    actual, q, sig = np.stack(acts), np.stack(qs), np.asarray(sigs)
    split = max(MIN_ORIGINS // 2, int(round(k * 2 / 3)))
    held_actual = actual[split:]
    fit_scale = fit_scales(actual[:split], q[:split], target)
    held_q = np.stack([apply_scales(q[i], fit_scale) for i in range(split, k)])
    raw_cov, cal_cov = _coverage(held_actual, q[split:]), _coverage(held_actual, held_q)
    raw_gap = abs(raw_cov - target)
    # Second pass: if the first-pass scales still leave the held-out coverage well under target, the windows the scales
    # were fitted on understated the spread. Refit on the same fit windows at a higher level, the target plus the shortfall
    # (capped at MAX_LEVEL), and keep that refit only if it lands closer to the target on the held-out windows.
    level = target
    second_pass = None
    if target - cal_cov > GAP_STEP:
        level2 = min(MAX_LEVEL, target + (target - cal_cov))
        scale2 = fit_scales(actual[:split], q[:split], level2)
        held2 = np.stack([apply_scales(q[i], scale2) for i in range(split, k)])
        cov2 = _coverage(held_actual, held2)
        adopted = abs(cov2 - target) < abs(cal_cov - target)
        second_pass = {"level": round(level2, 3), "scales_by_horizon_third": [round(s, 3) for s in scale2],
                       "heldout_coverage": round(cov2, 3), "adopted": bool(adopted)}
        if adopted:
            level, cal_cov = level2, cov2
    # The volatility-aware alternative: scores in units of the EWMA volatility at each origin, fitted on the same windows.
    vol = None
    if sig.min() > 0:
        vscale = fit_vol_scales(actual[:split], q[:split], sig[:split], target)
        held_v = np.stack([vol_band(q[i], float(sig[i]), vscale) for i in range(split, k)])
        vol_cov = _coverage(held_actual, held_v)
        vol = {"heldout_coverage": round(vol_cov, 3), "width_vs_raw": round(_width(held_v) / max(_width(q[split:]), 1e-12), 3)}
    method = "quantile_scaling"
    # Only as a repair of a band that is too narrow, and only when it is wider on average: a band is never narrowed.
    if (vol is not None and target - raw_cov > GAP_STEP and vol["width_vs_raw"] >= 1.0
            and abs(vol["heldout_coverage"] - target) + 1e-9 < abs(cal_cov - target)):
        method, cal_cov = "ewma_volatility", vol["heldout_coverage"]
    # The adaptive candidate: scales fitted on the fit windows alone (no held-out window moves the alpha path), then applied
    # as fixed scales to the held-out windows. It is chosen only when it lands at least GAP_STEP closer to target than the
    # method already chosen, and only when it widens the band on average, so it is never narrower than raw.
    adaptive_cov = None
    if split >= 2:
        aci_fit = aci_scales(_ratios(actual[:split], q[:split]), target)["scales"]
        held_a = np.stack([apply_scales(q[i], aci_fit) for i in range(split, k)])
        adaptive_cov = _coverage(held_actual, held_a)
        aci_width = _width(held_a) / max(_width(q[split:]), 1e-12)
        if aci_width >= 1.0 and abs(adaptive_cov - target) <= abs(cal_cov - target) - GAP_STEP + 1e-9:
            method, cal_cov = "adaptive_conformal", adaptive_cov
    cal_gap = abs(cal_cov - target)
    final = engine.forecast_controlled(x, horizon, control) if control is not None and not control.is_default \
        else engine.forecast(x, horizon)
    raw_q = np.asarray(final.q, dtype=float)
    if method == "ewma_volatility":
        scales = fit_vol_scales(actual, q, sig, target)
        cal_q = vol_band(raw_q, ewma_sigma(x), scales)
    elif method == "adaptive_conformal":
        scales = aci_scales(_ratios(actual, q), target)["scales"]
        cal_q = apply_scales(raw_q, scales)
    else:
        scales = fit_scales(actual, q, level)
        cal_q = apply_scales(raw_q, scales)
    width_ratio = _width(cal_q) / max(_width(raw_q), 1e-12)
    use = (use_calibrated(raw_gap, cal_gap, max(scales), width_ratio) if method == "quantile_scaling"
           else bool(raw_gap > GAP_STEP and cal_gap < raw_gap))
    n_held, n_win = int(held_actual.size), k - split
    caveats = ["Past windows share one series, so they are not independent; the coverage figures carry that uncertainty.",
               "Conformal coverage assumes future errors resemble the past ones. A regime change breaks it.",
               f"Check windows: {n_win} windows of {horizon} periods ({n_held} points). The Wilson intervals use the windows as "
               "the sample size, because the points inside a window are not independent.",
               "The method, the scale level and the use decision were chosen by comparing coverage on these same windows, so the "
               "held-out coverage figures are optimistic. They are a diagnostic, not an unbiased test of the chosen band."]
    if method == "ewma_volatility" and width_ratio > MAX_SCALE:
        caveats.append(f"TimesFM's own band was far too narrow here (held-out coverage {raw_cov:.0%}); the range shown follows the "
                       f"series' recent volatility and is {width_ratio:.1f} times wider.")
    if k - split < 3:
        caveats.append("Fewer than three held-out windows: the held-out coverage is a rough check only.")
    return {
        "horizon": horizon, "target_coverage": target, "origins": k, "fitted_on": split, "tested_on": k - split,
        "scales_by_horizon_third": [round(s, 3) for s in scales],
        "heldout_coverage_raw": round(raw_cov, 3), "heldout_coverage_calibrated": round(cal_cov, 3),
        "raw_gap": round(raw_gap, 3), "calibrated_gap": round(cal_gap, 3), "target": target,
        "scale_level": round(level, 3), "second_pass": second_pass,
        "width_ratio": round(width_ratio, 3),
        "method": method, "ewma_volatility": vol,
        "heldout_coverage_adaptive": None if adaptive_cov is None else round(adaptive_cov, 3),
        "heldout_points": n_held, "heldout_windows": n_win,
        "heldout_raw_wilson90": _wilson_share(raw_cov, n_win),
        "heldout_calibrated_wilson90": _wilson_share(cal_cov, n_win),
        # Audit item 15: the held-out windows chose the method and the level, so their coverage is reported as selected.
        "selection": {"held_out_used_for_choice": True, "optimistic": True},
        "recommend": "use_calibrated" if use else "keep_raw",
        "reason": ("the raw band is more than five points from target on held-out windows, and the calibrated band is closer" if use else
                   "the raw band is within five points of target, or calibration does not bring it closer on held-out windows"),
        "point": [round(float(v), 4) for v in raw_q[:, Q_MID]],
        "raw": {"q10": [round(float(v), 4) for v in raw_q[:, Q_LO]], "q90": [round(float(v), 4) for v in raw_q[:, Q_HI]]},
        "calibrated": {"q10": [round(float(v), 4) for v in cal_q[:, Q_LO]], "q90": [round(float(v), 4) for v in cal_q[:, Q_HI]],
                       "quantiles": {f"{lvl:.1f}": [round(float(v), 4) for v in cal_q[:, i]]
                                     for i, lvl in enumerate([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])}},
        "caveats": caveats,
    }
