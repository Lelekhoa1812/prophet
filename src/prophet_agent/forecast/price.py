"""Price series: detect one, compare forecasts with the random walk on the same origins, and report timing numbers.

Motivation vs Logic
Motivation: a price is the case where a forecast is most likely to look better than it is. Out of sample, the historical mean
and the random walk are hard to beat for asset prices (Goyal and Welch 2008, a source claim, not re-measured here), so a model
that is only compared with "repeat the last value" can look good on a rising stretch and still add nothing. Interval methods
must be judged on coverage and on the interval score, not on the centre line alone. Timing questions are answered from the
distribution the forecast implies, reported as numbers, not as a turning-point call.
Logic: prices are modelled as a random walk in log prices. Four baselines give quantile paths per step: a zero-drift random
walk with full-history volatility, the historical mean log return as drift, an EWMA volatility band (RiskMetrics lambda 0.94),
and a GJR-GARCH(1,1) volatility band fitted with scipy (normal innovations; falls back to EWMA when the fit fails, which is
reported). Every method is scored on the same rolling origins with MASE on prices, directional accuracy, the 80 percent
coverage and the Winkler (interval) score. A method "beats the random walk" only when its mean MASE is lower, a paired one-sided
sign-flip test reaches 5 percent, and it wins at least 60 percent of origins. Timing numbers come from a normal fitted to the nine
quantiles of each step in log price: the probability that the price is below today's price, and the step with the lowest median.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy.optimize import minimize
from scipy.signal import lfilter
from scipy.stats import norm

from prophet_agent.forecast.metrics import QS, paired_signflip

RW = "rw_zero_drift"
METHODS = ("rw_zero_drift", "hist_mean_drift", "ewma_rw", "garch_rw")
METHOD_LABELS = {
    "rw_zero_drift": "random walk, zero drift, full-history volatility",
    "hist_mean_drift": "random walk, historical mean log return as drift, full-history volatility",
    "ewma_rw": "random walk, EWMA volatility (RiskMetrics lambda 0.94), zero drift",
    "garch_rw": "random walk, GJR-GARCH(1,1) volatility with normal innovations, zero drift",
    "timesfm": "TimesFM 3 (the model in this backtest)",
}
LAMBDA = 0.94               # RiskMetrics decay for daily volatility
PERSISTENCE_MIN = 0.9       # lag-1 autocorrelation of the levels: a random walk is about 1, white noise about 0
MIN_POINTS = 60
ALPHA = 0.05                # one-sided level for "beats the random walk"
WIN_SHARE = 0.6             # share of origins a method must win to be said to beat the random walk
NOMINAL_COVERAGE = 0.8
WINKLER_ALPHA = 0.2         # the 80 percent central interval
Z9 = norm.ppf(QS)           # standard normal quantiles at the nine levels 0.1 .. 0.9 (index 4 is the median)


def detect_price_series(values: Any, dates: Any = None, *, force: bool | None = None) -> dict[str, Any]:
    """Whether a series looks like a price: positive, persistent, and daily or business-daily when dates are given.

    Motivation vs Logic
    Motivation: the price baselines only make sense for a positive level that wanders (a random walk). Applying them to a count
    or a rate would compare against the wrong null, so the classification is explicit and the caller can force it.
    Logic: all values must be finite and positive; the lag-1 autocorrelation of the levels must be at least 0.9; at least 60
    points are needed. With dates, the median gap must be one day (daily, or business days where the weekend gaps are 3 days
    and the median stays 1). Without dates the frequency is not checked and is reported as None. force=True or False overrides
    the decision and records that it was forced.
    """
    if force is not None:
        return {"is_price": bool(force), "forced": True, "reasons": ["set by the caller"]}
    x = np.asarray(values, dtype=float).ravel()
    finite = bool(np.isfinite(x).all()) if x.size else False
    positive = finite and bool((x > 0).all())
    reasons: list[str] = []
    if not positive:
        reasons.append("values are not all positive and finite")
    if len(x) < MIN_POINTS:
        reasons.append(f"{len(x)} points; at least {MIN_POINTS} are needed")
    lag1 = None
    if finite and len(x) >= 3 and np.std(x[:-1]) > 0 and np.std(x[1:]) > 0:
        lag1 = float(np.corrcoef(x[:-1], x[1:])[0, 1])
    persistent = lag1 is not None and lag1 >= PERSISTENCE_MIN
    if not persistent:
        reasons.append(f"lag-1 autocorrelation {None if lag1 is None else round(lag1, 3)} is below {PERSISTENCE_MIN}")
    daily = _daily_check(dates)
    if daily is False:
        reasons.append("the dates are not daily or business-daily")
    is_price = positive and len(x) >= MIN_POINTS and persistent and daily is not False
    return {"is_price": bool(is_price), "forced": False,
            "lag1_autocorrelation": None if lag1 is None else round(lag1, 3), "positive": positive,
            "daily_or_business_day": daily, "reasons": reasons or ["positive, persistent and daily-like"]}


def _daily_check(dates: Any) -> bool | None:
    """True when the median gap between dates is one day, False when it is not or the dates are not increasing, None when
    no usable dates were given."""
    if dates is None:
        return None
    try:
        d = np.array(list(dates), dtype="datetime64[D]")
    except (ValueError, TypeError):
        return None
    if len(d) < 3:
        return None
    gaps = np.diff(d).astype(int)
    if np.any(gaps <= 0):
        return False
    return bool(np.median(gaps) == 1)


def _ewma_next_var(r: np.ndarray) -> float:
    """One-step-ahead variance of log returns from an EWMA with lambda 0.94, started at the full-sample variance.

    Motivation vs Logic
    Motivation: a fixed full-history variance is slow to notice a calm or a volatile stretch; EWMA is the standard quick
    response, and its bands are what the 80 percent coverage is checked against.
    Logic: s2 = lambda s2 + (1 - lambda) r^2 over the returns in order. After the last return, s2 is the forecast for the next
    step. Longer horizons use the same variance per step (RiskMetrics).
    """
    if len(r) == 0:
        return 0.0
    s0 = float(np.var(r, ddof=1)) if len(r) > 1 else float(r[0] ** 2)
    # lfilter runs s2_t = lambda s2_{t-1} + (1 - lambda) r_t^2 with the start value s2_0 = s0 (IIR filter; same as the loop).
    s2 = lfilter([1.0 - LAMBDA], [1.0, -LAMBDA], np.asarray(r, dtype=float) ** 2, zi=[LAMBDA * s0])[0]
    return float(s2[-1])


def _gjr_sigma2(params: np.ndarray, y: np.ndarray, s0: float) -> np.ndarray:
    """Conditional variance path of a GJR-GARCH(1,1) on demeaned returns in percent, computed with an IIR filter (fast)."""
    omega, alpha, gamma, beta = params
    v = np.empty(len(y))
    v[0] = s0
    v[1:] = omega + (alpha + gamma * (y[:-1] < 0)) * y[:-1] ** 2
    return lfilter([1.0], [1.0, -beta], v)


def _gjr_nll(params: np.ndarray, y: np.ndarray, s0: float) -> float:
    s2 = _gjr_sigma2(params, y, s0)
    if not np.all(np.isfinite(s2)) or np.any(s2 <= 0):
        return 1e12
    return float(0.5 * np.sum(np.log(s2) + y * y / s2))


def _garch_var_path(r: np.ndarray, horizon: int) -> tuple[np.ndarray, dict[str, Any]]:
    """Per-step variance of a GJR-GARCH(1,1) forecast (in return units), and the fit note. Falls back to EWMA when the fit fails.

    Motivation vs Logic
    Motivation: GARCH is the usual volatility model for prices, and it lets a large move raise the band more than a small one
    (the leverage term gamma for falling prices). It is kept only as a baseline, not as a model of returns.
    Logic: the variance recursion is fitted by Gaussian likelihood with SLSQP, with positivity bounds and persistence kept
    below 0.999. Multi-step variance follows E[s2(t+k)] = omega + (alpha + gamma/2 + beta) E[s2(t+k-1)]. Innovations are
    treated as normal; the fit note says when EWMA was used instead.
    """
    y = (r - r.mean()) * 100.0
    s0 = float(np.var(y, ddof=1)) if len(y) > 1 else 0.0
    steps = np.arange(1, horizon + 1, dtype=float)
    fallback = {"fallback": "ewma", "reason": "GARCH fit did not converge or variance was flat"}
    if s0 <= 0:
        return steps * _ewma_next_var(r), fallback
    x0 = np.array([0.05 * s0, 0.05, 0.05, 0.9])
    bounds = [(1e-10 * s0, None), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0)]
    cons = [{"type": "ineq", "fun": lambda p: 0.999 - (p[1] + 0.5 * p[2] + p[3])}]
    try:
        res = minimize(_gjr_nll, x0, args=(y, s0), method="SLSQP", bounds=bounds, constraints=cons,
                       options={"maxiter": 200})
        p = np.asarray(res.x, dtype=float)
        ok = bool(res.success) and bool(np.all(np.isfinite(p)))
    except (ValueError, FloatingPointError, np.linalg.LinAlgError):
        ok = False
    if not ok:
        return steps * _ewma_next_var(r), fallback
    omega, alpha, gamma, beta = p
    s2 = _gjr_sigma2(p, y, s0)
    nxt = omega + (alpha + gamma * float(y[-1] < 0)) * float(y[-1]) ** 2 + beta * float(s2[-1])
    pers = alpha + 0.5 * gamma + beta
    path = np.empty(horizon)
    path[0] = nxt
    for k in range(1, horizon):
        path[k] = omega + pers * path[k - 1]
    info = {"fallback": None, "persistence": round(float(pers), 4), "alpha": round(float(alpha), 4),
            "gamma": round(float(gamma), 4), "beta": round(float(beta), 4)}
    return np.cumsum(path) / 1e4, info


def _band(hist: np.ndarray, horizon: int, method: str) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Cumulative log drift and cumulative log variance per step for one baseline, from the history only."""
    r = np.diff(np.log(hist))
    steps = np.arange(1, horizon + 1, dtype=float)
    if method == "rw_zero_drift":
        return np.zeros(horizon), steps * float(np.var(r, ddof=1)), {}
    if method == "hist_mean_drift":
        return steps * float(r.mean()), steps * float(np.var(r, ddof=1)), {}
    if method == "ewma_rw":
        return np.zeros(horizon), steps * _ewma_next_var(r), {}
    if method == "garch_rw":
        var_cum, info = _garch_var_path(r, horizon)
        return np.zeros(horizon), var_cum, info
    raise ValueError(f"unknown price method {method!r}; choose from {METHODS}")


def method_forecast(hist: Any, horizon: int, method: str) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Median (horizon,) and quantile paths (horizon, 9) of one price baseline, at the levels QS, plus a fit note.

    Motivation vs Logic
    Motivation: the rolling comparison and the timing numbers need the same quantile shape as the TimesFM bands, so that
    coverage and the interval score read the same way for every method.
    Logic: the log price at step h is log P_T + mu_h + z_q sqrt(var_h), with z_q the normal quantiles at QS. The median is
    exp(log P_T + mu_h). Prices are positive by construction.
    """
    h_arr = np.asarray(hist, dtype=float)
    if h_arr.ndim != 1 or len(h_arr) < 3:
        raise ValueError("at least 3 prices are needed for a price baseline")
    if not (np.isfinite(h_arr).all() and (h_arr > 0).all()):
        raise ValueError("price baselines need positive, finite prices")
    mu, var, info = _band(h_arr, int(horizon), method)
    logp0 = math.log(h_arr[-1])
    sd = np.sqrt(np.maximum(var, 0.0))
    median = np.exp(logp0 + mu)
    q = np.exp(logp0 + mu[:, None] + sd[:, None] * Z9[None, :])
    return median, q, info


def _winkler(a: np.ndarray, lo: np.ndarray, hi: np.ndarray, alpha: float = WINKLER_ALPHA) -> np.ndarray:
    """Interval score of a central interval [lo, hi] at level 1 - alpha (Winkler 1972; Gneiting and Raftery 2007), per step."""
    return (hi - lo) + (2.0 / alpha) * np.maximum(lo - a, 0.0) + (2.0 / alpha) * np.maximum(a - hi, 0.0)


def _scale(hist: np.ndarray, season: int) -> float:
    """The MASE denominator at one origin: mean absolute change at lag season (one step when there is no season)."""
    m = max(1, int(season))
    if len(hist) <= 1:
        return float("nan")
    lag = m if len(hist) > m else 1
    s = float(np.mean(np.abs(hist[lag:] - hist[:-lag])))
    return s if s > 1e-12 else float("nan")


def _nanmean(v: np.ndarray) -> float | None:
    return None if not np.isfinite(v).any() else round(float(np.nanmean(v)), 4)

def score_forecasts(values: Any, origins: list[int], horizon: int, season: int,
                    forecasts: dict[str, list[tuple[np.ndarray, np.ndarray]]]) -> dict[str, Any]:
    """Score (median, quantile paths) per origin for each method on the same origins.

    Motivation vs Logic
    Motivation: "beats the random walk" must mean the same test for every method, on the same windows, or the comparison is
    not a comparison.
    Logic: per origin and method: MASE of the median (mean absolute error over the horizon divided by the lag-season change
    of the history); directional accuracy (whether the median moves in the same direction as the actual, a zero median move
    counts half); coverage of the 10 to 90 percent band (nominal 0.8); and the Winkler score of the 80 percent interval, in
    price units and scaled by the MASE denominator. Each method's per-origin errors are then tested against the random
    walk's with a paired one-sided sign-flip test. Origins that cannot be scaled are left out for every method alike.
    """
    x = np.asarray(values, dtype=float)
    k = len(origins)
    scales = np.array([_scale(x[:o], season) for o in origins])
    per: dict[str, dict[str, np.ndarray]] = {}
    for name, pairs in forecasts.items():
        mase = np.full(k, np.nan)
        dire = np.full(k, np.nan)
        cov = np.full(k, np.nan)
        wink_s = np.full(k, np.nan)
        wink_p = np.full(k, np.nan)
        for j, o in enumerate(origins):
            s = scales[j]
            if not np.isfinite(s):
                continue
            med, q = pairs[j]
            a = x[o:o + horizon]
            p0 = x[o - 1]
            mase[j] = float(np.mean(np.abs(a - med))) / s
            sf, sa = np.sign(med - p0), np.sign(a - p0)
            dire[j] = float(np.mean(np.where(sf == 0, 0.5, (sf == sa).astype(float))))
            cov[j] = float(np.mean((a >= q[:, 0]) & (a <= q[:, 8])))
            w = _winkler(a, q[:, 0], q[:, 8])
            wink_p[j] = float(np.mean(w))
            wink_s[j] = float(np.mean(w)) / s
        per[name] = {"mase": mase, "dire": dire, "cov": cov, "wink_s": wink_s, "wink_p": wink_p}
    if RW not in per:
        raise ValueError("the random walk baseline must be scored to compare methods with it")
    rw_err = per[RW]["mase"]
    rw_mean = _nanmean(rw_err)
    out: dict[str, Any] = {}
    for name, v in per.items():
        keep = np.isfinite(v["mase"]) & np.isfinite(rw_err)
        n_valid = int(keep.sum())
        entry: dict[str, Any] = {
            "mase": _nanmean(v["mase"]), "directional_accuracy": _nanmean(v["dire"]),
            "coverage_80": _nanmean(v["cov"]), "nominal_coverage": NOMINAL_COVERAGE,
            "winkler_scaled": _nanmean(v["wink_s"]), "winkler_price": _nanmean(v["wink_p"]),
            "origins_scored": int(np.isfinite(v["mase"]).sum()),
        }
        if name == RW:
            entry.update({"lower_mase_than_random_walk": False, "beats_random_walk": False})
        else:
            st = paired_signflip(rw_err, v["mase"])
            wins = int((v["mase"][keep] < rw_err[keep]).sum())
            lower = entry["mase"] is not None and rw_mean is not None and entry["mase"] < rw_mean
            beats = bool(lower and n_valid > 0 and st["p_cand_better"] <= ALPHA and wins >= WIN_SHARE * n_valid)
            entry.update({"lower_mase_than_random_walk": lower, "beats_random_walk": beats,
                          "wins_vs_random_walk": wins, "origins_compared": n_valid,
                          "p_lower_error_than_random_walk": round(float(st["p_cand_better"]), 4)})
        out[name] = entry
    return out


def comparison_statement(methods: dict[str, Any], origins: int, horizon: int) -> dict[str, Any]:
    """Plain statements of the comparison: best method, whether any beat the random walk, and the coverage achieved.

    Motivation vs Logic
    Motivation: the answer must say when nothing beats the random walk, in words the planner cannot soften.
    Logic: the sentences follow from the flags in ``methods``. "No method beat the random walk" appears exactly when no method
    met all three conditions. Coverage is reported beside its nominal level.
    """
    beats = [m for m in methods if m != RW and methods[m]["beats_random_walk"]]
    lower_only = [m for m in methods if m != RW and methods[m]["lower_mase_than_random_walk"] and m not in beats]
    scored = {m: v["mase"] for m, v in methods.items() if v["mase"] is not None}
    best = min(scored, key=scored.get) if scored else None
    rw_mase = methods.get(RW, {}).get("mase")
    parts = []
    if beats:
        parts.append(f"{_join(beats)} beat the random walk on these {origins} origins of {horizon} steps: lower mean MASE, a "
                     f"paired one-sided p-value of at most {ALPHA}, and lower error on at least {WIN_SHARE:.0%} of origins.")
    else:
        parts.append(f"No method beat the random walk on these {origins} origins of {horizon} steps.")
    if lower_only:
        parts.append(f"{_join(lower_only)} had a lower mean MASE than the random walk, but not by a margin that holds up on "
                     "these origins.")
    if best is not None and rw_mase is not None:
        if best == RW:
            parts.append(f"The random walk has the lowest mean MASE ({rw_mase:.3f}).")
        else:
            parts.append(f"Lowest mean MASE: {best} at {scored[best]:.3f}; random walk at {rw_mase:.3f}.")
    cov = [f"{m} {v['coverage_80']:.2f}" for m, v in methods.items() if v["coverage_80"] is not None]
    if cov:
        parts.append("Share of actuals inside the 80 percent range (nominal 0.80): " + ", ".join(cov) + ".")
    return {"statement": " ".join(parts), "best_method": best, "beat_random_walk": beats,
            "lower_mase_only": lower_only, "no_method_beat_random_walk": not beats}


def _join(names: list[str]) -> str:
    return names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]


def _origins(n: int, horizon: int, k: int) -> list[int]:
    """The cut points used by rolling_backtest: k windows of the horizon, the last one ending at the final point."""
    return [n - horizon * (k - j) for j in range(k)]


def price_comparison(values: Any, horizon: int, origins_used: int, season: int = 1,
                     timesfm: tuple[np.ndarray, np.ndarray] | None = None) -> dict[str, Any]:
    """Compare the price baselines (and TimesFM, when given) with the random walk on the same origins.

    ``timesfm`` is (points (k, horizon), quantile paths (k, horizon, 9)) for the same origins, or None. ``origins_used`` must be
    the number of windows the rolling backtest used, so the origins match exactly.
    """
    x = np.asarray(values, dtype=float)
    if not (np.isfinite(x).all() and (x > 0).all()):
        raise ValueError("the price comparison needs positive, finite values; pass price false for this series")
    if origins_used < 2:
        raise ValueError("at least 2 origins are needed to compare methods")
    origins = _origins(len(x), horizon, origins_used)
    fcs: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {}
    notes: dict[str, list[dict[str, Any]]] = {}
    for name in METHODS:
        pairs, infos = [], []
        for o in origins:
            med, q, info = method_forecast(x[:o], horizon, name)
            pairs.append((med, q))
            infos.append(info)
        fcs[name] = pairs
        notes[name] = infos
    if timesfm is not None:
        pts, qs = timesfm
        fcs["timesfm"] = [(np.asarray(pts[j], dtype=float), np.asarray(qs[j], dtype=float)) for j in range(origins_used)]
    methods = score_forecasts(x, origins, horizon, season, fcs)
    garch_fallbacks = sum(1 for i in notes["garch_rw"] if i.get("fallback"))
    stmt = comparison_statement(methods, origins_used, horizon)
    return {"basis": "measured: rolling origins of this series, scored after the fact", "origins": origins_used,
            "horizon": horizon, "season_for_scale": max(1, int(season)), "random_walk": RW,
            "methods": methods, "method_labels": {m: METHOD_LABELS[m] for m in methods},
            "garch_ewma_fallback_origins": garch_fallbacks, **stmt,
            "note": ("Beating the random walk here is a statement about these origins only. The 80 percent coverage is "
                     "compared with the nominal 0.80; a band much narrower than that covers less than it claims.")}


def price_outlook(values: Any, horizon: int, methods: tuple[str, ...] = METHODS) -> dict[str, dict[str, Any]]:
    """Median and quantile paths of each baseline from the end of the series, as arrays a caller can use."""
    x = np.asarray(values, dtype=float)
    out = {}
    for m in methods:
        med, q, info = method_forecast(x, horizon, m)
        out[m] = {"median": med, "q": q, "fit": info}
    return out


def timing_numbers(q: Any, last_price: float, *, lowest_at: str = "median") -> dict[str, Any]:
    """Numbers for a timing question, read from the fitted distribution of each step.

    Motivation vs Logic
    Motivation: a planner asked about timing needs the probability that a future price is below today's, and where the median
    is lowest, without these being sold as a forecast of a turning point.
    Logic: at each step the nine quantiles of the log price are fitted by a line in the normal quantiles (intercept a, slope
    b). The probability that the price is below today's is Phi((log P0 - a) / b). The step with the lowest median is the
    smallest value of the median path. When the median does not move (zero drift) no step is named. Everything here is a
    property of the fitted distribution; the wording is left to the answer.
    """
    qa = np.asarray(q, dtype=float)
    if qa.ndim != 2 or qa.shape[1] != len(QS):
        raise ValueError("quantile paths must have shape (horizon, 9)")
    p0 = float(last_price)
    if not p0 > 0:
        raise ValueError("the last price must be positive")
    logq = np.log(qa)
    X = np.column_stack([np.ones(len(QS)), Z9])
    coef, *_ = np.linalg.lstsq(X, logq.T, rcond=None)
    a, b = coef[0], np.maximum(coef[1], 1e-12)
    p_below = norm.cdf((math.log(p0) - a) / b)
    med = qa[:, 4]
    flat = bool(np.ptp(med) <= 1e-12 * p0)
    step = None if flat else int(np.argmin(med)) + 1
    return {"last_price": round(p0, 6), "steps": int(qa.shape[0]),
            "prob_below_today_by_step": [round(float(v), 4) for v in p_below],
            "lowest_median_step": step,
            "lowest_median_price": None if step is None else round(float(med[step - 1]), 6),
            "prob_below_today_at_lowest_median_step": None if step is None else round(float(p_below[step - 1]), 4),
            "median_flat": flat,
            "basis": ("property of the fitted distribution, from the quantile paths of the method named by the caller; "
                      "not a prediction of a turning point")}


def price_timing(values: Any, horizon: int, methods: tuple[str, ...] = METHODS) -> dict[str, Any]:
    """Timing numbers for each baseline, from the end of the series."""
    x = np.asarray(values, dtype=float)
    res = {m: timing_numbers(v["q"], x[-1]) for m, v in price_outlook(x, horizon, methods).items()}
    return {"last_price": round(float(x[-1]), 6), "horizon": int(horizon), "methods": res}
