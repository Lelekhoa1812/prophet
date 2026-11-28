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


WIS_LEVELS = ((0, 8, 0.2), (1, 7, 0.4), (2, 6, 0.6), (3, 5, 0.8))   # (lower, upper) quantile index and alpha of each central interval


def wis(actual: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Weighted interval score per observation (Bracher, Ray, Gneiting and Reich 2021, arXiv 2005.12881, PLOS Comput Biol
    17(2): e1008618): WIS = (0.5 |y - m| + sum_k (alpha_k / 2) IS_alpha_k) / (K + 0.5), with K = 4 central intervals.

    Motivation vs Logic
    Motivation: a truncated pinball average is not a standard proper score for a forecast given as quantiles, and the audit
    found no interval score anywhere (A2-09). The nine stored quantiles already contain four central intervals and the
    median, so WIS costs nothing extra and approximates CRPS.
    Logic: for each central interval [l, u] at level alpha, IS_alpha = (u - l) + (2 / alpha) (l - y) when y < l, plus
    (2 / alpha) (y - u) when y > u. The median term is 0.5 |y - m|. The score is proper, so the expected score is lowest for the
    true quantiles. Returned per observation: ``actual`` shape with the quantile axis last in ``q``.
    """
    y = np.asarray(actual, dtype=float)
    qa = np.asarray(q, dtype=float)
    total = 0.5 * np.abs(y - qa[..., 4])
    for lo, hi, alpha in WIS_LEVELS:
        lower, upper = qa[..., lo], qa[..., hi]
        is_alpha = ((upper - lower) + np.where(y < lower, (2.0 / alpha) * (lower - y), 0.0)
                    + np.where(y > upper, (2.0 / alpha) * (y - upper), 0.0))
        total = total + (alpha / 2.0) * is_alpha
    return total / (len(WIS_LEVELS) + 0.5)


NOMINAL_COVERAGE = 0.8      # the share a 10 to 90 percent band is built to contain (quantile indices 0 and 8 of QS)


def coverage(actual: np.ndarray, q: np.ndarray, lo: int = 0, hi: int = 8) -> float:
    """Share of actuals inside the band from quantile index lo to hi (default 10 to 90, nominal NOMINAL_COVERAGE).

    The figure is measured, not assumed: compare it with NOMINAL_COVERAGE, and do not read a band as 80 percent because
    of its quantile labels. A band built from a sd that is too small covers less (synthetic: a random walk's seasonal naive
    covered about 47 percent)."""
    a = np.asarray(actual, dtype=float)
    finite = np.isfinite(a)              # a missing actual is not a miss: it is left out of the share
    if not finite.any():
        return float("nan")
    inside = (a >= q[..., lo]) & (a <= q[..., hi])
    return float(inside[finite].mean())


def mase(actual: np.ndarray, pred: np.ndarray, hist_scale: np.ndarray) -> float:
    s = np.where(hist_scale > 0, hist_scale, np.nan)
    return float(np.nanmean(np.abs(actual - pred).mean(axis=1) / s))


SIGNFLIP_EXACT_MAX = 16     # up to this many pairs every sign pattern is enumerated (2^16 = 65,536 rows); above it, Monte Carlo
SIGNFLIP_DRAWS = 20000      # random sign patterns for the Monte Carlo p-value


def paired_signflip(base: np.ndarray, cand: np.ndarray, *, seed: int = 0, draws: int = SIGNFLIP_DRAWS) -> dict:
    """One-sided paired sign-flip (permutation) test of the mean error difference, with no distributional assumption.

    Motivation vs Logic
    Motivation: the percentile bootstrap of a mean of 6 to 12 paired differences rejects a true null about twice as often
    as its nominal level (audit A2-06: 0.089 at 6 origins against 0.05), so a verdict read from it is anti-conservative.
    Logic: d_i = base_i - cand_i. Under the null that the two forecasts have the same error law, each d_i is symmetric about
    zero, so flipping its sign leaves the law unchanged. The one-sided p-value is the share of sign patterns whose mean is at
    least the observed mean (Good, Permutation Tests, 2005, sign-flip form; Politis and Romano 1994 for the resampling
    view). Up to SIGNFLIP_EXACT_MAX pairs every pattern is enumerated, so the test is exact and its smallest one-sided p is
    1/2^n; above that a seeded Monte Carlo with the standard (1 + hits) / (draws + 1) correction is used.
    Returns p_cand_better (cand has the lower error), p_base_better, the observed mean difference (positive means cand is
    lower), the number of pairs and the method. NaN pairs are dropped first.
    """
    b = np.asarray(base, dtype=float).reshape(-1)
    c = np.asarray(cand, dtype=float).reshape(-1)
    if b.shape != c.shape:
        raise ValueError("the two error arrays must cover the same origins")
    keep = np.isfinite(b) & np.isfinite(c)
    d = (b - c)[keep]
    n = int(d.size)
    if n == 0:
        return {"p_cand_better": float("nan"), "p_base_better": float("nan"), "mean_diff": float("nan"), "n": 0,
                "method": None}
    obs = float(d.mean())
    tol = 1e-12 * max(1.0, float(np.abs(d).sum()))
    if n <= SIGNFLIP_EXACT_MAX:
        bits = (np.arange(2 ** n)[:, None] >> np.arange(n)[None, :]) & 1
        signs = 1.0 - 2.0 * bits
        means = (signs @ d) / n
        p_cand = float(np.mean(means >= obs - tol))
        p_base = float(np.mean(means <= obs + tol))
        method = "signflip_exact"
    else:
        rng = np.random.default_rng(seed)
        signs = rng.choice(np.array([-1.0, 1.0]), size=(draws, n))
        means = (signs @ d) / n
        p_cand = float((1 + np.sum(means >= obs - tol)) / (draws + 1))
        p_base = float((1 + np.sum(means <= obs + tol)) / (draws + 1))
        method = "signflip_montecarlo"
    return {"p_cand_better": p_cand, "p_base_better": p_base, "mean_diff": obs, "n": n, "method": method}


def diebold_mariano(base: np.ndarray, cand: np.ndarray, lag: int) -> dict:
    """One-sided Diebold-Mariano test, with the Harvey-Leybourne-Newbold small-sample correction, of base losses against cand.

    Motivation vs Logic
    Motivation (Mission 10 batch 2): the combination's gain over a comparator is computed on overlapping forecast windows
    (an origin every step, windows h steps long), so consecutive loss differences are correlated and the sign-flip test above,
    which assumes independent pairs, overstates the evidence.
    Logic: d_t = base_t - cand_t. Its long-run variance sums the autocovariances up to ``lag`` (h - 1 for windows of h steps
    one origin apart) with equal weights, as Diebold and Mariano (1995) and Harvey, Leybourne and Newbold (1997) set it; when
    that sum is not positive, Bartlett weights are used instead (Newey and West 1987). DM = mean(d) / sqrt(var / n); the HLN
    factor sqrt((n + 1 - 2h + h(h - 1) / n) / n) rescales it, and the p-value is read from Student t with n - 1 degrees of
    freedom. Measured on simulated equal-loss differences that are moving averages over h steps (2026-11-28): the test still
    rejects 7 to 13 per cent of the time at a nominal 5 per cent for n = 60 to 240 and h = 6 to 24, so it is reported beside
    the sign-flip test on non-overlapping windows (exact) and a gain counts as significant only when both agree.
    """
    from scipy import stats
    b = np.asarray(base, dtype=float).reshape(-1)
    c = np.asarray(cand, dtype=float).reshape(-1)
    if b.shape != c.shape:
        raise ValueError("the two loss arrays must cover the same origins")
    d = (b - c)[np.isfinite(b) & np.isfinite(c)]
    n = int(d.size)
    lag = max(0, min(int(lag), n - 2))
    out = {"p_cand_better": float("nan"), "stat": float("nan"), "mean_diff": float(d.mean()) if n else float("nan"),
           "n": n, "lag": lag, "method": "dm_hln"}
    if n < 8:
        return {**out, "note": "fewer than 8 pairs"}
    dc = d - d.mean()
    gam = [float(dc @ dc) / n] + [float(dc[k:] @ dc[:-k]) / n for k in range(1, lag + 1)]
    var = gam[0] + 2.0 * sum(gam[1:])
    if var <= 0:
        var = gam[0] + 2.0 * sum((1.0 - k / (lag + 1)) * g for k, g in enumerate(gam[1:], start=1))
        out["kernel"] = "bartlett"
    if var <= 0:
        return {**out, "note": "zero variance of the loss difference"}
    h = lag + 1
    stat = float(float(d.mean()) / np.sqrt(var / n) * np.sqrt(max((n + 1 - 2 * h + h * (h - 1) / n) / n, 1e-12)))
    return {**out, "p_cand_better": float(stats.t.sf(stat, df=n - 1)), "stat": stat}


def compare_losses(base: np.ndarray, cand: np.ndarray, h: int, step: int = 1) -> dict:
    """Both tests of base against cand on rolling-origin losses: Diebold-Mariano-HLN on every origin, and the exact sign-flip
    test on origins at least h steps apart (independent windows). significant is True only when both p-values are below
    0.05; with fewer than five independent windows the sign-flip test cannot reach 0.05 and nothing is called significant."""
    b = np.asarray(base, dtype=float).reshape(-1)
    c = np.asarray(cand, dtype=float).reshape(-1)
    lag = max(0, -(-int(h) // max(1, int(step))) - 1)
    dm = diebold_mariano(b, c, lag)
    every = max(1, -(-int(h) // max(1, int(step))))
    idx = np.arange(len(b) - 1, -1, -every)[::-1]            # the newest window and every h-th one before it
    sf = paired_signflip(b[idx], c[idx])
    sig = bool(np.isfinite(dm["p_cand_better"]) and dm["p_cand_better"] < 0.05 and sf["n"] >= 5
               and sf["p_cand_better"] < 0.05)
    return {"dm": {k: (round(v, 6) if isinstance(v, float) else v) for k, v in dm.items()},
            "independent": {"n": sf["n"], "p_cand_better": round(sf["p_cand_better"], 6) if sf["n"] else None,
                            "method": sf["method"]},
            "significant": sig}
