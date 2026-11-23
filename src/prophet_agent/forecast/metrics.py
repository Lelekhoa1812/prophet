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
