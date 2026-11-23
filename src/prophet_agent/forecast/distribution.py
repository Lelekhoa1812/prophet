"""Distribution toolkit: turn forecast quantiles into samples, so decisions can use expectations.

Ported from src/lab/engine/distribution.py. Domain words are removed. "Weeks" are now generic periods.

Motivation vs Logic
Motivation: decisions such as "will stock cover the next period" add demand over several periods. A median hides skew
(lumpy demand has a median below its mean), and scaling per-period quantiles by a sub-period share treats every
period's error as perfectly correlated, which overstates the spread of a multi-period sum. On the twin, the measured
correlation of forecast errors between periods ahead is about 0.09.
Logic: each period's forecast is a piecewise-linear quantile function with tapered tails. Periods are tied together
with a one-factor Gaussian copula (correlation rho, default 0.1). Draws are deterministic and shared by every series,
so results are reproducible and cheap. Sub-periods inside one period move together (same draw), scaled by their share.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

LEVELS = np.arange(1, 10) / 10
_ERF = np.vectorize(math.erf)


def _phi(x: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + _ERF(x / math.sqrt(2.0)))


def random_walk_corr(periods: int) -> np.ndarray:
    """Correlation between the cumulative errors of a random walk at horizons h and h': sqrt(min(h, h') / max(h, h')).

    Motivation vs Logic
    Motivation: a single rho treats the 1-step and 7-step errors of a price-like series as equally related, which they are
    not (audit A2-04: periods 1 and 7 correlate 0.378 for a random walk, periods 4 and 5 correlate 0.894).
    Logic: the cumulative error at horizon h is the sum of h steps, so Cov(h, h') = min(h, h') and the correlation follows.
    """
    h = np.arange(1, int(periods) + 1, dtype=float)
    lo = np.minimum.outer(h, h)
    hi = np.maximum.outer(h, h)
    return np.sqrt(lo / hi)


def corr_from_errors(errors: np.ndarray, min_origins: int = 20) -> tuple[np.ndarray, str]:
    """Correlation between periods estimated from standardised backtest errors, origins by periods.

    Motivation vs Logic
    Motivation: the correlation between horizons is a property of the series, so a constant taken from another series is
    wrong for price-like data (audit A2-04). Estimating it needs origins; with few origins the estimate is noise.
    Logic: rows with any non-finite error are dropped. With fewer than ``min_origins`` complete rows, or a constant column,
    the random-walk structure is returned and the source says so. Otherwise the sample correlation of the errors is returned,
    labelled as estimated.
    """
    e = np.asarray(errors, dtype=float)
    if e.ndim != 2:
        raise ValueError("errors must be origins by periods")
    periods = e.shape[1]
    e = e[np.isfinite(e).all(axis=1)]
    if e.shape[0] < min_origins or periods < 2:
        return random_walk_corr(periods), "random_walk_fallback"
    sd = e.std(axis=0, ddof=1)
    if not (sd > 0).all():
        return random_walk_corr(periods), "random_walk_fallback"
    corr = np.corrcoef((e - e.mean(axis=0)) / sd, rowvar=False)
    if not np.isfinite(corr).all():
        return random_walk_corr(periods), "random_walk_fallback"
    return corr, "estimated_from_origins"


def _cholesky_corr(corr: np.ndarray) -> np.ndarray:
    """Lower Cholesky factor of a correlation matrix. An estimate that is not positive definite is clipped to the nearest
    positive semi-definite matrix (eigenvalues floored at a small positive value) and rescaled to unit diagonal."""
    c = np.asarray(corr, dtype=float)
    c = (c + c.T) / 2.0
    w, v = np.linalg.eigh(c)
    c2 = (v * np.clip(w, 1e-10, None)) @ v.T
    s = 1.0 / np.sqrt(np.diag(c2))
    return np.linalg.cholesky(c2 * np.outer(s, s))


@dataclass(frozen=True)
class Draws:
    """Common random numbers: uniform draws (n, periods) with marginal U(0,1) and correlation rho between periods."""

    u: np.ndarray

    @classmethod
    def make(cls, n: int = 1000, periods: int = 8, rho: float = 0.1, seed: int = 11) -> "Draws":
        rng = np.random.default_rng(seed)
        z0 = rng.standard_normal(n)
        e = rng.standard_normal((n, periods))
        x = math.sqrt(rho) * z0[:, None] + math.sqrt(1.0 - rho) * e
        return cls(_phi(x))

    @classmethod
    def make_corr(cls, n: int, corr: np.ndarray, seed: int = 11) -> "Draws":
        """Uniform draws (n, periods) whose Gaussian copula has the given correlation matrix between periods.

        Motivation vs Logic
        Motivation: a single rho understates the spread of a multi-period total when errors accumulate (audit A2-04: a random
        walk's 7-period total has sd 11.83; the single-rho copula gave 5.29 at rho 0 and 6.60 at rho 0.1, synthetic).
        Logic: correlated standard normals are e @ L.T with L the Cholesky factor of ``corr``; the uniforms are their normal
        CDF. The marginals stay uniform, so the caller maps each period through its own quantile function and scales it.
        """
        c = np.asarray(corr, dtype=float)
        if c.ndim != 2 or c.shape[0] != c.shape[1]:
            raise ValueError("corr must be a square matrix")
        L = _cholesky_corr(c)
        e = np.random.default_rng(seed).standard_normal((n, c.shape[0]))
        return cls(_phi(e @ L.T))


_CACHE: dict[tuple, Draws] = {}


def draws(n: int = 1000, periods: int = 8, rho: float = 0.1, seed: int = 11) -> Draws:
    key = (n, periods, round(rho, 4), seed)
    if key not in _CACHE:
        _CACHE[key] = Draws.make(n, periods, rho, seed)
    return _CACHE[key]


def quantile_points(q: np.ndarray, lower: float = 0.6, upper: float = 0.6) -> tuple[np.ndarray, np.ndarray]:
    """Levels and values of a piecewise-linear quantile function from nine quantiles, with tapered tails."""
    q = np.sort(np.asarray(q, dtype=float))
    lo = q[0] - lower * (q[4] - q[0])
    # A demand quantile function is floored at zero; one that is negative throughout (a balance, a margin) is not clipped.
    # Clipping it made the tail interpolate from 0 down to the 10th percentile, so the lower tail was cut off (R6m probe).
    if q[0] >= 0:
        lo = max(0.0, lo)
    hi = q[8] + upper * (q[8] - q[4])
    us = np.concatenate([[0.0], LEVELS, [1.0]])
    vals = np.concatenate([[lo], q, [hi]])
    return us, vals


def period_samples(qrows: np.ndarray, d: Draws, lower: float = 0.6, upper: float = 0.6) -> np.ndarray:
    """Samples (n, P) of per-period demand from quantile rows (P, 9)."""
    n = d.u.shape[0]
    out = np.empty((n, len(qrows)))
    for w in range(len(qrows)):
        us, vals = quantile_points(qrows[w], lower, upper)
        out[:, w] = np.interp(d.u[:, w], us, vals)
    return out


def subperiod_samples(qrows: np.ndarray, period_index: np.ndarray, share: np.ndarray, d: Draws) -> np.ndarray:
    """Samples (n, S): each sub-period takes its period's draw times that sub-period's share.

    period_index[j] is the row of qrows for sub-period j. share[j] is the sub-period's share divided by the number of
    sub-periods in a period, so the shares of one full period sum to 1.
    """
    per = period_samples(qrows, d)
    return per[:, period_index] * share[None, :]


def expected_shortfall(total: np.ndarray, position: float) -> float:
    """E[max(demand - position, 0)]: expected units that cannot be served from position."""
    return float(np.maximum(total - position, 0.0).mean())


def expected_leftover(total: np.ndarray, quantity: float) -> float:
    """E[max(quantity - demand, 0)]: expected units of `quantity` still unused."""
    return float(np.maximum(quantity - total, 0.0).mean())


def prob_exceeds(total: np.ndarray, position: float) -> float:
    """P(demand > position): chance that position runs out."""
    return float((total > position).mean())
