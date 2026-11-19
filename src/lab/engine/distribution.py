"""Distribution toolkit: turn forecast quantiles into samples, so decisions can use expectations.

Motivation vs Logic
Motivation: decisions such as 'will stock cover the next week' add demand over several days and weeks. Plugging in
a median hides skew (lumpy demand has a median below its mean), and scaling weekly quantiles by days/7 treats every
week's error as perfectly correlated, which overstates the spread of a multi-week sum. The measured correlation of
forecast errors between weeks ahead on the twin is about 0.09.
Logic: each weekly forecast is a piecewise-linear quantile function with tapered tails. Weeks are tied together with a
one-factor Gaussian copula (correlation rho, default 0.1). Draws are deterministic and shared by every series, so
results are reproducible and cheap. Days inside one week move together (same draw), scaled by the weekday pattern.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

LEVELS = np.arange(1, 10) / 10
_ERF = np.vectorize(math.erf)


def _phi(x: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + _ERF(x / math.sqrt(2.0)))


@dataclass(frozen=True)
class Draws:
    """Common random numbers: uniform draws (n, weeks) with marginal U(0,1) and correlation rho between weeks."""
    u: np.ndarray

    @classmethod
    def make(cls, n: int = 1000, weeks: int = 8, rho: float = 0.1, seed: int = 11) -> "Draws":
        rng = np.random.default_rng(seed)
        z0 = rng.standard_normal(n)
        e = rng.standard_normal((n, weeks))
        x = math.sqrt(rho) * z0[:, None] + math.sqrt(1.0 - rho) * e
        return cls(_phi(x))


_CACHE: dict[tuple, Draws] = {}


def draws(n: int = 1000, weeks: int = 8, rho: float = 0.1, seed: int = 11) -> Draws:
    key = (n, weeks, round(rho, 4), seed)
    if key not in _CACHE:
        _CACHE[key] = Draws.make(n, weeks, rho, seed)
    return _CACHE[key]


def quantile_points(q: np.ndarray, lower: float = 0.6, upper: float = 0.6) -> tuple[np.ndarray, np.ndarray]:
    """Levels and values of a piecewise-linear quantile function from nine quantiles, with tapered tails."""
    q = np.sort(np.asarray(q, dtype=float))
    lo = max(0.0, q[0] - lower * (q[4] - q[0]))
    hi = q[8] + upper * (q[8] - q[4])
    us = np.concatenate([[0.0], LEVELS, [1.0]])
    vals = np.concatenate([[lo], q, [hi]])
    return us, vals


def week_samples(qrows: np.ndarray, d: Draws, lower: float = 0.6, upper: float = 0.6) -> np.ndarray:
    """Samples (n, W) of weekly demand from quantile rows (W, 9)."""
    n = d.u.shape[0]
    out = np.empty((n, len(qrows)))
    for w in range(len(qrows)):
        us, vals = quantile_points(qrows[w], lower, upper)
        out[:, w] = np.interp(d.u[:, w], us, vals)
    return out


def daily_samples(qrows: np.ndarray, week_index: np.ndarray, day_share: np.ndarray, d: Draws) -> np.ndarray:
    """Samples (n, days): each day takes its week's draw times that day's share of the week.

    week_index[j] is the row of qrows for day j; day_share[j] is the weekday share divided by 7, so a full
    week's shares sum to 1.
    """
    wk = week_samples(qrows, d)
    return wk[:, week_index] * day_share[None, :]


def expected_shortfall(total: np.ndarray, position: float) -> float:
    """E[max(demand - position, 0)]: expected units that cannot be served from position."""
    return float(np.maximum(total - position, 0.0).mean())


def expected_leftover(total: np.ndarray, quantity: float) -> float:
    """E[max(quantity - demand, 0)]: expected units of `quantity` still unsold."""
    return float(np.maximum(quantity - total, 0.0).mean())


def prob_exceeds(total: np.ndarray, position: float) -> float:
    """P(demand > position): chance that position runs out."""
    return float((total > position).mean())
