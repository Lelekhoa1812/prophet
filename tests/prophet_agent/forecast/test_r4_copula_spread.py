"""Round 4, A2-04: a single-rho copula understates the spread of a multi-period total of random-walk errors.

Finding A2-04: a random walk with 7 daily steps has a true sd of 11.83 for the 7-period total of its cumulative errors
(sqrt(140)). Draws.make with rho = 0.1 gives a total sd of 5.29 at rho = 0 and 6.60 at rho = 0.1 (measured, synthetic,
200,000 draws), so the spread is 0.45 to 0.56 of the truth. The fix is a correlation matrix between periods (Cholesky
factor), estimated from backtest errors when there are enough origins, else the random-walk structure
corr(h, h') = sqrt(min / max).
"""
from __future__ import annotations

import math

import numpy as np
from scipy.stats import norm

from prophet_agent.forecast import distribution as dist

H = 7
TRUE_TOTAL_SD = math.sqrt(140.0)   # Var(sum_h S_h) = sum_i (H + 1 - i)^2 for i = 1..H = 140


def _random_walk_paths(n: int, seed: int) -> np.ndarray:
    """Cumulative errors S_h of a random walk with unit steps, (n, H)."""
    steps = np.random.default_rng(seed).standard_normal((n, H))
    return np.cumsum(steps, axis=1)


def test_total_spread_matches_random_walk():
    # The implied correlation of the cumulative errors, measured by simulation (as the finding's fix describes).
    implied = np.corrcoef(_random_walk_paths(50_000, seed=7), rowvar=False)
    d = dist.Draws.make_corr(n=200_000, corr=implied, seed=11)
    # Per-period marginals: each cumulative error has sd sqrt(h); the copula only supplies the dependence.
    errors = norm.ppf(d.u) * np.sqrt(np.arange(1, H + 1))
    total_sd = errors.sum(axis=1).std()
    assert abs(total_sd / TRUE_TOTAL_SD - 1) < 0.05


def test_random_walk_fallback_correlation_is_sqrt_of_min_over_max():
    c = dist.random_walk_corr(H)
    assert c.shape == (H, H)
    assert np.allclose(np.diag(c), 1.0)
    assert math.isclose(c[0, H - 1], math.sqrt(1 / H))   # periods 1 and 7: 0.378
    assert math.isclose(c[3, 4], math.sqrt(4 / 5))       # periods 4 and 5: 0.894
    assert np.allclose(c, c.T)


def test_corr_from_errors_falls_back_to_random_walk_with_few_origins():
    rng = np.random.default_rng(3)
    corr, source = dist.corr_from_errors(rng.standard_normal((10, H)))
    assert source == "random_walk_fallback"
    assert np.allclose(corr, dist.random_walk_corr(H))


def test_corr_from_errors_estimates_correlation_with_enough_origins():
    rng = np.random.default_rng(5)
    z = rng.standard_normal((400, 2))
    errors = np.column_stack([z[:, 0], 0.8 * z[:, 0] + 0.6 * z[:, 1]])   # correlation 0.8 by construction
    corr, source = dist.corr_from_errors(errors, min_origins=20)
    assert source == "estimated_from_origins"
    assert abs(corr[0, 1] - 0.8) < 0.1


def test_make_corr_handles_an_estimate_that_is_not_positive_definite():
    bad = np.array([[1.0, 0.9, -0.9], [0.9, 1.0, 0.9], [-0.9, 0.9, 1.0]])   # no valid correlation has these entries
    d = dist.Draws.make_corr(n=5_000, corr=bad, seed=2)
    assert np.isfinite(d.u).all() and (d.u > 0).all() and (d.u < 1).all()
