"""R6m, audit item 1: each baseline band must cover close to the 80 percent its 10 to 90 quantiles are labelled with.

Motivation: the moving-average band was built as sd * Z * sqrt(h) with sd from one-step errors. A level forecast of a
stationary or seasonal series does not accumulate error with the horizon, so that band over-covered: on a synthetic
period-4 series it covered about 98 percent at horizons 3 to 8, and on stationary noise about 96 percent (measured,
synthetic, 400 replications). The seasonal-year band had the same sqrt(h) growth and covered 77 to 100 percent across
horizons (measured, synthetic). The fix measures each horizon's own error spread from the rule's past h-step errors.

The seasonal-naive phase band is already close to nominal on a seasonal series (measured, synthetic: about 78 to 80
percent). It is tested here so that a change to the shared band code cannot move it away from the label. Its known
limit is a random walk, where a no-growth band under-covers; the backtest reports measured coverage so that is visible.
"""
from __future__ import annotations

import numpy as np

from prophet_agent.forecast.baselines import moving_average, seasonal_naive, seasonal_year

H = 8
REPS = 400


def _stationary(rng, n):
    s = 100 + rng.normal(0.0, 2.0, n + H)
    return s[:n], s[n:]


def _periodic(rng, n):
    pattern = np.tile(np.array([10.0, 14.0, 7.0, 12.0]), (n + H) // 4 + 1)
    s = pattern[: n + H] + rng.normal(0.0, 2.0, n + H)
    return s[:n], s[n:]


def _random_walk(rng, n):
    s = 100 + np.cumsum(rng.normal(0.0, 1.0, n + H))
    return s[:n], s[n:]


def _coverage_by_horizon(make_history, rule, n=48, seed=7):
    """Share of the H future values inside the 10 to 90 band, by horizon, over REPS replications (synthetic)."""
    rng = np.random.default_rng(seed)
    hits = np.zeros(H)
    widths = np.zeros(H)
    for _ in range(REPS):
        hist, fut = make_history(rng, n)
        _, q = rule(hist)
        hits += (fut >= q[:, 0]) & (fut <= q[:, 8])
        widths += q[:, 8] - q[:, 0]
    return hits / REPS, widths / REPS


def test_moving_average_band_covers_close_to_80_percent_on_stationary_noise():
    cov, _ = _coverage_by_horizon(_stationary, lambda s: moving_average(s, H, 4))
    assert abs(cov.mean() - 0.8) < 0.04, cov
    assert ((cov > 0.7) & (cov < 0.9)).all(), cov


def test_moving_average_band_does_not_widen_with_horizon_when_the_level_is_stationary():
    _, width = _coverage_by_horizon(_stationary, lambda s: moving_average(s, H, 4))
    assert np.allclose(width, width[0], rtol=0.15), width


def test_moving_average_band_covers_close_to_80_percent_on_a_period_4_series():
    cov, _ = _coverage_by_horizon(_periodic, lambda s: moving_average(s, H, 4))
    assert 0.72 < cov.mean() < 0.86, cov


def test_moving_average_band_widens_with_horizon_on_a_random_walk():
    cov, width = _coverage_by_horizon(_random_walk, lambda s: moving_average(s, H, 4))
    assert width[-1] > width[0] * 1.2, width          # a random walk's level error does accumulate
    assert 0.68 < cov.mean() < 0.92, cov               # small-sample under-coverage is measured, not hidden (synthetic)


def test_seasonal_naive_phase_band_stays_near_its_label_on_a_period_4_series():
    cov, _ = _coverage_by_horizon(_periodic, lambda s: seasonal_naive(s, H, season=4, k=4))
    assert 0.75 < cov.mean() < 0.85, cov


def test_seasonal_year_band_covers_close_to_80_percent_and_does_not_grow_within_the_season():
    rng = np.random.default_rng(1)
    hits = np.zeros(H)
    width = None
    for _ in range(REPS):
        hist, fut = _periodic(rng, 60)
        _, q = seasonal_year(hist, H, season=4)
        hits += (fut >= q[:, 0]) & (fut <= q[:, 8])
        width = q[:, 8] - q[:, 0]
    cov = hits / REPS
    assert 0.72 < cov.mean() < 0.88, cov
    # The forecast repeats the last season, so its error does not grow with the horizon.
    assert np.allclose(width, width[0], rtol=1e-9), width

