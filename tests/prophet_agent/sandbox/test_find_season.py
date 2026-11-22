"""find_season: the period, not its multiples, and no shift of the time axis when values are missing.

Motivation vs Logic
Motivation: a clean period-7 series returned lag 168 (a multiple of 7), and the same series with 20 percent missing values
returned lag 6, because the NaN were dropped and the time axis moved.
Logic: missing values stay on the grid, and each lag uses only the pairs where both points are observed. The strongest
lag is the shortest candidate whose autocorrelation is within SEASON_TOLERANCE of the best peak.
"""
from __future__ import annotations

import numpy as np

from prophet_agent.sandbox import prophet_lib as pl


def _periodic(period: int, n: int, seed: int = 0, noise: float = 0.3) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    return np.sin(2 * np.pi * t / period) + noise * rng.normal(size=n)


def test_a_clean_period_seven_series_returns_seven_not_its_multiple():
    res = pl.find_season(_periodic(7, 400))
    assert res["strongest_lag"] == 7 and res["strength"] > 0.5
    assert res["best_peak_lag"] % 7 == 0           # the best peak is a multiple, and it is not reported


def test_twenty_percent_missing_values_keep_the_period_seven():
    for seed in range(3):
        rng = np.random.default_rng(100 + seed)
        x = _periodic(7, 400, seed=seed)
        x[rng.random(400) < 0.2] = np.nan
        assert pl.find_season(x)["strongest_lag"] == 7


def test_missing_values_do_not_shift_the_axis():
    x = _periodic(7, 400, seed=5)
    gapped = x.copy()
    gapped[[3, 50, 51, 200]] = np.nan                # a few holes; the others keep their place
    clean_acf = pl.find_season(x, candidates=(7,))["autocorrelation_by_lag"][7]
    gapped_acf = pl.find_season(gapped, candidates=(7,))["autocorrelation_by_lag"][7]
    assert abs(clean_acf - gapped_acf) < 0.1


def test_a_monthly_series_with_period_twelve_returns_twelve():
    res = pl.find_season(_periodic(12, 120, seed=2))
    assert res["strongest_lag"] == 12


def test_white_noise_has_no_season():
    res = pl.find_season(np.random.default_rng(9).normal(size=400))
    assert res["strongest_lag"] is None and res["strength"] is None


def test_a_short_series_says_so_rather_than_guessing():
    res = pl.find_season(np.arange(8, dtype=float))
    assert "note" in res and res["points"] == 8


def test_a_constant_series_has_no_season():
    assert "note" in pl.find_season(np.ones(60))
