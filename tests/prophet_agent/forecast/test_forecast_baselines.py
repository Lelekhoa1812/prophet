"""Offline checks for prophet_agent.forecast.baselines (ported from src/lab/engine/baselines.py)."""
import numpy as np

from prophet_agent.forecast import baselines


def test_moving_average_of_constant_series_is_flat_with_ordered_band():
    s = np.full(30, 7.0)
    point, q = baselines.moving_average(s, 4, window=4)
    assert point.shape == (4,)
    assert q.shape == (4, 9)
    assert np.allclose(point, 7.0)
    assert np.allclose(q, 7.0)  # zero residual sd, so the band collapses onto the level
    assert (q >= 0).all()


def test_moving_average_band_is_nonnegative_and_sorted_for_noisy_series():
    rng = np.random.default_rng(1)
    s = np.abs(rng.normal(2.0, 3.0, 40))
    point, q = baselines.moving_average(s, 6, window=4)
    assert (q >= 0).all()
    assert (np.diff(q, axis=-1) >= 0).all()
    assert np.allclose(q[:, 4], point)


def test_seasonal_naive_repeats_same_phase_values():
    s = np.tile([1.0, 5.0], 10)  # period 2, last value 5
    point, q = baselines.seasonal_naive(s, 2, season=2, k=4)
    assert np.allclose(point, [1.0, 5.0])
    assert q.shape == (2, 9)


def test_seasonal_year_flat_series_forecasts_the_flat_level():
    s = np.full(120, 10.0)
    point, q = baselines.seasonal_year(s, 4, season=52)
    assert np.allclose(point, 10.0)
    assert q.shape == (4, 9)


def test_seasonal_year_falls_back_to_moving_average_on_short_history():
    s = np.arange(5, dtype=float)
    p_year, q_year = baselines.seasonal_year(s, 3, season=52)
    p_ma, q_ma = baselines.moving_average(s, 3, 8)
    assert np.allclose(p_year, p_ma)
    assert np.allclose(q_year, q_ma)
