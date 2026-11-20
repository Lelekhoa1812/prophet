"""Offline checks for prophet_agent.forecast.tsb (ported from the TSB class in src/lab/engine/models.py)."""
import numpy as np

from prophet_agent.forecast.tsb import TSB


def test_all_zero_history_forecasts_zero():
    fc = TSB().predict([np.zeros(60)], horizon=3)
    assert fc.point.shape == (1, 3)
    assert fc.q.shape == (1, 3, 9)
    assert np.all(fc.q == 0)


def test_lumpy_history_has_small_median_and_long_right_tail():
    # one large order every 6 steps: the chance of demand is about 1/6, so the median is zero
    s = np.zeros(120)
    s[::6] = 100.0
    fc = TSB().predict([s], horizon=4)
    assert fc.point.shape == (1, 4)
    assert np.all(fc.point == 0)
    assert fc.q[0, :, 8].max() > 0  # the 90th percentile sees the big order
    assert np.all(np.diff(fc.q, axis=-1) >= 0)
    assert np.all(fc.q >= 0)


def test_steady_history_forecasts_the_order_size():
    fc = TSB().predict([np.full(80, 10.0)], horizon=2)
    assert np.allclose(fc.q, 10.0)
    assert np.allclose(fc.point, 10.0)
