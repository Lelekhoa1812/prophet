"""Offline checks for prophet_agent.forecast.metrics (ported from src/lab/engine/metrics.py)."""
import numpy as np
import pytest

from prophet_agent.forecast import metrics


def test_wape_and_bias_hand_values():
    actual = np.array([[10.0, 10.0]])
    assert metrics.wape(actual, np.array([[8.0, 12.0]])) == pytest.approx(0.2)
    assert metrics.bias(actual, np.array([[12.0, 12.0]])) == pytest.approx(0.2)
    assert metrics.bias(actual, np.array([[8.0, 8.0]])) == pytest.approx(-0.2)


def test_zero_demand_gives_nan():
    zero = np.zeros((1, 3))
    assert np.isnan(metrics.wape(zero, zero + 1))
    assert np.isnan(metrics.bias(zero, zero + 1))
    assert np.isnan(metrics.pinball(zero, np.ones((1, 3, 9))))


def test_pinball_is_zero_for_exact_quantiles_and_known_for_zero_forecast():
    actual = np.array([[10.0]])
    exact = np.full((1, 1, 9), 10.0)
    assert metrics.pinball(actual, exact) == pytest.approx(0.0)
    # all quantiles at 0: loss per level is QS * 10, mean over levels is 0.5 * 10, normalised by demand 10
    zero_q = np.zeros((1, 1, 9))
    assert metrics.pinball(actual, zero_q) == pytest.approx(0.5)


def test_coverage_counts_actuals_inside_10_to_90_band():
    q = np.broadcast_to(np.arange(1.0, 10.0), (3, 9))
    actual = np.array([4.0, 0.0, 10.0])
    assert metrics.coverage(actual, q) == pytest.approx(1 / 3)


def test_mase_skips_zero_scale_and_averages_the_rest():
    actual = np.zeros((2, 3))
    pred = np.array([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]])
    assert metrics.mase(actual, pred, np.array([2.0, 2.0])) == pytest.approx(0.75)
    assert metrics.mase(actual, pred, np.array([0.0, 2.0])) == pytest.approx(1.0)
