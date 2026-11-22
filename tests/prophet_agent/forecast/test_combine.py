"""Inverse-MSE combination: weights come from the held-out errors given, per horizon third, and only from them."""
from __future__ import annotations

import numpy as np
import pytest

from prophet_agent.forecast.combine import combine


def test_the_model_with_half_the_error_variance_gets_twice_the_weight():
    a, b = np.ones(6), np.zeros(6)
    err_a = np.full((4, 6), 1.0)
    err_b = np.full((4, 6), 2.0 ** 0.5)          # mse 2 against 1
    out = combine(a, b, err_a, err_b)
    assert out["weights_a"] == pytest.approx([2 / 3] * 3, abs=1e-4)
    assert out["point"] == pytest.approx(np.full(6, 2 / 3), abs=1e-4)


def test_weights_are_per_horizon_third():
    a, b = np.ones(9), np.zeros(9)
    err_a = np.tile([1.0, 1.0, 1.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0], (3, 1))
    err_b = np.tile([4.0, 4.0, 4.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], (3, 1))
    out = combine(a, b, err_a, err_b)
    # first third: mse_a 1 against mse_b 16, so a gets 16/17; the last two thirds reverse it, so a gets 1/17
    assert out["weights_a"] == pytest.approx([16 / 17, 1 / 17, 1 / 17], abs=1e-4)
    assert np.allclose(out["point"][:3], 16 / 17) and np.allclose(out["point"][3:], 1 / 17)


def test_with_no_earlier_errors_the_weights_are_equal():
    out = combine(np.ones(5), np.zeros(5), None, None)
    assert out["weights_a"] == [0.5, 0.5, 0.5] and out["point"] == pytest.approx(np.full(5, 0.5))
    out = combine(np.ones(5), np.zeros(5), np.zeros((0, 5)), np.zeros((0, 5)))
    assert out["weights_b"] == [0.5, 0.5, 0.5]


def test_the_weights_ignore_anything_but_the_errors_passed_in():
    a, b = np.ones(4), np.zeros(4)
    err_a, err_b = np.full((3, 4), 1.0), np.full((3, 4), 3.0)
    first = combine(a, b, err_a, err_b)
    again = combine(np.full(4, 100.0), np.full(4, -50.0), err_a, err_b)    # different points, same errors
    assert first["weights_a"] == again["weights_a"]


def test_the_two_forecasts_must_cover_the_same_horizon():
    with pytest.raises(ValueError, match="same horizon"):
        combine(np.ones(4), np.ones(5), None, None)
