"""Baseline bands are floored at zero only when the history never went below zero."""
from __future__ import annotations

import numpy as np

from prophet_agent.forecast.baselines import moving_average, seasonal_naive


def test_a_non_negative_history_keeps_its_bands_at_or_above_zero():
    s = np.array([1.0, 0.0, 2.0, 0.5, 0.0, 1.0] * 4)
    _, q = moving_average(s, 12, window=6)
    assert (q >= 0).all()


def test_a_history_that_goes_negative_is_not_clipped_at_zero():
    s = np.array([-5.0, -2.0, -6.0, -4.0, -5.5, -3.0] * 4)     # e.g. a margin or a temperature
    p, q = moving_average(s, 6, window=6)
    assert q.min() < 0 and not (q == 0).any(), "the lower band runs below zero: it is not clipped to zero"
    assert np.allclose(p, s[-6:].mean())
    _, q2 = seasonal_naive(s, 7, season=6)
    assert (q2 < 0).any()
