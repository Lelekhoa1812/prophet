"""R6m, audit item 2: the distribution's lower tail is floored at zero only for a series that never goes below zero.

quantile_points floored every lower tail at zero, so a series that does go negative (a balance, a margin, a temperature)
had its samples clipped and its lower tail cut off. The baselines already apply the rule "floor only a non-negative
history" (R3, backtest of bands); this module now follows the same rule. Demand, which is non-negative, is unchanged.
"""
from __future__ import annotations

import numpy as np

from prophet_agent.forecast import distribution as dist


def test_a_negative_quantile_function_keeps_its_lower_tail_below_zero():
    q = np.array([-9.0, -8.0, -7.0, -6.0, -5.0, -4.0, -3.0, -2.0, -1.0])
    us, vals = dist.quantile_points(q)
    assert vals[0] < q[0], vals[0]                       # tapered below the 10th percentile, not clipped at zero
    assert vals[0] == np.float64(q[0] - 0.6 * (q[4] - q[0]))


def test_a_non_negative_quantile_function_is_still_floored_at_zero():
    q = np.array([0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
    us, vals = dist.quantile_points(q)
    assert vals[0] == 0.0                                # the taper would reach -1.2; demand cannot be below zero


def test_samples_from_a_negative_series_can_fall_below_zero():
    d = dist.Draws.make(n=2000, periods=2, seed=5)
    qrows = np.tile(np.array([-9.0, -8.0, -7.0, -6.0, -5.0, -4.0, -3.0, -2.0, -1.0]), (2, 1))
    s = dist.period_samples(qrows, d)
    assert s.min() < -9.0                                # the lower tail reaches below the 10th percentile, not clipped at 0
