"""R6m, audit item 2: numerical edge cases in the forecast maths (all inputs synthetic).

Each test reproduces one defect found by a small probe before the fix. The probe for coverage gave 0.5 for a band that
contained its only finite actual: a missing actual was counted as a miss.
"""
from __future__ import annotations

import numpy as np
import pytest

from prophet_agent.forecast import baselines, metrics, tsb


def test_coverage_leaves_out_missing_actuals_instead_of_counting_them_as_misses():
    actual = np.array([[np.nan, 5.0]])
    q = np.array([[[0.0, 1, 2, 3, 4, 5, 6, 7, 8], [0.0, 1, 2, 3, 4, 5, 6, 7, 8]]])
    assert metrics.coverage(actual, q) == 1.0                       # the one finite actual is inside 0 to 8
    assert np.isnan(metrics.coverage(np.array([[np.nan]]), q[:, :1]))  # nothing to measure


# ------------------------------------------------------------------------------------------ TSB: missing periods --

LUMPY = np.array([0.0, 3.0, 0.0, 0.0, 5.0, 2.0, 0.0, 4.0, 0.0, 0.0, 6.0, 1.0, 0.0, 0.0, 0.0, 3.0] * 3)


def test_tsb_volume_ignores_a_missing_period_instead_of_reading_it_as_no_sale():
    with_gap = LUMPY.copy()
    with_gap[5] = np.nan
    without_gap = np.delete(LUMPY, 5)
    assert np.allclose(tsb.expected_volume(with_gap, 3), tsb.expected_volume(without_gap, 3))


def test_tsb_sampler_ignores_a_missing_period_too():
    # A frequent-demand series, so the sampled quantiles depend on the chance of demand and would move if the gap read as zero.
    dense = np.tile(np.array([2.0, 0.0, 4.0, 3.0]), 12)
    with_gap = dense.copy()
    with_gap[5] = np.nan
    without_gap = np.delete(dense, 5)
    q_gap = tsb.TSB().predict([with_gap], 3).q
    q_clean = tsb.TSB().predict([without_gap], 3).q
    assert np.allclose(q_gap, q_clean)


def test_intermittence_is_judged_on_the_observed_periods_only():
    with_gap = np.concatenate([LUMPY, [np.nan] * 10])
    assert tsb.is_intermittent(with_gap) == tsb.is_intermittent(LUMPY)


# ---------------------------------------------------------------------------------------- baselines: empty history --

def test_moving_average_refuses_an_empty_history_instead_of_forecasting_zero():
    with pytest.raises(ValueError):
        baselines.moving_average(np.array([]), 3, 4)
    with pytest.raises(ValueError):
        baselines.moving_average(np.array([1.0, 2.0]), 3, 0)
