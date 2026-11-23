"""Baselines refuse a history with missing values instead of returning NaN.

Motivation vs Logic
Motivation (GOAL.md, known defect): the baselines propagated NaN. A moving average or a seasonal naive forecast with a NaN
in its window returned a NaN point and a NaN band, which then reached a score or a chart without a reason.
Logic: a baseline has no honest value for a history with a gap (a gap is not zero and not the last level), so each one
raises a ValueError that says how many values are missing and what to do. The rolling backtest already refuses such a
series at its entry; the same rule now holds for the baseline functions it calls. Synthetic series only.
"""
import numpy as np
import pytest

from prophet_agent.forecast import baselines
from prophet_agent.forecast.rolling import _drift, _mean_last, _snaive
from prophet_agent.forecast.tsb import expected_volume


def _series_with_gap():
    s = 50 + 4 * np.sin(2 * np.pi * np.arange(60) / 7) + np.random.default_rng(1).normal(0, 1, 60)
    s[10] = np.nan                       # early in the history, so the old code's windows near the end were still finite
    return s


@pytest.mark.parametrize("call", [
    lambda s: baselines.moving_average(s, 4, 7),
    lambda s: baselines.seasonal_naive(s, 4, 7),
    lambda s: baselines.seasonal_year(s, 4, 7),
    lambda s: _snaive(s, 4, 7),
    lambda s: _drift(s, 4),
    lambda s: _mean_last(s, 4, 7),
], ids=["moving_average", "seasonal_naive", "seasonal_year", "rolling_snaive", "rolling_drift", "rolling_mean_last"])
def test_a_baseline_refuses_a_history_with_a_missing_value_and_says_how_many(call):
    with pytest.raises(ValueError, match="1 missing or non-finite value"):
        call(_series_with_gap())


def test_a_missing_value_in_the_last_window_is_refused_not_returned_as_nan():
    s = _series_with_gap()
    s[10] = 1.0
    s[-1] = np.nan
    with pytest.raises(ValueError, match="missing"):
        baselines.moving_average(s, 3, 4)


def test_an_infinite_value_is_refused_like_a_missing_one():
    s = _series_with_gap()
    s[10] = np.inf
    with pytest.raises(ValueError, match="missing or non-finite"):
        baselines.seasonal_naive(s, 3, 7)


def test_a_complete_history_is_unchanged_by_the_check():
    s = 50 + np.sin(np.arange(60.0))
    point, q = baselines.moving_average(s, 4, 7)
    assert np.isfinite(point).all() and np.isfinite(q).all()
    point, q = baselines.seasonal_naive(s, 4, 7)
    assert np.isfinite(point).all() and np.isfinite(q).all()


def test_tsb_expected_volume_treats_a_missing_period_as_unknown_not_zero():
    # The TSB chance skips a missing period (it is not a period without demand), so the expected volume stays finite.
    s = np.array([0.0, 3.0, np.nan, 0.0, 4.0, 0.0, 2.0] * 8)
    out = expected_volume(s, 3)
    assert np.isfinite(out).all()
