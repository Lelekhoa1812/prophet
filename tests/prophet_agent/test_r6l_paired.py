"""forecast/paired.py: the seeded paired bootstrap of a relative error gain.

The sign convention is the one the backtest relies on: a positive gain means the candidate has the lower error, and the
gain is relative to the reference's mean error.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from prophet_agent.forecast.paired import bootstrap_gain


def test_constant_improvement_is_the_relative_gain_with_a_degenerate_interval():
    # Every pair differs by 1 on a reference mean of 2, so the gain is exactly one half and every resample agrees.
    out = bootstrap_gain(np.array([2.0, 2.0, 2.0, 2.0]), np.array([1.0, 1.0, 1.0, 1.0]), seed=0, confidence=0.9, resamples=500)
    assert out["gain"] == pytest.approx(0.5)
    assert out["lo"] == pytest.approx(0.5)
    assert out["hi"] == pytest.approx(0.5)


def test_a_candidate_with_higher_error_has_a_negative_gain():
    out = bootstrap_gain(np.ones(4), np.full(4, 2.0), seed=0, confidence=0.9, resamples=200)
    assert out["gain"] == pytest.approx(-1.0)
    assert out["hi"] < 0


def test_pairs_with_a_nan_on_either_side_are_dropped_before_the_reference_mean_is_taken():
    base = np.array([2.0, np.nan, 2.0, 2.0])
    cand = np.array([1.0, 5.0, np.nan, 1.0])
    # Only the first and last pairs survive: differences of 1 on a reference mean of 2, so the gain is 0.5.
    out = bootstrap_gain(base, cand, seed=3, confidence=0.9, resamples=200)
    assert out["gain"] == pytest.approx(0.5)


def test_gain_is_relative_to_the_reference_mean_not_to_the_absolute_difference():
    out = bootstrap_gain(np.full(5, 4.0), np.full(5, 3.0), seed=0, confidence=0.9, resamples=100)
    assert out["gain"] == pytest.approx(0.25)


@pytest.mark.parametrize("base, cand", [
    (np.zeros(3), np.ones(3)),                 # reference mean is zero: the relative gain is undefined
    (np.full(3, -1.0), np.zeros(3)),           # negative reference mean: also undefined
    (np.array([np.nan]), np.array([1.0])),     # no usable pair at all
    (np.array([]), np.array([])),
])
def test_an_undefined_relative_gain_is_nan_on_all_three_fields(base, cand):
    out = bootstrap_gain(base, cand, seed=0, confidence=0.9, resamples=50)
    assert all(math.isnan(out[k]) for k in ("gain", "lo", "hi"))


def test_the_same_seed_gives_the_same_interval_and_the_interval_brackets_the_gain():
    rng = np.random.default_rng(11)
    base = 10.0 + rng.normal(0.0, 1.0, 50)
    cand = base - 0.5 + rng.normal(0.0, 1.0, 50)

    first = bootstrap_gain(base, cand, seed=7, confidence=0.9, resamples=2000)
    again = bootstrap_gain(base, cand, seed=7, confidence=0.9, resamples=2000)
    assert first == again
    assert first["lo"] <= first["gain"] <= first["hi"]


def test_a_higher_confidence_level_gives_a_wider_interval_on_the_same_resamples():
    rng = np.random.default_rng(5)
    base = 10.0 + rng.normal(0.0, 1.0, 40)
    cand = base - 0.3 + rng.normal(0.0, 1.0, 40)

    narrow = bootstrap_gain(base, cand, seed=2, confidence=0.5, resamples=2000)
    wide = bootstrap_gain(base, cand, seed=2, confidence=0.9, resamples=2000)
    assert (wide["hi"] - wide["lo"]) > (narrow["hi"] - narrow["lo"])
    assert wide["lo"] <= narrow["lo"] and narrow["hi"] <= wide["hi"]
