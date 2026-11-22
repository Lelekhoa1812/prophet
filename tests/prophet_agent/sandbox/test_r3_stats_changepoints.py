"""Round 3 regression (A2-05): changepoints() must not report a repeating seasonal step as a regime change.

Motivation: four years of daily values with a winter step (+0.8) every year and no regime change gave 5 changepoints
on white noise, and an AR(1) noise gave 5 as well. The search ran on the raw series. Logic: with a known period, remove
the phase median first (the same idea as lab.robust_flags), then search the residual, requiring segments at least one
period long. Without a period the function behaves as before.
"""
from __future__ import annotations

import numpy as np
import pytest

from prophet_agent.sandbox import prophet_stats as ps

PERIOD = 365
YEARS = 4


def _annual_step_series(seed: int, sd: float = 1.0) -> np.ndarray:
    """Four years of daily values: +0.8 for the first 90 days of every year, plus white noise. No regime change."""
    t = np.arange(PERIOD * YEARS)
    seasonal = 0.8 * ((t % PERIOD) < 90)
    return 10.0 + seasonal + np.random.default_rng(seed).normal(0.0, sd, t.size)


def test_changepoints_ignore_annual_step_when_period_is_given():
    # Regression: before the fix this returned five changes at 80, 320, 445, 685 and 810 (white noise, measured).
    for seed in range(5):
        changes, report = ps.changepoints(_annual_step_series(seed), period=PERIOD)
        assert changes == [], f"seed {seed}: seasonal step reported as {[c['index'] for c in changes]}"
        assert report["deseasonalised_by"] == PERIOD


def test_changepoints_still_finds_a_real_shift_on_seasonal_data():
    x = _annual_step_series(seed=7)
    shift_at = 1000  # a true level shift of 3 standard deviations, after the seasonal pattern
    x[shift_at:] += 3.0
    changes, report = ps.changepoints(x, period=PERIOD)
    assert len(changes) == 1, [c["index"] for c in changes]
    assert abs(changes[0]["index"] - shift_at) <= 15
    # Measured after the phase median is removed: the residual shift is close to the true 3.0.
    assert 2.5 < changes[0]["shift"] < 3.5
    assert report["n_changes"] == 1


def test_changepoints_without_period_is_unchanged_on_plain_shift():
    rng = np.random.default_rng(3)
    x = np.r_[rng.normal(0, 1, 120), rng.normal(3, 1, 120)]
    changes, report = ps.changepoints(x)
    assert len(changes) == 1 and abs(changes[0]["index"] - 120) <= 5
    assert "deseasonalised_by" not in report


def test_changepoints_period_must_leave_two_segments():
    # A segment must be at least one period long, so a period over half the data leaves nothing to search: say so.
    with pytest.raises(ValueError, match="needs at least 160"):
        ps.changepoints(np.arange(100.0), min_size=20, period=80)
