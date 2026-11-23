"""Round 6c (2026-11-23): an overlapping lag is a usage error, not a "not enough history" verdict.

Live retail business user (f1cc71cb): change_test(recent=28, lag=7) on 738 points returned "not enough history" with a
point-count note. The planner repeated the call, and the run stalled. Every series here is synthetic (seeded).
"""
import numpy as np
import pytest

from prophet_agent.sandbox import prophet_stats as ps


def _series(n=738, seed=3):
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    return 100 + 10 * np.sin(2 * np.pi * t / 7) + rng.normal(0, 4, n)


def test_an_overlapping_lag_raises_and_names_the_fix():
    with pytest.raises(ValueError, match="must be at least recent"):
        ps.change_test(_series(), recent=28, lag=7)


def test_a_valid_lag_on_the_same_series_still_returns_a_verdict():
    verdict, rep = ps.change_test(_series(), recent=28, lag=28)
    assert verdict in ("rise", "decline", "within usual variation")


def test_a_short_series_still_says_not_enough_history():
    verdict, rep = ps.change_test(np.arange(30.0), recent=28)
    assert verdict == "not enough history" and rep["points"] == 30
