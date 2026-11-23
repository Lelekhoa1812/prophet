"""Round 6k (2026-11-23): the delivered seasonal line uses the season the backtest scored.

Audit (report_tool.py:225-226): the delivered seasonal rule measured its season on the whole history, while the backtest
measures it on the history before its first test window. A weekly pattern in the last 56 points gave season 7 delivered
and season 1 scored, in 40 of 40 synthetic probes. All series here are synthetic.
"""
import numpy as np

from prophet_agent.forecast.rolling import DEFAULT_ORIGINS, backtest_season, pick_season
from prophet_agent.tools import report_tool as R


def _series(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = 240
    x = np.cumsum(rng.normal(0, 1, n))
    x[-56:] += np.tile([3, 1, -1, -2, 0, 2, -3], 8)     # a weekly pattern only in the last 56 points
    return x


def test_the_probe_series_disagree_between_whole_history_and_the_backtest():
    # Guards the test itself: if these ever agree, the test below stops proving anything.
    x = _series(0)
    assert pick_season(x, None) != backtest_season(x, 7, DEFAULT_ORIGINS, None)[0]


def test_the_delivered_seasonal_rule_uses_the_season_the_backtest_scored():
    x = _series(0)
    line, m = R._rule_line(list(x), 7, "seasonal_naive", None)
    assert m == backtest_season(x, 7, DEFAULT_ORIGINS, None)[0]
    assert len(line) == 7


def test_a_season_given_by_the_caller_is_used_as_given():
    x = _series(1)
    _, m = R._rule_line(list(x), 7, "mean_last_season", 4)
    assert m == 4
