"""The health check and the backtest name the same season (GOAL.md, known defect: one source of truth).

Motivation vs Logic
Motivation: the health check took its period from dominant_period while the backtest took its season from choose_season. On
the Berlin daily temperature the health check said 356 steps and the backtest 365, so a flat forecast could be judged against
one cycle and scored against another. On an annual synthetic series dominant_period gave 5 or 347 depending on the seed.
Logic: assess and season_check read the period from choose_season, the same function the backtest uses on the history before
its test windows. dominant_period stays in the sandbox library for its own contract. Synthetic series only.
"""
import numpy as np
import pytest

from prophet_agent.forecast.health import assess, season_check
from prophet_agent.forecast.rolling import backtest_season


def _annual_daily(n=1612, amplitude=10.0, noise=3.0, seed=3):
    t = np.arange(n)
    return 50 + amplitude * np.sin(2 * np.pi * t / 365) + np.random.default_rng(seed).normal(0, noise, n)


@pytest.mark.parametrize("seed", [3, 5])
def test_the_health_check_and_the_backtest_name_the_same_annual_season(seed):
    x = _annual_daily(seed=seed)
    backtest, _ = backtest_season(x, horizon=14, n_origins=12, given=None)
    point = np.full(14, x[-1])
    out = assess(x, point, context_used=len(x))
    assert backtest == 365
    assert out["period"] == backtest


@pytest.mark.parametrize("seed", [3, 5])
def test_a_given_season_is_checked_against_the_same_annual_cycle(seed):
    x = _annual_daily(seed=seed)
    assert season_check(365, x) is None
    out = season_check(5, x)
    assert out is not None and out["period_measured"] == 365
    assert "365 steps" in out["warning"]

