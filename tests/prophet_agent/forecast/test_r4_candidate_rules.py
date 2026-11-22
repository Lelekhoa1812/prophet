"""Round 4, A4-12: the backtest's candidate rules must match the demand class, and each rule must print its window.

Finding A4-12: retail rep4 picked tsb_expected (an intermittent-demand rule) as the best simple rule for a series with
ADI = 1.0. The rolling backtest still builds its candidate list with TSB for every series that is not lumpy. Measured
offline with a fake engine (synthetic): on a random walk of 160 points, best_baseline is tsb_expected (MASE 1.08 against
naive 1.35). The class filter lives in baselines.candidate_rules and rolling_backtest builds its rule set with it.

Intermittency uses tsb.is_intermittent (ADI above 1.32). The finding's "or fewer than three in four periods" is implied by
it: a series with sales in under 75 percent of periods has ADI above 1.333.
"""
from __future__ import annotations

import numpy as np
import pytest

from prophet_agent.forecast import rolling as R
from prophet_agent.forecast.types import Forecast

SIMPLE = ("naive", "seasonal_naive", "drift", "mean_last_season", "tsb_expected")


class NaiveEngine:
    def forecast_batch(self, contexts, horizon):
        point = np.stack([np.full(horizon, float(c[-1])) for c in contexts])
        return Forecast(point, np.repeat(point[..., None], 9, axis=-1))


def _random_walk(n=160, seed=2):
    return 100 + np.cumsum(np.random.default_rng(seed).standard_normal(n))


def _intermittent(n=120, seed=6):
    rng = np.random.default_rng(seed)
    s = np.zeros(n)
    s[rng.choice(n, size=n // 4, replace=False)] = rng.gamma(2.0, 5.0, n // 4)   # sales in 1 period in 4
    return s


def test_continuous_series_gets_no_tsb_candidate():
    from prophet_agent.forecast.baselines import candidate_rules
    kept = candidate_rules(SIMPLE, _random_walk())
    assert "tsb_expected" not in kept
    assert kept == ("naive", "seasonal_naive", "drift", "mean_last_season")


def test_intermittent_series_keeps_tsb_candidate():
    from prophet_agent.forecast.baselines import candidate_rules
    assert "tsb_expected" in candidate_rules(SIMPLE, _intermittent())


def test_a_series_that_goes_below_zero_is_never_intermittent():
    from prophet_agent.forecast.baselines import candidate_rules
    s = _intermittent()
    s[0] = -1.0
    assert "tsb_expected" not in candidate_rules(SIMPLE, s)


def test_backtest_best_baseline_is_not_tsb_on_a_continuous_series():
    out = R.rolling_backtest(_random_walk(), 4, n_origins=12, season=1, engine=NaiveEngine())
    assert out["best_baseline"] != "tsb_expected"


def test_rule_windows_names_each_rule_window():
    from prophet_agent.forecast.baselines import rule_windows
    w = rule_windows(season=7, n_points=200)
    assert w["naive"] == 1
    assert w["seasonal_naive"] == 7
    assert w["drift"] == 200
    assert w["mean_last_season"] == 7
    assert w["tsb_expected"] == 52
    assert rule_windows(season=1, n_points=200)["mean_last_season"] == R.MEAN_LAST_FLOOR


def test_baselines_window_floor_matches_rolling():
    from prophet_agent.forecast import baselines
    assert baselines.MEAN_LAST_FLOOR == R.MEAN_LAST_FLOOR


def test_mean_last_season_centre_equals_the_stated_window_mean():
    # Guard for the WTI case (A4-12): the centre must be the mean of the window the output states, with season 1.
    s = _random_walk()
    out = R.rolling_backtest(s, 4, n_origins=8, season=1, engine=NaiveEngine())
    stated = out["mean_last_season_window"]["points"]
    assert stated == out["models"]["mean_last_season"]["window_points"] == 4
    assert R._mean_last(s, 1, 1)[0] == pytest.approx(s[-stated:].mean())
