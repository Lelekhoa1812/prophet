"""Round 3, A6-06: the rule named mean_last_season must say which window it averaged.

Finding A6-06 (code-read, measured on WTI rep4): _mean_last averages x[-max(m, 4):], so with no season (m = 1) the
"mean of the last season" is a 4-point mean. The reports and the answer call it the last season. The fix states the
window in the output: its length, the season it came from, and a note that says when no season was found.
"""
from __future__ import annotations

import numpy as np

from prophet_agent.forecast import rolling as R
from prophet_agent.forecast.types import Forecast


class NaiveEngine:
    def forecast_batch(self, contexts, horizon):
        point = np.stack([np.full(horizon, float(c[-1])) for c in contexts])
        return Forecast(point, np.repeat(point[..., None], 9, axis=-1))


def _series(n=160, seed=2):
    return 100 + np.cumsum(np.random.default_rng(seed).standard_normal(n))


def test_without_a_season_the_window_is_four_points_and_is_stated():
    out = R.rolling_backtest(_series(), 4, n_origins=8, season=1, engine=NaiveEngine())
    w = out["mean_last_season_window"]
    assert w["points"] == 4 and w["season"] == 1
    assert "no season" in w["note"] and "last season" not in w["note"]
    assert out["models"]["mean_last_season"]["window_points"] == 4


def test_with_a_season_the_window_is_the_season():
    x = _series(200)
    out = R.rolling_backtest(x, 4, n_origins=8, season=12, engine=NaiveEngine())
    assert out["mean_last_season_window"]["points"] == 12
    assert out["models"]["mean_last_season"]["window_points"] == 12


def test_the_rule_value_is_the_mean_of_the_stated_window():
    hist = np.arange(1.0, 21.0)
    assert np.allclose(R._mean_last(hist, 3, 1), hist[-4:].mean())
    assert np.allclose(R._mean_last(hist, 3, 12), hist[-12:].mean())
