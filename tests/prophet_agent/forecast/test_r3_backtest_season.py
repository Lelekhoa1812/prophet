"""Round 3, A2-01: the backtest and calibration must choose the season from history before the first test window.

Finding A2-01 (measured): choose_season on the full series read the test windows. A 200-point random walk with no season
gives season 1; the same history with a 56-point weekly pattern appended gives season 7, and that season then sets the MASE
scale and the seasonal baseline for every window. The fix measures the season on x[:n - horizon * n_origins], the part of
the series before the earliest window any backtest could test.
"""
from __future__ import annotations

import numpy as np

from prophet_agent.forecast import rolling as R
from prophet_agent.forecast.types import Forecast

HORIZON = 4
N_ORIGINS = 12
TEST_POINTS = HORIZON * N_ORIGINS      # the whole tail is test windows, so the pattern lies only inside them


class NaiveEngine:
    """Stub engine: repeats the last value. It never loads a model and reads only the history it is given."""

    def forecast_batch(self, contexts, horizon):
        point = np.stack([np.full(horizon, float(c[-1])) for c in contexts])
        return Forecast(point, np.repeat(point[..., None], 9, axis=-1))


def _history(seed=3, n=200):
    return np.cumsum(np.random.default_rng(seed).standard_normal(n))


def _pattern():
    return np.tile([3.0, -2.0, 1.0, 0.0, -1.0, 2.0, -3.0], 8)[:TEST_POINTS]


def _tail(hist, values):
    return hist[-1] + np.cumsum(values)


def test_season_is_measured_before_the_first_test_window_not_on_it():
    hist = _history()
    rng = np.random.default_rng(11)
    with_pattern = np.concatenate([hist, _tail(hist, rng.standard_normal(TEST_POINTS) * 0.2 + _pattern())])
    without = np.concatenate([hist, _tail(hist, rng.standard_normal(TEST_POINTS) * 0.2)])
    for series in (with_pattern, without):
        out = R.rolling_backtest(series, HORIZON, n_origins=N_ORIGINS, engine=NaiveEngine())
        assert out["season_used"] == 1


def test_changing_values_inside_the_test_windows_cannot_change_the_season():
    hist = _history()
    base = np.concatenate([hist, _tail(hist, np.random.default_rng(5).standard_normal(TEST_POINTS))])
    changed = base.copy()
    changed[len(hist):] = np.tile([50.0, -50.0], TEST_POINTS // 2) + np.arange(TEST_POINTS) * 3.0
    a = R.rolling_backtest(base, HORIZON, n_origins=N_ORIGINS, engine=NaiveEngine())
    b = R.rolling_backtest(changed, HORIZON, n_origins=N_ORIGINS, engine=NaiveEngine())
    assert a["season_used"] == b["season_used"] == 1
    assert a["mase_scale"] == b["mase_scale"]


def test_a_season_given_by_the_caller_is_used_unchanged():
    out = R.rolling_backtest(_history(), HORIZON, n_origins=N_ORIGINS, season=7, engine=NaiveEngine())
    assert out["season_used"] == 7 and out["season_reason"] == "given by the caller"


def test_calibrate_and_backtest_share_one_season_rule():
    hist = _history()
    series = np.concatenate([hist, _tail(hist, _pattern() * 1.0)])
    season, reason = R.backtest_season(series, HORIZON, N_ORIGINS, None)
    assert season == 1 and "before any test window" in reason
