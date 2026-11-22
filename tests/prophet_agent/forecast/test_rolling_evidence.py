"""Strength of evidence in the rolling backtest: wins per window, the paired interval, the verdict, the default of twelve
windows, and the blend with leakage checks. The engines are fakes. The oracle engine reads the future on purpose, so the
test controls whether TimesFM wins; no number here is a measurement of TimesFM."""
from __future__ import annotations

import numpy as np
import pytest

from prophet_agent.forecast import rolling as R
from prophet_agent.forecast.types import Forecast


class OracleEngine:
    """Predicts the true next window plus a fixed bias and noise (TimesFM 'wins' by construction when noise is small).
    With mode='naive' it repeats the last value, which makes it identical to the naive baseline."""

    def __init__(self, x, bias=0.0, noise=0.0, mode="oracle", seed=0):
        self.x, self.bias, self.noise, self.mode = np.asarray(x, dtype=float), bias, noise, mode
        self.rng = np.random.default_rng(seed)

    def forecast_batch(self, contexts, horizon):
        pts = []
        for ctx in contexts:
            if self.mode == "naive":
                p = np.full(horizon, float(ctx[-1]))
            else:
                o = len(ctx)
                p = self.x[o:o + horizon] + self.bias + self.rng.normal(0, self.noise, horizon)
            pts.append(p)
        point = np.stack(pts)
        return Forecast(point, np.repeat(point[..., None], 9, axis=-1))


def _walk(n=200, seed=7):
    return 100 + np.cumsum(np.random.default_rng(seed).normal(0, 1.0, n))


def _step(x):
    return float(np.mean(np.abs(np.diff(x))))


def test_timesfm_better_when_it_wins_every_window_by_a_margin():
    x = _walk()
    out = R.rolling_backtest(x, 8, season=1, engine=OracleEngine(x, noise=0.05 * _step(x)))
    assert out["verdict"] == "timesfm_better"
    assert out["wins_vs_best_baseline"] == out["n_origins"] == out["origins"] == 12
    lo, hi = out["timesfm_vs_best_baseline_interval"]["interval_pct"]
    assert lo > 0 and out["timesfm_vs_best_baseline_interval"]["confidence"] == R.CONFIDENCE


def test_baseline_better_when_the_interval_lies_entirely_below_zero():
    x = _walk()
    out = R.rolling_backtest(x, 8, season=1, engine=OracleEngine(x, bias=6.0 * _step(x), noise=0.05 * _step(x)))
    assert out["verdict"] == "baseline_better"
    assert out["timesfm_vs_best_baseline_interval"]["interval_pct"][1] < 0
    assert out["wins_vs_best_baseline"] == 0


def test_no_clear_difference_when_timesfm_repeats_the_naive_rule():
    x = _walk()
    out = R.rolling_backtest(x, 8, season=1, engine=OracleEngine(x, mode="naive"))
    assert out["verdict"] == "no_clear_difference"
    lo, hi = out["timesfm_vs_best_baseline_interval"]["interval_pct"]
    assert lo <= 0 <= hi, "a repeat of the naive rule cannot be shown to beat or lose to the best rule"


def test_the_bootstrap_is_seeded_so_the_same_series_gives_the_same_interval():
    x = _walk()
    a = R.rolling_backtest(x, 8, season=1, engine=OracleEngine(x, noise=0.5 * _step(x), seed=3))
    b = R.rolling_backtest(x, 8, season=1, engine=OracleEngine(x, noise=0.5 * _step(x), seed=3))
    assert a["timesfm_vs_best_baseline_interval"] == b["timesfm_vs_best_baseline_interval"]


def test_twelve_windows_by_default_when_the_series_is_long_enough_and_fewer_when_it_is_not():
    x = _walk(400)
    out = R.rolling_backtest(x, 8, season=1, engine=OracleEngine(x))
    assert out["origins"] == 12 and "origins_note" not in out
    short = _walk(60)
    out = R.rolling_backtest(short, 8, season=1, engine=OracleEngine(short))
    assert out["origins"] < 12 and out["n_origins"] == out["origins"]
    assert "only" in out["origins_note"] and "of the 12 requested" in out["origins_note"]


def test_the_mase_scale_is_named_and_follows_the_season_used():
    x = _walk()
    out = R.rolling_backtest(x, 8, season=1, engine=OracleEngine(x))
    assert out["mase_scale"] == "mean absolute one-step change"
    t = np.arange(260)
    seasonal = 50 + 10 * np.sin(2 * np.pi * t / 12) + np.random.default_rng(1).normal(0, 1, 260)
    out = R.rolling_backtest(seasonal, 6, season=12, engine=OracleEngine(seasonal))
    assert out["mase_scale"] == "mean absolute seasonal change at lag 12" and out["season_used"] == 12


def test_the_blend_is_scored_only_when_asked_and_its_baseline_is_not_the_best_rule():
    x = _walk()
    assert "timesfm_combined" not in R.rolling_backtest(x, 8, season=1, engine=OracleEngine(x))["models"]
    out = R.rolling_backtest(x, 8, season=1, engine=OracleEngine(x, noise=0.5 * _step(x)), combine_forecasts=True)
    assert "timesfm_combined" in out["models"]
    assert out["best_baseline"] != "timesfm_combined" and out["models"]["timesfm_combined"]["mase"] is not None
    assert len(out["combination"]["weight_timesfm_by_origin"]) == out["origins"]


def _trend_engine(x):
    """Forecasts from the history only: the mean of the last four points, so it cannot read the future."""
    class Hist:
        def forecast_batch(self, contexts, horizon):
            point = np.stack([np.full(horizon, float(np.mean(c[-4:]))) for c in contexts])
            return Forecast(point, np.repeat(point[..., None], 9, axis=-1))
    return Hist()


def test_blend_weights_at_each_origin_use_only_earlier_origins():
    """Poisoning the actuals of one window must leave the weights of that origin and of every earlier origin unchanged.
    Only later origins, whose weights are fitted on the poisoned window, may move."""
    n, h = 200, 8
    x = _walk(n, seed=9)
    clean = R.rolling_backtest(x, h, season=1, engine=_trend_engine(x), combine_forecasts=True)
    poisoned_series = x.copy()
    origin_3 = n - h * (12 - 3)
    poisoned_series[origin_3:origin_3 + h] += 500.0
    dirty = R.rolling_backtest(poisoned_series, h, season=1, engine=_trend_engine(poisoned_series), combine_forecasts=True)
    wc, wd = clean["combination"]["weight_timesfm_by_origin"], dirty["combination"]["weight_timesfm_by_origin"]
    assert wc[:4] == wd[:4], "weights at origins 1 to 4 must not see the window that was poisoned at origin 4"
    assert clean["combination"]["baseline_by_origin"][:4] == dirty["combination"]["baseline_by_origin"][:4]
    assert wc != wd, "the poisoned window should move the later weights, otherwise the test proves nothing"
    assert clean["combination"]["weight_timesfm_by_origin"][0] == [0.5, 0.5, 0.5]
