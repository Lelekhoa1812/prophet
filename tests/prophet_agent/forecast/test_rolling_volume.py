"""Volume bias and the TSB expectation in the rolling backtest. A median-type point forecast of lumpy demand is the defect:
MASE and WAPE alone rate a zero forecast well on intermittent data, so the backtest must also report what share of the
volume the point forecast covered."""
from __future__ import annotations

import numpy as np
import pytest

from prophet_agent.forecast.rolling import rolling_backtest
from prophet_agent.forecast.tsb import TSB, expected_volume
from prophet_agent.forecast.types import Forecast


class ConstantEngine:
    """Returns the same point for every window: zeros reproduces a median forecast of lumpy demand."""
    def __init__(self, level: float):
        self.level = level

    def forecast_batch(self, contexts, horizon):
        n = len(contexts)
        point = np.full((n, horizon), self.level)
        q = np.repeat(point[..., None], 9, axis=-1)
        return Forecast(point, q)


def _intermittent(n=400, seed=11):
    rng = np.random.default_rng(seed)
    occur = rng.random(n) < 0.3
    return np.where(occur, rng.poisson(6, n) + 1, 0).astype(float)


def test_a_median_of_zero_on_lumpy_demand_is_reported_as_understating_volume():
    # Updated for audit A6-01: on lumpy demand the delivered point is TSB's expected volume, so the zero median is reported
    # as its own model (timesfm_median) and the note is about it; "timesfm" is the delivered expected volume.
    x = _intermittent()
    out = rolling_backtest(x, 30, n_origins=8, season=1, engine=ConstantEngine(0.0))
    assert out["models"]["timesfm_median"]["volume_bias_pct"] == pytest.approx(-100.0)
    assert out["models"]["timesfm"]["volume_bias_pct"] > -50
    assert "understates volume" in out["timesfm_volume_note"]


def test_an_exact_forecast_has_no_volume_bias_and_no_note():
    x = np.full(200, 10.0)
    out = rolling_backtest(x, 10, n_origins=6, season=1, engine=ConstantEngine(10.0))
    assert out["models"]["timesfm"]["volume_bias_pct"] == pytest.approx(0.0)
    assert "timesfm_volume_note" not in out


def test_tsb_expected_is_a_baseline_with_its_own_volume_bias():
    x = _intermittent()
    out = rolling_backtest(x, 30, n_origins=8, season=1, engine=ConstantEngine(0.0))
    tsb = out["models"]["tsb_expected"]
    assert tsb["mase"] is not None and tsb["volume_bias_pct"] is not None
    # Over stationary lumpy demand the expectation should land near the actual volume, not near zero.
    assert abs(tsb["volume_bias_pct"]) < 25


def test_expected_volume_is_chance_of_demand_times_mean_size():
    x = np.array([0, 4, 0, 0, 6, 0, 8, 0, 0, 10], dtype=float)
    p = 0.5
    for v in x:
        p = p + 0.15 * ((1.0 if v > 0 else 0.0) - p)
    expect = p * np.mean([4, 6, 8, 10])
    assert np.allclose(expected_volume(x, 3), expect)


def test_expected_volume_of_a_series_that_always_sells_is_its_size():
    x = np.full(120, 5.0)
    assert np.allclose(expected_volume(x, 4), 5.0)


def test_the_tsb_median_is_zero_where_its_expectation_is_not():
    """Documents the reason the baseline exists: the TSB quantile median sits at zero for lumpy demand, the expectation does not."""
    x = np.zeros(200)
    x[::10] = 8.0                                    # one order of 8 every ten periods: occurrence about 0.1
    median = TSB(n=4000).predict([x], 3).point[0]
    assert np.all(median == 0.0)
    level = expected_volume(x, 3)[0]
    assert 0.0 < level < 8.0                          # positive where the median is zero; below one order of 8 (TSB weights recent zeros)
