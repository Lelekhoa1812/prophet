"""M9-1: backtest_series adds the price baselines on the same origins when the series is price-like.

The TimesFM engine is replaced by a stub module (no weights, no network). The stub's point is the last value and its band is
a fixed widening, so the TimesFM numbers here are a test fixture, not a model result. All series are synthetic.
"""
from __future__ import annotations

import sys
import types

import numpy as np
import pytest
from scipy.stats import norm

from prophet_agent.tools import data_tools
from prophet_agent.forecast.price import METHODS, RW

Z9 = norm.ppf(np.arange(1, 10) / 10)


class _Fc:
    def __init__(self, point, q):
        self.point = point
        self.q = q


class _StubEngine:
    def forecast_batch(self, histories, horizon):
        pts, qs = [], []
        for h in histories:
            last = float(h[-1])
            pts.append(np.full(horizon, last))
            qs.append(last * np.exp(0.02 * np.sqrt(np.arange(1, horizon + 1)))[:, None] * np.exp(0.02 * Z9)[None, :])
        return _Fc(np.stack(pts), np.stack(qs))


@pytest.fixture
def stub_timesfm(monkeypatch):
    mod = types.ModuleType("prophet_agent.forecast.timesfm")

    class TimesFMEngine:
        @staticmethod
        def get():
            return _StubEngine()

    mod.TimesFMEngine = TimesFMEngine
    monkeypatch.setitem(sys.modules, "prophet_agent.forecast.timesfm", mod)
    return mod


def _prices(seed, n, mu=0.0):
    rng = np.random.default_rng(seed)
    return list(np.round(100 * np.exp(np.cumsum(rng.normal(mu, 0.01, n))), 4))   # synthetic


def test_price_series_gets_the_baselines_on_the_same_origins(stub_timesfm):
    values = _prices(31, 300)
    out = data_tools._backtest({"values": values, "horizon": 10, "n_origins": 4}, "k")
    assert "price" in out
    price = out["price"]
    assert set(price["methods"]) == set(METHODS) | {"timesfm"}
    assert price["origins"] == out["origins"] == 4
    assert price["random_walk"] == RW
    assert isinstance(price["statement"], str) and price["statement"]
    assert "80 percent range" in price["statement"]              # the coverage achieved is stated beside its nominal level


def test_timesfm_is_scored_on_the_price_block_the_same_way_as_in_the_backtest(stub_timesfm):
    out = data_tools._backtest({"values": _prices(32, 300), "horizon": 10, "n_origins": 4}, "k")
    tf_price = out["price"]["methods"]["timesfm"]["mase"]
    tf_backtest = out["models"]["timesfm"]["mase"]
    assert tf_price == pytest.approx(tf_backtest, abs=1e-3)


def test_statement_says_when_no_method_beat_the_random_walk(stub_timesfm):
    out = data_tools._backtest({"values": _prices(33, 300), "horizon": 10, "n_origins": 4}, "k")
    price = out["price"]
    if price["no_method_beat_random_walk"]:
        assert "No method beat the random walk" in price["statement"]
    else:
        assert "beat the random walk" in price["statement"]


def test_a_non_price_series_is_backtested_as_before(stub_timesfm):
    rng = np.random.default_rng(34)
    values = list(np.round(np.sin(np.arange(300) / 5.0) * 10 + rng.normal(0, 1, 300), 4))   # synthetic: negative, mean-reverting
    out = data_tools._backtest({"values": values, "horizon": 10, "n_origins": 4}, "k")
    assert "price" not in out


def test_price_false_skips_the_baselines_on_a_price_series(stub_timesfm):
    out = data_tools._backtest({"values": _prices(35, 300), "horizon": 10, "n_origins": 4, "price": False}, "k")
    assert "price" not in out


def test_price_true_forces_the_baselines_and_refuses_non_positive_values(stub_timesfm):
    rng = np.random.default_rng(36)
    values = list(np.round(rng.normal(0, 1, 300), 4))                  # synthetic: not positive
    with pytest.raises(ValueError, match="positive"):
        data_tools._backtest({"values": values, "horizon": 10, "n_origins": 4, "price": True}, "k")


def test_price_argument_must_be_boolean(stub_timesfm):
    with pytest.raises(ValueError, match="price must be true or false"):
        data_tools._backtest({"values": _prices(37, 300), "horizon": 10, "n_origins": 4, "price": "yes"}, "k")
