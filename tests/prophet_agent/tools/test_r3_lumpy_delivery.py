"""Round 3 follow-up (A6-01, A6-02): the forecast tool uses the one definition of intermittent demand in forecast/tsb.py and
delivers the TSB band the backtest scores. Before, the tool kept its own copy that called a series with negative values
(a price, a balance) intermittent, and delivered TSB's raw band, whose 90th quantile could sit below the point it shipped."""
import sys

import numpy as np

import prophet_agent.tools.timesfm_tool  # noqa: F401  (the package re-exports a function of the same name)
from prophet_agent.forecast import tsb

tool = sys.modules["prophet_agent.tools.timesfm_tool"]


class Engine:
    checkpoint = "test"
    global_context = 15360


def _call(series):
    h = 6
    point = np.zeros(h)
    q = np.zeros((h, 9))
    return tool._one(np.asarray(series, dtype=float), point, q, h, Engine())


def test_the_tool_shares_the_backtest_definition():
    assert tool.is_intermittent is tsb.is_intermittent


def test_a_series_below_zero_is_not_delivered_as_lumpy_demand():
    series = [0, 0, -3, 0, 0, 5, 0, 0, -2, 0, 0, 4] * 5
    out = _call(series)
    assert out["config"]["point_basis"] == "median"


def test_lumpy_delivery_band_contains_the_point():
    rng = np.random.default_rng(3)
    series = np.where(rng.random(200) < 0.05, rng.integers(1, 30, 200), 0)     # demand in about 1 period in 20
    out = _call(series)
    assert out["config"]["point_basis"].startswith("expected demand")
    point = np.asarray(out["point"])
    lo, hi = np.asarray(out["quantiles"]["0.1"]), np.asarray(out["quantiles"]["0.9"])
    assert (lo <= point + 1e-9).all() and (point <= hi + 1e-9).all()
