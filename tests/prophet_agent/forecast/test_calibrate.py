"""Interval calibration: a too-narrow band is widened and checked on held-out windows; a good band is left alone."""
import numpy as np
import pytest

from prophet_agent.forecast import calibrate as cal
from prophet_agent.forecast.types import Forecast

Z = np.array([-1.2816, -0.8416, -0.5244, -0.2533, 0.0, 0.2533, 0.5244, 0.8416, 1.2816])


class Engine:
    """Forecasts the last value with a normal band of the stated sd. The truth below has its own noise sd."""
    def __init__(self, sd):
        self.sd = sd

    def forecast(self, series, horizon):
        last = float(series[-1])
        q = last + self.sd * Z[None, :] * np.ones((horizon, 1))
        return Forecast(np.full(horizon, last, dtype=np.float32), q.astype(np.float32))


def _series(noise, n=400, seed=1):
    rng = np.random.default_rng(seed)
    return 100 + np.cumsum(rng.normal(0, noise, n))


def test_a_band_that_is_too_narrow_is_widened_and_the_held_out_coverage_improves():
    x = _series(2.0)
    out = cal.calibrate(x, 8, engine=Engine(sd=1.0), n_origins=18)
    assert out["heldout_coverage_raw"] < 0.7, "the stated 80 percent band covers far less"
    assert out["heldout_coverage_calibrated"] > out["heldout_coverage_raw"] + 0.1
    assert max(out["scales_by_horizon_third"]) > 1.5 and out["recommend"] == "use_calibrated"
    assert all(c >= r for c, r in zip(out["calibrated"]["q90"], out["raw"]["q90"])), "a band is only widened"
    assert out["point"] == [round(float(x[-1]), 4)] * 8, "the point forecast is never moved"


def test_a_band_that_already_covers_is_kept():
    x = _series(0.5, seed=3)
    # a random walk's h-step error grows with sqrt(h); a band sized for the longest horizon over-covers the short ones
    out = cal.calibrate(x, 4, engine=Engine(sd=3.0), n_origins=18)
    assert out["recommend"] == "keep_raw" and min(out["scales_by_horizon_third"]) == 1.0


def test_scales_never_shrink_and_are_capped():
    actual = np.zeros((10, 6))
    q = np.tile(Engine(1.0).forecast(np.array([0.0]), 6).q, (10, 1, 1))
    assert cal.fit_scales(actual + 0.01, q) == [1.0, 1.0, 1.0]
    far = cal.fit_scales(actual + 1000.0, q)
    assert far == [cal.MAX_SCALE] * 3


def test_too_short_a_series_is_refused_with_the_fix():
    with pytest.raises(ValueError, match="at least 6"):
        cal.calibrate(_series(1.0, n=40), 12, engine=Engine(1.0))
    with pytest.raises(ValueError, match="NaN"):
        cal.calibrate([1.0, float("nan")] * 50, 4, engine=Engine(1.0))


def test_apply_scales_keeps_nonnegative_series_nonnegative_and_the_median_fixed():
    q = np.array([[1.0, 2, 3, 4, 5, 6, 7, 8, 9]] * 6)
    out = cal.apply_scales(q, [3.0, 3.0, 3.0])
    assert (out >= 0).all() and (out[:, 4] == 5).all() and out[0, 8] == 17.0


def test_the_tool_validates_arguments_and_returns_a_compact_view():
    from prophet_agent.tools.calibrate_tool import _run, observe
    x = [float(v) for v in _series(2.0)]
    with pytest.raises(ValueError, match="unknown arguments"):
        _run({"values": x, "horizon": 8, "bogus": 1}, "k", lambda: Engine(1.0))
    with pytest.raises(ValueError, match="n_origins"):
        _run({"values": x, "horizon": 8, "n_origins": 3}, "k", lambda: Engine(1.0))
    res = _run({"values": x, "horizon": 8, "n_origins": 18}, "k", lambda: Engine(1.0))
    view = observe(res)
    assert view["recommend"] == "use_calibrated" and "q90" in view["calibrated"] and len(view["calibrated"]["q90"]) == 8
