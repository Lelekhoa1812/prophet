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


def test_the_decision_follows_the_target_not_a_fixed_gain():
    """The old rule needed a 0.02 improvement, so a band that moved from 0.692 to 0.708 against 0.80 was kept. The rule now
    is: raw more than 0.05 from target AND calibrated closer. That case is accepted; a raw band already within 0.05 is not
    touched, however the calibrated one moves."""
    assert cal.use_calibrated(raw_gap=0.108, cal_gap=0.092, max_scale=1.2, width_ratio=1.2)
    assert not cal.use_calibrated(raw_gap=0.04, cal_gap=0.0, max_scale=1.2, width_ratio=1.2)
    assert not cal.use_calibrated(raw_gap=0.2, cal_gap=0.21, max_scale=1.2, width_ratio=1.2)
    assert not cal.use_calibrated(raw_gap=0.2, cal_gap=0.1, max_scale=1.0, width_ratio=1.0)     # nothing was widened
    assert not cal.use_calibrated(raw_gap=0.2, cal_gap=0.1, max_scale=2.0, width_ratio=cal.MAX_SCALE + 1)


def test_the_gaps_and_target_are_reported_and_agree_with_the_coverage_figures():
    out = cal.calibrate(_series(2.0), 8, engine=Engine(sd=1.0), n_origins=18)
    assert out["target"] == cal.TARGET
    assert out["raw_gap"] == pytest.approx(abs(out["heldout_coverage_raw"] - out["target"]), abs=1e-3)
    assert out["calibrated_gap"] == pytest.approx(abs(out["heldout_coverage_calibrated"] - out["target"]), abs=1e-3)


def test_a_second_pass_refits_at_a_higher_level_when_the_held_out_coverage_is_still_under_target():
    """A band of sd 1 on a random walk with steps of sd 3: the first pass leaves the held-out windows more than five points
    under target, so the second pass fits at a higher level (the target plus the shortfall) and keeps it because it lands
    closer. The band is never narrowed."""
    out = cal.calibrate(_series(3.0, seed=5), 8, engine=Engine(sd=1.0), n_origins=18)
    assert out["target"] - out["heldout_coverage_raw"] > 0.05
    sp = out["second_pass"]
    assert sp is not None and sp["adopted"] is True
    assert sp["level"] > out["target"] and out["scale_level"] == sp["level"]
    assert sp["heldout_coverage"] == out["heldout_coverage_calibrated"]
    assert out["calibrated_gap"] < out["raw_gap"]
    assert all(s >= 1.0 for s in sp["scales_by_horizon_third"]), "the second pass never narrows either"


def test_no_second_pass_when_the_first_pass_is_already_close():
    out = cal.calibrate(_series(0.5, seed=3), 4, engine=Engine(sd=3.0), n_origins=18)
    assert out["second_pass"] is None and out["scale_level"] == out["target"]
