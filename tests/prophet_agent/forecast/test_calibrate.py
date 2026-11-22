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


def test_volatility_clustering_is_repaired_by_the_ewma_band_and_coverage_has_a_wilson_interval():
    rng = np.random.default_rng(7)
    sd = np.where((np.arange(900) // 150) % 2 == 0, 0.3, 3.0)          # calm and turbulent regimes alternate
    x = 100 + np.cumsum(rng.normal(0, sd))
    out = cal.calibrate(x, 10, engine=Engine(sd=0.5), n_origins=18)    # the model's band ignores the regime
    assert out["heldout_coverage_raw"] < 0.75
    assert out["method"] == "ewma_volatility" and out["recommend"] == "use_calibrated"
    assert abs(out["heldout_coverage_calibrated"] - 0.8) < abs(out["heldout_coverage_raw"] - 0.8)
    lo, hi = out["heldout_calibrated_wilson90"]
    assert lo <= out["heldout_coverage_calibrated"] <= hi
    assert cal.wilson(9, 12) == cal.wilson(9, 12) and cal.wilson(0, 0) == [0.0, 1.0]


def test_ewma_sigma_uses_only_the_history_it_is_given():
    a = np.cumsum(np.ones(50))
    assert cal.ewma_sigma(a) == pytest.approx(1.0)
    assert cal.ewma_sigma(np.r_[a, 1000.0]) > cal.ewma_sigma(a)


def _split_conformal_scales(scores, target=0.8, buckets=3):
    """The plain split-conformal scale per horizon bucket, fitted on every window at once (no adaptation)."""
    out = []
    for idx in cal._buckets(scores.shape[1], buckets):
        cell = np.sort(scores[:, idx].ravel())
        k = min(cell.size, int(np.ceil((cell.size + 1) * target)))
        out.append(float(cell[k - 1]))
    return out


def test_aci_scale_follows_errors_that_double_and_ends_above_the_whole_period_scale():
    """Scores of |N(0,1)| for 60 windows, then the same scores doubled for 60 more. The split-conformal scale fitted on the
    whole period mixes the two regimes; the adaptive scale moves its alpha down as misses pile up after the jump, so its
    final scale sits above that whole-period scale."""
    rng = np.random.default_rng(11)
    scores = np.vstack([np.abs(rng.normal(0, 1.0, (60, 3))), 2 * np.abs(rng.normal(0, 1.0, (60, 3)))])
    aci = cal.aci_scales(scores)
    whole = _split_conformal_scales(scores)
    assert all(a > w for a, w in zip(aci["scales"], whole)), (aci["scales"], whole)
    assert min(aci["alpha_final"]) < 0.2, "misses after the jump must push the miscoverage level down"
    assert all(s >= 1.0 for s in aci["scales"]) and all(0.01 <= a <= 0.5 for a in aci["alpha_final"])


def test_aci_on_a_stationary_well_calibrated_stream_keeps_alpha_near_target_and_the_scale_near_split_conformal():
    """Half-normal scores over 400 windows, so the true 80 percent quantile is 1.2816. ACI's guarantee is on the long-run
    average miscoverage level, so the test checks the average of each bucket's alpha path (after a burn-in of 50 windows)
    within 0.03 of 0.2. The final alpha is not held to a tight band: with the fast rate 0.05 it moves by up to 0.04 a window,
    and on this stream the final alpha ranged 0.10 to 0.30 across buckets, which moves the final scale by up to about 25
    percent (observed on seeds 1 to 5 and 23, where the worst bucket was 25.1 percent off). The scale tolerance is therefore
    30 percent; the sampling error of the split-conformal scale itself is about 3 to 4 percent."""
    rng = np.random.default_rng(23)
    scores = np.abs(rng.normal(0, 1.0, (400, 3)))
    aci = cal.aci_scales(scores)
    whole = _split_conformal_scales(scores)
    for path in aci["alpha_path"]:
        assert np.mean(path[50:]) == pytest.approx(0.2, abs=0.03), aci["alpha_path"]
    for s, w in zip(aci["scales"], whole):
        assert abs(s - w) / w < 0.30, (aci["scales"], whole)


def test_calibrate_on_a_volatility_jump_reports_the_adaptive_coverage_and_never_narrows_the_band():
    """Calm first 70 percent, then a volatility jump: the model's band ignores the regime. Whatever method is chosen, the
    adaptive held-out coverage is reported, and the chosen band is never narrower than raw at any step."""
    rng = np.random.default_rng(13)
    n = 500
    sd = np.where(np.arange(n) < int(0.7 * n), 0.3, 3.0)
    x = 100 + np.cumsum(rng.normal(0, sd))
    out = cal.calibrate(x, 8, engine=Engine(sd=0.5), n_origins=18)
    assert out["heldout_coverage_adaptive"] is not None
    assert 0.0 <= out["heldout_coverage_adaptive"] <= 1.0
    assert out["method"] in {"quantile_scaling", "ewma_volatility", "adaptive_conformal"}
    if out["method"] == "adaptive_conformal":
        assert out["heldout_coverage_adaptive"] == out["heldout_coverage_calibrated"]
    raw_w = np.array(out["raw"]["q90"]) - np.array(out["raw"]["q10"])
    cal_w = np.array(out["calibrated"]["q90"]) - np.array(out["calibrated"]["q10"])
    assert (cal_w >= raw_w - 1e-9).all(), "a band is only widened, never narrowed below raw"
    assert out["point"] == [round(float(x[-1]), 4)] * 8


def test_the_adaptive_scales_are_fitted_on_fit_windows_only_so_held_out_values_cannot_move_them(monkeypatch):
    """Two series that agree up to the first held-out origin and differ after it. The adaptive scales are fitted from the
    fit windows; the first call to aci_scales inside calibrate receives exactly those windows, so its input must be the
    same in both runs, while the held-out coverage differs."""
    base = _series(2.0, seed=9)
    seen = []
    real = cal.aci_scales

    def spy(scores, *args, **kwargs):
        seen.append(np.asarray(scores).copy())
        return real(scores, *args, **kwargs)

    monkeypatch.setattr(cal, "aci_scales", spy)
    out_a = cal.calibrate(base, 8, engine=Engine(sd=1.0), n_origins=18)
    fit_a = seen[0]
    seen.clear()
    # The first held-out origin is the one at index fitted_on; every window before it is untouched by the change below.
    first_held_origin = len(base) - 8 * (out_a["origins"] - out_a["fitted_on"])
    changed = base.copy()
    changed[first_held_origin:] += np.linspace(-30.0, 30.0, len(base) - first_held_origin)
    out_b = cal.calibrate(changed, 8, engine=Engine(sd=1.0), n_origins=18)
    fit_b = seen[0]
    assert fit_a.shape == (out_a["fitted_on"], 8)
    assert np.array_equal(fit_a, fit_b), "held-out values reached the fit of the adaptive scales"
    assert out_a["heldout_coverage_adaptive"] != out_b["heldout_coverage_adaptive"], "the held-out windows did change"
