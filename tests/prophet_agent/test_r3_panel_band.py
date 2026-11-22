"""A2-02: the global GBM 10 to 90 band must be calibrated on earlier windows only (split conformal), and the uncalibrated
band is reported alongside it. Synthetic panels with heavy-tailed noise give a known target, so coverage is measured."""
import numpy as np
import pytest

from prophet_agent.forecast import global_gbm as gg

H = 7
SEASON = 7


def heavy_panel(seed: int, entities: int = 20, T: int = 200):
    """Each entity: its own level times a weekly cycle and a slow trend, with Student-t (3 df) multiplicative noise."""
    rng = np.random.default_rng(seed)
    t = np.arange(T + H)
    out = {}
    for i in range(entities):
        level = rng.uniform(5, 50)
        shape = 1 + 0.2 * np.sin(2 * np.pi * t / SEASON) + 0.002 * t
        out[f"e{i}"] = level * shape * np.exp(rng.standard_t(3, T + H) * 0.08)
    return out


def held_out_coverage(bands_for, seeds=(0, 1, 2, 3)) -> float:
    """Share of held-out actual values inside the band, pooled over entities and seeds. The band comes from history only."""
    hits, total = 0, 0
    for seed in seeds:
        full = heavy_panel(seed)
        T = len(next(iter(full.values()))) - H
        hist = {k: v[:T] for k, v in full.items()}
        bands = bands_for(hist)
        for k, v in full.items():
            y = v[T:T + H]
            hits += int(((y >= np.asarray(bands[k]["q10"])) & (y <= np.asarray(bands[k]["q90"]))).sum())
            total += H
    return hits / total


def test_gbm_band_calibrated_on_noisy_panel():
    """Regression: the default band (split conformal on earlier windows) covers 80 percent on held-out windows, within 0.05.
    Before the fix the same construction gave about 0.68 (measured), because the quantile models fit their own training rows."""
    cov = held_out_coverage(lambda hist: gg.global_forecast(hist, H, season=SEASON))
    assert abs(cov - 0.80) <= 0.05, f"held-out coverage {cov:.3f} is outside 0.80 +/- 0.05"


def test_uncalibrated_band_still_under_covers_on_the_same_construction():
    """The raw band is kept (calibrate_origins=0) so the two can be reported side by side. It must stay under target here,
    which is the reason calibration is needed."""
    cov = held_out_coverage(lambda hist: gg.global_forecast(hist, H, season=SEASON, calibrate_origins=0))
    assert cov < 0.72, f"raw coverage {cov:.3f}"


def test_calibration_does_not_move_the_point_forecast():
    """Calibration only widens or narrows the band around the middle quantile; the point (and so every MASE) is unchanged."""
    hist = {k: v[:200] for k, v in heavy_panel(5).items()}
    raw = gg.global_forecast(hist, H, season=SEASON, calibrate_origins=0)
    cal = gg.global_forecast(hist, H, season=SEASON)
    for k in hist:
        np.testing.assert_allclose(cal[k]["point"], raw[k]["point"], rtol=1e-9, atol=1e-9)
        assert (np.asarray(cal[k]["q10"]) <= np.asarray(cal[k]["point"]) + 1e-9).all()
        assert (np.asarray(cal[k]["point"]) <= np.asarray(cal[k]["q90"]) + 1e-9).all()


def test_calibration_uses_only_windows_before_the_forecast_origin():
    """The calibration scores and their training rows come from the history itself. Changing values after the forecast
    origin cannot change the band, because the origin is the end of the history that is passed in."""
    full = heavy_panel(7)
    T = 200
    hist = {k: v[:T] for k, v in full.items()}
    later = {k: np.concatenate([v[:T], v[T:] * 1000.0]) for k, v in full.items()}
    a = gg.global_forecast(hist, H, season=SEASON)
    b = gg.global_forecast({k: v[:T] for k, v in later.items()}, H, season=SEASON)
    for k in hist:
        np.testing.assert_array_equal(a[k]["q10"], b[k]["q10"])
        np.testing.assert_array_equal(a[k]["q90"], b[k]["q90"])


def test_a_short_panel_falls_back_to_the_raw_band_with_a_reason():
    """Too few windows to calibrate: the band is the raw one and the reason is recorded, never silently narrowed."""
    # 20 points leave no window of 7 after the 14-point minimum training span, so no calibration window fits.
    short = {k: v[:20] for k, v in heavy_panel(3, entities=3, T=20).items()}
    out, info = gg.global_forecast_info(short, H, season=SEASON)
    raw = gg.global_forecast(short, H, season=SEASON, calibrate_origins=0)
    assert info["fallback"] and "calibration" in info["fallback"]
    for k in short:
        np.testing.assert_allclose(out[k]["q10"], raw[k]["q10"])
        np.testing.assert_allclose(out[k]["q90"], raw[k]["q90"])


def test_panel_backtest_reports_raw_and_calibrated_coverage_side_by_side():
    panel = heavy_panel(1, entities=6, T=160)
    res = gg.panel_backtest(panel, H, n_origins=3, season=SEASON, extended=True)
    assert set(res["coverage_80_raw"]) == {"global_gbm"} and set(res["coverage_80"]) == {"global_gbm"}
    assert 0 <= res["coverage_80_raw"]["global_gbm"] <= 1 and 0 <= res["coverage_80"]["global_gbm"] <= 1
    assert res["calibration"]["conformal_level"] == pytest.approx(0.8)
