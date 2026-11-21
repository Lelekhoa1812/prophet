"""Unit tests for the sandbox statistics toolkit (prophet_stats). Synthetic data with fixed seeds, so every run is the same."""
from __future__ import annotations

import importlib
import json
import sys

import numpy as np
import pandas as pd
import pytest

from prophet_agent.sandbox import prophet_stats as ps


def _rng(seed: int = 0) -> np.random.Generator:
    return np.random.default_rng(seed)


def _json_safe(obj) -> None:
    """Fail if the object cannot be written as strict JSON (NaN and numpy types are not allowed)."""
    json.dumps(obj, allow_nan=False)


# ---------------------------------------------------------------------------------------------------------------------
# hampel
# ---------------------------------------------------------------------------------------------------------------------

def test_hampel_flags_injected_spike():
    x = np.sin(np.arange(120) / 5.0) + _rng(1).normal(0, 0.05, 120)
    x[60] += 10.0
    flags, report = ps.hampel(x, window=7, n_sigmas=3.0, causal=True)
    assert flags[60]
    assert report["count"] == int(flags.sum())
    assert report["causal"] is True and report["window"] == 7
    _json_safe(report)


def test_hampel_causal_does_not_use_future_values():
    x = np.sin(np.arange(120) / 5.0) + _rng(2).normal(0, 0.05, 120)
    x[30] += 8.0
    changed = x.copy()
    changed[60:] += 50.0  # only future values move
    f1, _ = ps.hampel(x, causal=True)
    f2, _ = ps.hampel(changed, causal=True)
    assert np.array_equal(f1[:60], f2[:60])


def test_hampel_constant_nan_and_short_inputs():
    flags, report = ps.hampel(np.full(30, 4.0))
    assert not flags.any() and report["count"] == 0
    flags, report = ps.hampel(np.full(30, np.nan))
    assert not flags.any() and report["share"] == 0.0
    with pytest.raises(ValueError, match="at least 7 points"):
        ps.hampel(np.arange(5.0), window=7)


# ---------------------------------------------------------------------------------------------------------------------
# stl_outliers
# ---------------------------------------------------------------------------------------------------------------------

def test_stl_outliers_flags_injected_spike_and_reports_non_causal():
    t = np.arange(240)
    x = 10 + 3 * np.sin(2 * np.pi * t / 12) + _rng(3).normal(0, 0.2, 240)
    x[100] += 12.0
    flags, report = ps.stl_outliers(x, period=12, k=3.0)
    assert flags[100]
    assert report["causal"] is False and report["period"] == 12
    _json_safe(report)


def test_stl_outliers_needs_two_full_periods():
    with pytest.raises(ValueError, match="at least 24 points"):
        ps.stl_outliers(np.arange(20.0), period=12)


# ---------------------------------------------------------------------------------------------------------------------
# impute
# ---------------------------------------------------------------------------------------------------------------------

def test_impute_linear_fills_short_gap_and_reports_it():
    x = np.arange(20.0)
    x[8:10] = np.nan
    s, report = ps.impute(x)
    assert np.allclose(s.iloc[8:10], [8.0, 9.0])
    assert report["filled_count"] == 2 and report["filled_positions"] == [8, 9]
    assert report["method_by_gap"][0]["method"] == "linear"
    _json_safe(report)


def test_impute_never_fills_leading_or_trailing_missing():
    x = np.arange(20.0)
    x[:3] = np.nan
    x[-2:] = np.nan
    s, report = ps.impute(x, method="linear")
    assert s.iloc[:3].isna().all() and s.iloc[-2:].isna().all()
    assert report["leading_missing"] == 3 and report["trailing_missing"] == 2
    assert report["filled_count"] == 0


def test_impute_auto_leaves_forty_point_gap_unfilled():
    x = np.sin(np.arange(100) / 7.0)
    x[30:70] = np.nan
    s, report = ps.impute(x, period=7)
    assert s.iloc[30:70].isna().all()
    assert report["filled_count"] == 0
    assert report["unfilled_gaps"][0]["start"] == 30 and report["unfilled_gaps"][0]["length"] == 40


def test_impute_seasonal_fill_is_exact_on_periodic_series():
    pattern = np.array([1.0, 3.0, 2.0, 5.0, 4.0, 0.0, 6.0])
    x = np.tile(pattern, 10)
    original = x.copy()
    x[30:35] = np.nan  # length 5: longer than 3, not longer than the period of 7
    s, report = ps.impute(x, period=7)
    assert report["method_by_gap"][0]["method"] == "seasonal"
    assert np.allclose(s.to_numpy(), original)


def test_impute_causal_is_unchanged_by_future_values():
    x = np.linspace(0, 10, 60)
    x[20:22] = np.nan
    changed = x.copy()
    changed[50:] += 100.0  # future only
    a, _ = ps.impute(x, causal=True)
    b, _ = ps.impute(changed, causal=True)
    assert np.allclose(a.iloc[:50].to_numpy(), b.iloc[:50].to_numpy(), equal_nan=True)


def test_impute_causal_refuses_two_sided_methods():
    x = np.linspace(0, 10, 40)
    x[10:12] = np.nan
    for method in ("spline", "knn", "kalman"):
        with pytest.raises(ValueError, match="two-sided"):
            ps.impute(x, method=method, causal=True)


def test_impute_spline_knn_and_kalman_fill_interior_gap():
    x = np.sin(np.arange(80) / 6.0)
    x[40:43] = np.nan
    for method in ("spline", "knn", "kalman"):
        s, report = ps.impute(x, method=method)
        assert s.iloc[40:43].notna().all(), method
        assert report["filled_count"] == 3, method
        assert report["method_by_gap"][0]["method"] == method


def test_impute_max_gap_caps_filling():
    x = np.arange(30.0)
    x[5:9] = np.nan  # gap of 4
    s, report = ps.impute(x, method="linear", max_gap=3)
    assert s.iloc[5:9].isna().all()
    assert "max_gap" in report["unfilled_gaps"][0]["reason"]


def test_impute_rejects_unknown_method_and_seasonal_without_period():
    with pytest.raises(ValueError, match="method must be one of"):
        ps.impute(np.arange(5.0), method="magic")
    with pytest.raises(ValueError, match="needs a period"):
        ps.impute(np.arange(5.0), method="seasonal")


# ---------------------------------------------------------------------------------------------------------------------
# smooth
# ---------------------------------------------------------------------------------------------------------------------

def test_smooth_savgol_causal_raises():
    with pytest.raises(ValueError, match="two-sided"):
        ps.smooth(np.arange(30.0), method="sg", causal=True)
    with pytest.raises(ValueError, match="two-sided"):
        ps.smooth(np.arange(30.0), method="loess", causal=True)


def test_smooth_ewm_is_causal():
    x = _rng(4).normal(0, 1, 80)
    changed = x.copy()
    changed[50:] += 30.0
    a = ps.smooth(x, method="ewm", window=5, causal=True)
    b = ps.smooth(changed, method="ewm", window=5, causal=True)
    assert np.allclose(a.iloc[:50].to_numpy(), b.iloc[:50].to_numpy())


def test_smooth_keeps_nan_missing():
    x = pd.Series(_rng(5).normal(0, 1, 40))
    x.iloc[10] = np.nan
    for method in ("ewm", "median", "mean"):
        out = ps.smooth(x, method=method, window=5)
        assert np.isnan(out.iloc[10]), method
        assert isinstance(out, pd.Series) and out.index.equals(x.index)


def test_smooth_loess_non_causal_runs_and_keeps_nan():
    x = np.sin(np.arange(100) / 9.0) + _rng(6).normal(0, 0.1, 100)
    x[50] = np.nan
    out = ps.smooth(x, method="loess", window=11, causal=False)
    assert np.isnan(out.iloc[50]) and out.notna().sum() == 99
    sg = ps.smooth(x, method="sg", window=9, causal=False)
    assert np.isnan(sg.iloc[50]) and sg.notna().sum() == 99


# ---------------------------------------------------------------------------------------------------------------------
# changepoints
# ---------------------------------------------------------------------------------------------------------------------

def test_changepoints_finds_injected_mean_shift():
    rng = _rng(7)
    x = np.concatenate([rng.normal(0, 1, 100), rng.normal(2, 1, 100)])
    changes, report = ps.changepoints(x, min_size=20)
    assert report["n_changes"] >= 1
    assert any(abs(c["index"] - 100) <= 5 for c in changes)
    first = changes[0]
    assert first["after_mean"] > first["before_mean"]
    _json_safe({"changes": changes, "report": report})


def test_changepoints_flat_noise_has_no_changes():
    changes, report = ps.changepoints(_rng(8).normal(0, 1, 200), min_size=20)
    assert changes == [] and report["n_changes"] == 0


def test_changepoints_constant_and_short_inputs():
    changes, report = ps.changepoints(np.full(60, 3.0), min_size=10)
    assert changes == [] and report["penalty"] == 0.0
    with pytest.raises(ValueError, match="at least 40"):
        ps.changepoints(np.arange(30.0), min_size=20)


# ---------------------------------------------------------------------------------------------------------------------
# derivative_features
# ---------------------------------------------------------------------------------------------------------------------

def test_derivative_features_columns_and_linear_ramp():
    x = 2.0 * np.arange(40) + 1.0
    df = ps.derivative_features(x, windows=(3, 7))
    assert list(df.columns) == [
        "diff1", "diff2", "slope_3", "std_3", "z_3", "slope_7", "std_7", "z_7",
    ]
    assert np.allclose(df["diff1"].iloc[1:], 2.0)
    assert np.allclose(df["diff2"].iloc[2:], 0.0)
    assert np.allclose(df["slope_7"].iloc[6:], 2.0)


def test_derivative_features_are_causal():
    rng = _rng(9)
    x = rng.normal(0, 1, 60)
    changed = x.copy()
    changed[40:] += 25.0
    a = ps.derivative_features(x, windows=(3, 7, 28))
    b = ps.derivative_features(changed, windows=(3, 7, 28))
    assert np.allclose(a.iloc[:40].to_numpy(), b.iloc[:40].to_numpy(), equal_nan=True)


def test_derivative_features_rejects_short_input_and_bad_window():
    with pytest.raises(ValueError, match="at least 28 points"):
        ps.derivative_features(np.arange(10.0), windows=(3, 28))
    with pytest.raises(ValueError, match="at least 2 points"):
        ps.derivative_features(np.arange(10.0), windows=(1,))


# ---------------------------------------------------------------------------------------------------------------------
# censoring_check
# ---------------------------------------------------------------------------------------------------------------------

def test_censoring_check_finds_injected_zero_run_between_sales():
    x = _rng(10).uniform(5, 10, 100)
    x[40:50] = 0.0
    report = ps.censoring_check(x)
    assert report["probable_stockouts"] == [{"start": 40, "end": 49, "length": 10}]
    assert report["longest_zero_run"] == 10
    assert "stock" in report["recommendation"].lower() or "outage" in report["recommendation"].lower()
    _json_safe(report)


def test_censoring_check_ignores_edge_zero_runs_and_reports_plateaus():
    x = _rng(11).uniform(2, 9, 60)
    x[:5] = 0.0  # zero run at the start of the record: not a stockout between sales
    report = ps.censoring_check(x)
    assert report["probable_stockouts"] == []
    assert report["longest_zero_run"] == 5

    y = _rng(11).uniform(2, 9, 60)
    y[30:34] = 1.0  # a run at the minimum (no zeros, so the minimum is 1.0)
    report = ps.censoring_check(y)
    assert report["floor_plateau"] is not None
    assert report["floor_plateau"]["value"] == 1.0 and report["floor_plateau"]["longest_run"] == 4
    assert report["ceiling_plateau"] is None


def test_censoring_check_all_nan_and_constant_do_not_crash():
    r = ps.censoring_check(np.full(10, np.nan))
    assert r["observed"] == 0 and r["probable_stockouts"] == []
    r = ps.censoring_check(np.full(10, 5.0))
    assert r["floor_plateau"] is not None and "constant" in r["recommendation"]
    _json_safe(r)


# ---------------------------------------------------------------------------------------------------------------------
# seasonal_strength and stationarity
# ---------------------------------------------------------------------------------------------------------------------

def test_seasonal_strength_high_for_sine_low_for_white_noise():
    t = np.arange(240)
    sine = np.sin(2 * np.pi * t / 12) + _rng(12).normal(0, 0.1, 240)
    strong = ps.seasonal_strength(sine, period=12)
    assert strong["seasonal_strength"] > 0.9
    noise = ps.seasonal_strength(_rng(13).normal(0, 1, 240), period=12)
    assert noise["seasonal_strength"] < 0.4
    assert strong["period"] == 12 and strong["n"] == 240
    _json_safe(strong)


def test_stationarity_white_noise_and_random_walk():
    # KPSS rejects at 5 percent about one time in twenty even for white noise, so the white-noise check asserts the
    # ADF rejection and accepts either stationary verdict rather than pinning one seed's KPSS p-value.
    rng = _rng(14)
    white = ps.stationarity(rng.normal(0, 1, 300))
    assert white["adf_p"] < 0.05
    assert white["verdict"] in ("stationary", "trend_stationary")
    walk = np.cumsum(_rng(15).normal(0, 1, 500))
    assert ps.stationarity(walk)["verdict"] == "unit_root"


def test_stationarity_constant_and_short_series():
    r = ps.stationarity(np.full(30, 2.0))
    assert r["verdict"] == "inconclusive" and r["adf_p"] is None
    with pytest.raises(ValueError, match="at least 20"):
        ps.stationarity(np.arange(10.0))


# ---------------------------------------------------------------------------------------------------------------------
# backtest_treatment
# ---------------------------------------------------------------------------------------------------------------------

_BT_N = 400
_BT_SPIKES = (101, 196, 291, 387)  # each sits just before an origin, inside its training window, never in an actual


def _bt_series() -> np.ndarray:
    t = np.arange(_BT_N)
    x = 5.0 * np.sin(2 * np.pi * t / 7) + _rng(16).normal(0, 0.3, _BT_N) + 20.0
    for p in _BT_SPIKES:
        # Large enough that the trailing Hampel flags it at every phase of the period-7 cycle (a +20 spike was
        # missed at two phases, because the seasonal swing inflates the MAD).
        x[p] += 40.0
    return x


def _despike(s: pd.Series) -> pd.Series:
    """Trailing Hampel replacement: uses only the series it is given (the training data)."""
    flags, _ = ps.hampel(s, window=7, n_sigmas=3.0, causal=True)
    med = s.rolling(7, min_periods=4).median().to_numpy()
    out = s.to_numpy(dtype=float).copy()
    out[flags] = med[flags]
    return pd.Series(out)


def test_backtest_identity_treatment_is_not_helped():
    report = ps.backtest_treatment(_bt_series(), lambda s: s.copy(), horizon=10, origins=5, period=7)
    assert report["helped"] is False
    assert report["wins"] == 0
    assert report["mase_treated"] == pytest.approx(report["mase_untreated"])
    _json_safe(report)


def test_backtest_spike_removal_helped_when_spikes_are_only_in_training():
    x = _bt_series()
    origins = 5
    horizon = 10
    cut = np.unique(np.round(np.linspace(8, _BT_N - horizon, origins)).astype(int))
    for p in _BT_SPIKES:  # the spikes must never fall in an actual window
        assert not any(o <= p < o + horizon for o in cut)
    report = ps.backtest_treatment(x, _despike, horizon=horizon, origins=origins, period=7)
    assert report["helped"] is True
    assert report["wins"] >= 3
    assert report["mean_improvement_pct"] > 2.0
    _json_safe(report)


def test_backtest_treatment_only_sees_history_before_each_origin():
    x = _bt_series()
    seen: list[int] = []

    def recording(s: pd.Series) -> pd.Series:
        seen.append(len(s))
        assert np.array_equal(s.to_numpy(), x[: len(s)])  # exactly the data before the origin, nothing after
        return s.copy()

    ps.backtest_treatment(x, recording, horizon=10, origins=5, period=7)
    assert seen == sorted(seen)
    assert max(seen) <= _BT_N - 10


def test_backtest_rejects_short_series_and_bad_treat_length():
    with pytest.raises(ValueError, match="needs at least"):
        ps.backtest_treatment(np.arange(12.0), lambda s: s, horizon=10, period=7)
    with pytest.raises(ValueError, match="same length"):
        ps.backtest_treatment(_bt_series(), lambda s: s.iloc[:-1], horizon=10, period=7)


# ---------------------------------------------------------------------------------------------------------------------
# isolation_outliers
# ---------------------------------------------------------------------------------------------------------------------

def test_isolation_outliers_flags_obvious_row():
    X = _rng(17).normal(0, 1, (200, 2))
    X[0] = [9.0, 9.0]
    flags, report = ps.isolation_outliers(X, contamination=0.01, random_state=0)
    assert flags[0]
    assert report["rows_scored"] == 200 and report["rows_skipped_missing"] == 0
    _json_safe(report)


def test_isolation_outliers_skips_missing_rows_and_needs_enough_rows():
    X = _rng(18).normal(0, 1, (30, 1))
    X[3, 0] = np.nan
    flags, report = ps.isolation_outliers(X)
    assert not flags[3] and report["rows_skipped_missing"] == 1
    with pytest.raises(ValueError, match="at least 10"):
        ps.isolation_outliers(np.arange(5.0))


# ---------------------------------------------------------------------------------------------------------------------
# Ledger
# ---------------------------------------------------------------------------------------------------------------------

def test_ledger_save_round_trip(tmp_path):
    ledger = ps.Ledger()
    ledger.add("stage 2", "Gap of 2 points at day 8", "Linear fill", evidence={"gap": 2, "share": np.float64(0.01)})
    ledger.add("stage 3", "Spike at index 60", "Flag and keep", evidence=None)
    entries = ledger.to_dict()
    assert [e["index"] for e in entries] == [1, 2]
    target = tmp_path / "out" / "ledger.json"
    written = ledger.save(str(target))
    assert target.exists() and written == str(target)
    loaded = json.loads(target.read_text(encoding="utf-8"))
    assert loaded == entries
    assert loaded[0]["evidence"] == {"gap": 2, "share": 0.01}


def test_ledger_rejects_empty_fields():
    ledger = ps.Ledger()
    with pytest.raises(ValueError, match="finding"):
        ledger.add("stage 1", "  ", "keep")
    assert ledger.to_dict() == []


# ---------------------------------------------------------------------------------------------------------------------
# Optional scikit-learn
# ---------------------------------------------------------------------------------------------------------------------

def test_module_loads_and_core_functions_work_without_sklearn(monkeypatch):
    for key in [k for k in list(sys.modules) if k == "sklearn" or k.startswith("sklearn.")]:
        monkeypatch.setitem(sys.modules, key, None)
    monkeypatch.setitem(sys.modules, "sklearn", None)
    reloaded = importlib.reload(ps)
    try:
        flags, _ = reloaded.hampel(np.arange(30.0))
        assert flags.shape == (30,)
        s, _ = reloaded.impute(np.array([1.0, np.nan, 3.0, 4.0]))
        assert s.iloc[1] == 2.0
        assert reloaded.censoring_check(np.arange(10.0))["observed"] == 10
    finally:
        monkeypatch.undo()
        importlib.reload(ps)


# ---------------------------------------------------------------------------------------------------------------------
# knn_impute (causal; lag and calendar features)
# ---------------------------------------------------------------------------------------------------------------------

def _weekly_table(n: int = 140, gap: int | None = 60) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=n, freq="D")
    y = 10.0 + 3.0 * np.sin(2 * np.pi * np.arange(n) / 7)
    if gap is not None:
        y[gap] = np.nan
    return pd.DataFrame({"y": y}, index=idx)


def test_knn_fill_matches_weekly_truth_for_a_single_gap():
    table = _weekly_table(140, gap=60)
    truth = 10.0 + 3.0 * np.sin(2 * np.pi * 60 / 7)
    filled, report = ps.knn_impute(table, ["y"], k=5, seasonal_lags=(1, 7))
    assert filled["y"].iloc[60] == pytest.approx(truth, abs=1e-6)
    assert report["positions"]["y"] == [60] and report["count"] == 1
    _json_safe(report)


def test_knn_fill_uses_only_values_before_the_gap():
    base = _weekly_table(140, gap=60)
    poisoned = base.copy()
    poisoned.iloc[61:, 0] = poisoned.iloc[61:, 0] + 1000.0  # only values after the gap move
    f1, _ = ps.knn_impute(base, ["y"])
    f2, _ = ps.knn_impute(poisoned, ["y"])
    assert f1["y"].iloc[60] == pytest.approx(f2["y"].iloc[60])
    assert np.array_equal(f1["y"].to_numpy()[:60], f2["y"].to_numpy()[:60], equal_nan=True)


def test_knn_does_not_mutate_input_and_is_deterministic():
    table = _weekly_table(140, gap=60)
    before = table.copy()
    f1, r1 = ps.knn_impute(table, ["y"])
    f2, r2 = ps.knn_impute(table, ["y"])
    pd.testing.assert_frame_equal(table, before)
    pd.testing.assert_frame_equal(f1, f2)
    assert r1 == r2 and f1 is not table


def test_knn_report_names_method_causal_and_calendar():
    table = _weekly_table(140, gap=60)
    _, report = ps.knn_impute(table, ["y"], k=3)
    assert report["method"] == "knn_causal" and report["causal"] is True
    assert report["k"] == 3 and report["seasonal_lags"] == [1, 7] and report["calendar_features"] is True
    assert report["donor_shortfalls"] == {"y": []} and report["unfilled"] == {"y": []}


def test_knn_leaves_leading_and_trailing_missing_alone():
    table = _weekly_table(60, gap=None)
    table.iloc[:3, 0] = np.nan
    table.iloc[-2:, 0] = np.nan
    table.iloc[30, 0] = np.nan
    filled, report = ps.knn_impute(table, ["y"])
    assert np.isnan(filled["y"].iloc[:3]).all() and np.isnan(filled["y"].iloc[-2:]).all()
    assert report["positions"]["y"] == [30]
    assert 0 not in report["positions"]["y"] and 58 not in report["positions"]["y"]


def test_knn_all_missing_column_stays_missing_without_crashing():
    table = _weekly_table(30, gap=None)
    table["z"] = np.nan
    filled, report = ps.knn_impute(table, ["z"])
    assert filled["z"].isna().all() and report["count"] == 0
    assert report["unfilled"] == {"z": []}  # no interior gap to fill, nothing listed


def test_knn_rejects_bad_inputs():
    table = _weekly_table(30, gap=10)
    with pytest.raises(ValueError, match="needs a table"):
        ps.knn_impute(np.arange(10.0), ["y"])
    with pytest.raises(ValueError, match="not in the table"):
        ps.knn_impute(table, ["nope"])
    with pytest.raises(ValueError, match="at least 1"):
        ps.knn_impute(table, ["y"], k=0)
    with pytest.raises(ValueError, match="positive whole numbers"):
        ps.knn_impute(table, ["y"], seasonal_lags=(0, 7))


# ---------------------------------------------------------------------------------------------------------------------
# kalman_smooth (local level, plain numpy)
# ---------------------------------------------------------------------------------------------------------------------

def test_kalman_recovers_constant_level_and_reduces_noise():
    x = pd.Series(5.0 + _rng(20).normal(0, 1.0, 300))
    smoothed, report = ps.kalman_smooth(x)
    assert abs(float(smoothed.iloc[150:].mean()) - 5.0) < 0.3
    assert float(smoothed.iloc[150:].std()) < 0.7 * float(x.iloc[150:].std())  # a causal filter lags, so it removes less noise than a smoother would
    assert report["q_ratio_source"] == "first_half_moments" and report["causal"] is True
    assert report["diagnostic_only"] is False
    _json_safe(report)


def test_kalman_tracks_noiseless_ramp_with_a_large_level_variance():
    ramp = pd.Series(np.arange(30.0) * 2.0 + 1.0)
    smoothed, report = ps.kalman_smooth(ramp, q_ratio=1e4)
    assert np.max(np.abs(smoothed.to_numpy() - ramp.to_numpy())) < 1e-2
    assert report["q_ratio"] == pytest.approx(1e4) and report["q_ratio_source"] == "given"


def test_kalman_flags_a_spike_and_keeps_it_in_the_output():
    y = _rng(21).normal(0, 1.0, 100)
    y[40] += 15.0
    filtered, report = ps.kalman_smooth(pd.Series(y))
    positions = [o["position"] for o in report["outliers"]]
    assert 40 in positions and report["outliers_kept"] is True
    assert report["max_influence_position"] == 40
    assert filtered.iloc[40] > filtered.iloc[39]  # kept, not removed: the filtered level moves toward the spike
    assert all(abs(o["z"]) > ps._OUTLIER_Z for o in report["outliers"])


def test_kalman_diagnostic_fills_interior_gap_and_keeps_ends_missing():
    y = pd.Series(np.linspace(0.0, 10.0, 40))
    y.iloc[[0, 1, 20, 38, 39]] = np.nan
    smoothed, report = ps.kalman_smooth(y, q_ratio=1e4, causal=False)
    assert np.isfinite(smoothed.iloc[20]) and abs(smoothed.iloc[20] - 10.0 * 20 / 39) < 0.05
    assert np.isnan(smoothed.iloc[[0, 1, 38, 39]]).all()
    assert report["interior_state_filled"] == 1
    assert report["diagnostic_only"] is True and "reads values after" in report["leak_warning"]


def test_kalman_constant_and_invalid_inputs():
    flat, report = ps.kalman_smooth(np.full(20, 3.0), causal=False)
    assert np.allclose(flat.to_numpy(), 3.0) and report["q_ratio_source"] == "constant series"
    with pytest.raises(ValueError, match="first half of the series is constant"):
        ps.kalman_smooth(np.full(20, 3.0))
    with pytest.raises(ValueError, match="at least 3 observed"):
        ps.kalman_smooth(np.array([1.0, np.nan, np.nan]))
    with pytest.raises(ValueError, match="zero or more"):
        ps.kalman_smooth(np.arange(10.0), q_ratio=-1.0)


# ---------------------------------------------------------------------------------------------------------------------
# savgol_smooth (edge-safe, two-sided, with report)
# ---------------------------------------------------------------------------------------------------------------------

def test_savgol_causal_reproduces_a_straight_line_after_warmup():
    line = np.arange(20.0) * 3.0 + 2.0
    smoothed, report = ps.savgol_smooth(line, window=5, order=2)
    assert np.isnan(smoothed.iloc[:4]).all()
    assert np.allclose(smoothed.to_numpy()[4:], line[4:], atol=1e-9)
    assert report["causal"] is True and report["diagnostic_only"] is False and report["warmup_points"] == 4


def test_savgol_diagnostic_reproduces_a_straight_line_including_the_edges():
    line = np.arange(20.0) * 3.0 + 2.0
    smoothed, report = ps.savgol_smooth(line, window=5, order=2, causal=False)
    assert np.allclose(smoothed.to_numpy(), line, atol=1e-9)
    assert report["causal"] is False and report["diagnostic_only"] is True and report["edge_points_each_side"] == 2
    assert "reads values after" in report["leak_warning"]


def test_savgol_reproduces_a_quadratic_with_order_two():
    t = np.arange(30.0)
    quad = 0.5 * t**2 - 2.0 * t + 7.0
    smoothed, _ = ps.savgol_smooth(quad, window=7, order=2)
    assert np.allclose(smoothed.to_numpy()[6:], quad[6:], atol=1e-6)
    centred, _ = ps.savgol_smooth(quad, window=7, order=2, causal=False)
    assert np.allclose(centred.to_numpy(), quad, atol=1e-6)


def test_savgol_keeps_missing_positions_missing_and_reports_them():
    x = np.arange(25.0)
    x[[0, 12]] = np.nan
    smoothed, report = ps.savgol_smooth(x, window=5, order=1)
    assert np.isnan(smoothed.iloc[12:17]).all() and np.isfinite(smoothed.iloc[17])  # windows holding a gap are NaN
    assert report["missing_kept"] == 2
    _, centred = ps.savgol_smooth(x, window=5, order=1, causal=False)
    assert centred["span"] == [1, 25] and centred["interpolated_for_filter"] == 1


def test_savgol_rejects_even_window_order_and_short_span():
    with pytest.raises(ValueError, match="odd window"):
        ps.savgol_smooth(np.arange(20.0), window=4, order=2)
    with pytest.raises(ValueError, match="below the window"):
        ps.savgol_smooth(np.arange(20.0), window=5, order=5)
    with pytest.raises(ValueError, match="at least 9 points"):
        ps.savgol_smooth(np.arange(5.0), window=9, order=2, causal=False)


# ---------------------------------------------------------------------------------------------------------------------
# rolling_slope_features (trailing windows only)
# ---------------------------------------------------------------------------------------------------------------------

def test_slope_features_are_exact_for_a_straight_line():
    line = pd.Series(np.arange(40.0) * 2.0 + 1.0)
    feats = ps.rolling_slope_features(line, windows=(3, 7))
    assert list(feats.columns) == ["diff1", "diff2", "slope_3", "curvature_3", "change_z_3",
                                   "slope_7", "curvature_7", "change_z_7"]
    tail = feats.iloc[10:]
    assert np.allclose(tail["slope_7"], 2.0) and np.allclose(tail["slope_3"], 2.0)
    assert np.allclose(tail["curvature_7"], 0.0, atol=1e-9)
    assert np.allclose(feats["diff1"].iloc[1:], 2.0) and np.allclose(feats["diff2"].iloc[2:], 0.0)


def test_slope_and_curvature_are_exact_for_a_quadratic():
    t = np.arange(40.0)
    feats = ps.rolling_slope_features(t**2, windows=(5,))
    # At position i the fitted quadratic is exact, so its slope at the latest point is 2i and curvature is 2.
    assert np.allclose(feats["slope_5"].iloc[4:], 2.0 * t[4:], atol=1e-6)
    assert np.allclose(feats["curvature_5"].iloc[4:], 2.0, atol=1e-6)


def test_slope_features_have_no_look_ahead():
    x = _rng(30).normal(0, 1, 120).cumsum()
    poisoned = x.copy()
    poisoned[80:] += 1000.0
    f1 = ps.rolling_slope_features(x, windows=(3, 7, 28))
    f2 = ps.rolling_slope_features(poisoned, windows=(3, 7, 28))
    pd.testing.assert_frame_equal(f1.iloc[:80], f2.iloc[:80])
    assert not np.allclose(f1.iloc[80:].to_numpy(dtype=float), f2.iloc[80:].to_numpy(dtype=float), equal_nan=True)


def test_slope_missing_value_only_blanks_the_windows_that_hold_it():
    x = np.arange(30.0)
    x[15] = np.nan
    feats = ps.rolling_slope_features(x, windows=(3,))
    assert feats["slope_3"].iloc[15:18].isna().all()  # the three windows that contain position 15
    assert feats["slope_3"].iloc[:2].isna().all()  # no full window yet
    assert feats["slope_3"].iloc[2:15].notna().all() and feats["slope_3"].iloc[18:].notna().all()


def test_slope_change_z_is_nan_on_zero_spread_and_inputs_are_checked():
    feats = ps.rolling_slope_features(np.arange(20.0), windows=(4,))
    assert feats["change_z_4"].iloc[5:].isna().all()  # constant first differences have no spread
    with pytest.raises(ValueError, match="at least 3 points"):
        ps.rolling_slope_features(np.arange(20.0), windows=(2,))
    with pytest.raises(ValueError, match="at least 28 points"):
        ps.rolling_slope_features(np.arange(10.0))


# ---------------------------------------------------------------------------------------------------------------------
# shrink_to_group (normal-normal empirical Bayes)
# ---------------------------------------------------------------------------------------------------------------------

def test_shrink_analytic_factor_is_one_half():
    # Sample variance of [0, 2] is 2 (ddof 1). Sampling variance 1 per series gives tau^2 = 1, so B = 1 / (1 + 1) = 0.5.
    shrunk, report = ps.shrink_to_group([0.0, 2.0], within_var=1.0, n_obs=1)
    assert np.allclose(shrunk, [0.5, 1.5])
    assert report["between_series_variance"] == pytest.approx(1.0)
    assert report["shrinkage_factors"] == pytest.approx([0.5, 0.5])
    assert report["group_mean"] == pytest.approx(1.0)


def test_shrink_identical_estimates_collapse_to_group_mean():
    shrunk, report = ps.shrink_to_group([1.0, 1.0, 1.0], within_var=0.5, n_obs=4)
    assert np.allclose(shrunk, 1.0)
    assert report["between_series_variance"] == 0.0 and report["shrinkage_factors"] == [1.0, 1.0, 1.0]


def test_shrink_precise_series_keep_their_own_estimates():
    shrunk, report = ps.shrink_to_group([0.0, 10.0], within_var=1e-6, n_obs=1)
    assert np.allclose(shrunk, [0.0, 10.0], atol=1e-3)
    assert max(report["shrinkage_factors"]) < 1e-5


def test_shrink_poisson_default_and_nan_are_skipped_not_guessed():
    shrunk, report = ps.shrink_to_group([0.1, 0.4, np.nan, 0.2], n_obs=[10, 10, 10, 10])
    assert np.isnan(shrunk[2]) and report["skipped"] == 1
    assert report["variance_source"] == "poisson_rate"
    kept = [f for f in report["shrinkage_factors"] if f is not None]
    assert len(kept) == 3 and all(0.0 <= f <= 1.0 for f in kept)
    _json_safe(report)


def test_shrink_rejects_single_series_and_bad_counts():
    with pytest.raises(ValueError, match="at least two series"):
        ps.shrink_to_group([0.3])
    with pytest.raises(ValueError, match="n_obs must be positive"):
        ps.shrink_to_group([0.1, 0.2], n_obs=[5, 0])
    with pytest.raises(ValueError, match="within_var must be positive"):
        ps.shrink_to_group([0.1, 0.2], within_var=-1.0)


# ---------------------------------------------------------------------------------------------------------------------
# combine_forecasts
# ---------------------------------------------------------------------------------------------------------------------

def test_combine_default_is_equal_weight_mean():
    combined, report = ps.combine_forecasts(np.array([[10.0, 20.0], [20.0, 30.0], [30.0, 40.0]]))
    assert np.allclose(combined, [20.0, 30.0])
    assert report["method"] == "mean" and report["weights"] == pytest.approx([1 / 3] * 3)
    _json_safe(report)


def test_combine_median_takes_the_middle_forecast():
    combined, report = ps.combine_forecasts(np.array([[10.0], [20.0], [90.0]]), weights="median")
    assert combined[0] == 20.0 and report["weights"] is None


def test_combine_inverse_error_weights_are_analytic():
    # errors [1, 1, 2] give weights [0.4, 0.4, 0.2]; 0.4 * 10 + 0.4 * 20 + 0.2 * 30 = 18.
    combined, report = ps.combine_forecasts(np.array([[10.0], [20.0], [30.0]]), weights="inverse_error", errors=[1, 1, 2])
    assert combined[0] == pytest.approx(18.0)
    assert report["weights"] == pytest.approx([0.4, 0.4, 0.2])


def test_combine_renormalises_weights_when_a_model_is_missing_a_step():
    forecasts = np.array([[10.0, 10.0], [20.0, np.nan], [30.0, 40.0]])
    combined, report = ps.combine_forecasts(forecasts, weights="inverse_error", errors=[1, 1, 2])
    assert combined[0] == pytest.approx(18.0)
    # Step 2 has models 1 and 3 only (weights 0.4 and 0.2, renormalised by 0.6): (0.4 * 10 + 0.2 * 40) / 0.6 = 20.
    assert combined[1] == pytest.approx((0.4 * 10.0 + 0.2 * 40.0) / 0.6)
    assert report["partial_steps"] == 1 and report["missing_cells"] == 1
    all_missing, _ = ps.combine_forecasts(np.array([[np.nan], [np.nan]]))
    assert np.isnan(all_missing[0])


def test_combine_accepts_tables_and_lists_and_names_dropped_models():
    table = pd.DataFrame({"h1": [1.0, 3.0], "h2": [2.0, 4.0]}, index=["ts", "ets"])
    c1, _ = ps.combine_forecasts(table)
    c2, _ = ps.combine_forecasts([np.array([1.0, 2.0]), np.array([3.0, 4.0])])
    assert np.allclose(c1, c2) and np.allclose(c1, [2.0, 3.0])
    _, report = ps.combine_forecasts(np.array([[1.0, 2.0], [np.nan, np.nan]]))
    assert report["dropped_models"] == ["model_1"]


def test_combine_rejects_bad_weights_errors_and_shapes():
    with pytest.raises(ValueError, match="weights must be one of"):
        ps.combine_forecasts(np.ones((2, 2)), weights="best")
    with pytest.raises(ValueError, match="need errors"):
        ps.combine_forecasts(np.ones((2, 2)), weights="inverse_error")
    with pytest.raises(ValueError, match="above zero"):
        ps.combine_forecasts(np.ones((2, 2)), weights="inverse_error", errors=[1.0, 0.0])
    with pytest.raises(ValueError, match="one value per model"):
        ps.combine_forecasts(np.ones((2, 2)), weights="inverse_error", errors=[1.0])
    with pytest.raises(ValueError, match="models by horizon"):
        ps.combine_forecasts(np.ones(4))


# ---------------------------------------------------------------------------------------------------------------------
# conformal_interval
# ---------------------------------------------------------------------------------------------------------------------

def test_conformal_half_width_is_the_largest_residual_when_n_is_nine():
    # n = 9 and alpha = 0.1 give rank ceil(10 * 0.9) = 9, the largest absolute residual.
    residuals = np.array([-0.5, 0.2, -1.5, 0.9, 0.1, -0.3, 0.4, 1.2, -0.7])
    lower, upper, report = ps.conformal_interval(residuals, np.array([10.0, 20.0]), alpha=0.1)
    assert np.allclose(lower, [8.5, 18.5]) and np.allclose(upper, [11.5, 21.5])
    assert report["half_width"] == pytest.approx(1.5) and report["rank"] == 9
    assert report["empirical_coverage"] == pytest.approx(1.0)


def test_conformal_coverage_on_simulated_noise_is_close_to_target():
    cal = _rng(40).normal(0, 1, 2000)
    fresh = _rng(41).normal(0, 1, 4000)
    lower, upper, report = ps.conformal_interval(cal, np.zeros(fresh.size), alpha=0.1)
    coverage = float(np.mean((fresh >= lower) & (fresh <= upper)))
    assert abs(coverage - 0.9) < 0.02
    assert abs(report["half_width"] - 1.645) < 0.1
    assert report["empirical_coverage"] >= 0.9


def test_conformal_rolling_follows_recent_volatility():
    rng = _rng(42)
    residuals = np.concatenate([rng.normal(0, 1.0, 500), rng.normal(0, 4.0, 100)])
    _, _, split = ps.conformal_interval(residuals, np.zeros(3), method="split")
    _, _, rolling = ps.conformal_interval(residuals, np.zeros(3), method="rolling", window=100)
    assert rolling["half_width"] > 2.0 * split["half_width"] / 1.5  # recent noise is four times larger
    assert rolling["window"] == 100 and rolling["calibration_size"] == 100


def test_conformal_rejects_too_few_residuals_and_bad_settings():
    with pytest.raises(ValueError, match="at least 9 finite residuals"):
        ps.conformal_interval(np.arange(5.0), np.zeros(1), alpha=0.1)
    with pytest.raises(ValueError, match="between 0 and 1"):
        ps.conformal_interval(np.arange(20.0), np.zeros(1), alpha=1.5)
    with pytest.raises(ValueError, match="needs window"):
        ps.conformal_interval(np.arange(20.0), np.zeros(1), method="rolling")
    with pytest.raises(ValueError, match="more residuals"):
        ps.conformal_interval(np.arange(20.0), np.zeros(1), method="rolling", window=50)


def test_conformal_drops_nan_residuals_and_gives_nan_for_missing_points():
    residuals = np.concatenate([np.linspace(-1, 1, 12), [np.nan, np.nan]])
    lower, upper, report = ps.conformal_interval(residuals, np.array([np.nan, 5.0]))
    assert report["dropped_residuals"] == 2 and report["calibration_size"] == 12
    assert np.isnan(lower[0]) and np.isnan(upper[0]) and upper[1] - lower[1] == pytest.approx(2 * report["half_width"])
    assert report["points_missing"] == 1


# ---------------------------------------------------------------------------------------------------------------------
# fuse_report
# ---------------------------------------------------------------------------------------------------------------------

def test_fuse_report_merges_steps_in_order_and_is_json_safe():
    a = {"method": "hampel", "count": 2, "flags": np.array([1, 0])}
    b = {"method": "knn_causal", "filled_count": np.int64(3)}
    merged = ps.fuse_report(a, b)
    assert [s["name"] for s in merged["steps"]] == ["hampel", "knn_causal"]
    assert merged["steps_count"] == 2 and merged["changed_total"] == 5
    _json_safe(merged)


def test_fuse_report_does_not_change_its_inputs_and_names_unnamed_steps():
    rep = {"count": 1, "nested": {"x": np.float64(2.0)}}
    before = json.dumps(rep, default=str, sort_keys=True)
    merged = ps.fuse_report(rep, {"count": 4})
    assert json.dumps(rep, default=str, sort_keys=True) == before
    assert merged["steps"][0]["name"] == "step_1" and merged["changed_total"] == 5
    merged["steps"][0]["report"]["count"] = 99
    assert rep["count"] == 1


def test_fuse_report_rejects_empty_and_non_dict_inputs():
    with pytest.raises(ValueError, match="at least one report"):
        ps.fuse_report()
    with pytest.raises(ValueError, match="not a dict"):
        ps.fuse_report({"count": 1}, [1, 2])


# ---------------------------------------------------------------------------------------------------------------------
# causal_canary and the causal default of each time-series helper
# ---------------------------------------------------------------------------------------------------------------------

def _canary_walk(seed: int = 50, n: int = 200) -> np.ndarray:
    return _rng(seed).normal(0, 1, n).cumsum() + 20.0


def _knn_series(s: pd.Series) -> pd.Series:
    frame = pd.DataFrame({"y": s.to_numpy()}, index=pd.date_range("2024-01-01", periods=len(s), freq="D"))
    return ps.knn_impute(frame, ["y"])[0]["y"]


def test_canary_flags_a_centred_mean_and_passes_a_trailing_mean():
    x = _canary_walk()
    centred = ps.causal_canary(lambda s: ps.smooth(s, "mean", window=5, causal=False), x)
    trailing = ps.causal_canary(lambda s: ps.smooth(s, "mean", window=5, causal=True), x)
    assert centred["causal"] is False and centred["max_abs_diff_before_origin"] > 0
    assert trailing["causal"] is True and trailing["max_abs_diff_before_origin"] == 0.0
    assert trailing["origin"] == 100 and trailing["shift"] == 500.0


def test_canary_counts_a_nan_that_appears_in_one_run_as_a_failure():
    def gap_if_late(s: pd.Series) -> np.ndarray:
        out = np.zeros(len(s))
        out[10] = np.nan if s.iloc[-1] > 100 else 0.0  # only the shifted run produces the gap
        return out

    result = ps.causal_canary(gap_if_late, _canary_walk())
    assert result["causal"] is False and result["max_abs_diff_before_origin"] == float("inf")


def test_canary_rejects_bad_settings_and_wrong_output_length():
    x = _canary_walk()
    with pytest.raises(ValueError, match="origin must be between"):
        ps.causal_canary(lambda s: s.to_numpy(), x, origin=0)
    with pytest.raises(ValueError, match="finite, non-zero"):
        ps.causal_canary(lambda s: s.to_numpy(), x, shift=0.0)
    with pytest.raises(ValueError, match="at least 4 points"):
        ps.causal_canary(lambda s: s.to_numpy(), np.arange(3.0))
    with pytest.raises(ValueError, match="one value per input point"):
        ps.causal_canary(lambda s: np.array([1.0]), x)


def test_default_mode_of_each_time_series_helper_passes_the_canary():
    x = _canary_walk(seed=51)
    x[[30, 70, 150]] = np.nan  # gaps before and after the origin
    helpers = {
        "knn_impute": _knn_series,
        "kalman_smooth": lambda s: ps.kalman_smooth(s)[0],
        "savgol_smooth": lambda s: ps.savgol_smooth(s, 7, 2)[0],
        "rolling_slope_features": lambda s: ps.rolling_slope_features(s, windows=(3, 7)),
        "combine_forecasts": lambda s: ps.combine_forecasts(np.vstack([s.to_numpy(), 2.0 * s.to_numpy()]))[0],
    }
    for name, fn in helpers.items():
        result = ps.causal_canary(fn, x)
        assert result["causal"] is True, (name, result)
        assert result["max_abs_diff_before_origin"] <= 1e-9, name


def test_diagnostic_modes_fail_the_canary_and_say_so_in_their_reports():
    x = _canary_walk(seed=52)
    assert ps.causal_canary(lambda s: ps.savgol_smooth(s, 7, 2, causal=False)[0], x)["causal"] is False
    assert ps.causal_canary(lambda s: ps.kalman_smooth(s, causal=False)[0], x)["causal"] is False
    _, sg_report = ps.savgol_smooth(x, 7, 2, causal=False)
    _, kf_report = ps.kalman_smooth(x, causal=False)
    for report in (sg_report, kf_report):
        assert report["diagnostic_only"] is True and report["causal"] is False
        assert "reads values after" in report["leak_warning"]
        _json_safe(report)


def test_causal_kalman_noise_is_fitted_on_the_first_half_only():
    x = _canary_walk(seed=53)
    later_moved = x.copy()
    later_moved[120:] += 300.0
    _, r1 = ps.kalman_smooth(x)
    _, r2 = ps.kalman_smooth(later_moved)
    assert r1["q_ratio"] == r2["q_ratio"] and r1["obs_variance"] == r2["obs_variance"]
    assert r1["calibration_end"] == 100


def test_changepoints_and_isolation_outliers_say_they_are_diagnostic_only():
    """The canary showed both are fitted on every row passed, so a flag at time t depends on later rows."""
    import numpy as np
    rng = np.random.default_rng(0)
    y = np.concatenate([rng.normal(0, 1, 60), rng.normal(6, 1, 100)])
    _, rep = ps.changepoints(y)
    assert rep["diagnostic_only"] is True and "backtest" in rep["leak_warning"]
    _, rep2 = ps.isolation_outliers(y)
    assert rep2["diagnostic_only"] is True
    def indicator(series):
        out = np.zeros(len(series))
        for c in ps.changepoints(series)[0]:
            out[c["index"]:] += 1.0
        return out

    # Later data changes the penalty (it scales with the variance of everything passed), so the early change moves.
    assert ps.causal_canary(indicator, y, origin=100)["causal"] is False
