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
