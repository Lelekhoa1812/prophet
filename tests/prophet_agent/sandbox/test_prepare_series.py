"""prepare_series: the step is inferred, gaps are never zero, edges are trimmed, long gaps are marked for the forecast to skip."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from prophet_agent.sandbox import prophet_lib as pl


def _frame(dates, values):
    return pd.DataFrame({"d": [x.strftime("%Y-%m-%d") for x in dates], "v": ["" if v != v else str(v) for v in values]})


def test_calendar_daily_with_holes_is_completed_and_gaps_are_interpolated_not_zeroed():
    idx = pd.date_range("2024-01-01", periods=200, freq="D")
    vals = np.arange(200, dtype=float)
    keep = np.ones(200, bool)
    keep[50:53] = False
    s, rep = pl.prepare_series(_frame(idx[keep], vals[keep]), "v", "d")
    assert rep["freq"] == "D" and len(s) == 200 and rep["filled_periods"] == 3
    assert s.iloc[51] == pytest.approx(51.0)                       # straight line through 49 and 53
    assert rep["ignore_ranges"] == []                              # three periods is a short gap


def test_business_days_are_recognised_by_the_missing_weekends():
    idx = pd.bdate_range("2024-01-01", periods=120)
    s, rep = pl.prepare_series(_frame(idx, np.arange(120.0)), "v", "d")
    assert rep["freq"] == "B" and len(s) == 120 and rep["filled_periods"] == 0


def test_a_long_interior_gap_is_filled_and_listed_for_the_forecast_to_ignore():
    idx = pd.date_range("2023-01-01", periods=400, freq="D")
    vals = np.sin(np.arange(400) / 20) + 5
    vals[100:160] = np.nan
    s, rep = pl.prepare_series(_frame(idx, vals), "v", "d")
    assert rep["ignore_ranges"] == [[100, 160]] and not s.isna().any()
    assert rep["long_gaps"][0]["periods"] == 60


def test_empty_edges_are_cut_off_and_reported():
    idx = pd.date_range("2024-01-01", periods=100, freq="D")
    vals = np.arange(100, dtype=float)
    vals[:5] = np.nan
    vals[-9:] = np.nan
    s, rep = pl.prepare_series(_frame(idx, vals), "v", "d")
    assert rep["dropped_leading"] == 5 and rep["dropped_tail"] == 9 and len(s) == 86
    assert s.index[-1] == pd.Timestamp("2024-04-09") - pd.Timedelta(days=9)


def test_duplicates_are_combined_and_reported():
    idx = list(pd.date_range("2024-01-01", periods=60, freq="D"))
    idx += [idx[10]]
    s, rep = pl.prepare_series(_frame(idx, [1.0] * 61), "v", "d")
    assert rep["duplicate_dates"] == 1 and s.iloc[10] == 2.0


def test_a_carried_forward_tail_is_flagged_but_a_floor_of_zeros_is_not():
    idx = pd.date_range("2024-01-01", periods=80, freq="D")
    carried = np.r_[np.random.default_rng(1).normal(50, 5, 60), np.full(20, 47.0)]
    _, rep = pl.prepare_series(pd.Series(carried, index=idx))
    assert rep["constant_tail"] == 20
    floor = np.r_[np.random.default_rng(1).poisson(3, 60).astype(float) + 1, np.zeros(20)]
    _, rep2 = pl.prepare_series(pd.Series(floor, index=idx))
    assert rep2["constant_tail"] == 0


def test_irregular_dates_ask_for_a_step_instead_of_guessing():
    idx = pd.to_datetime(["2024-01-01", "2024-01-03", "2024-02-11", "2024-02-12", "2024-05-30"])
    with pytest.raises(ValueError, match="freq"):
        pl.prepare_series(pd.Series([1.0, 2, 3, 4, 5], index=idx))


def test_the_report_names_the_repeating_cycle_and_how_many_times_the_history_holds_it():
    idx = pd.date_range("2022-01-01", periods=730, freq="D")
    weekly = 100 + 20 * np.sin(2 * np.pi * np.arange(730) / 7) + np.random.default_rng(3).normal(0, 2, 730)
    _, rep = pl.prepare_series(pd.Series(weekly, index=idx))
    assert rep["period"] == 7 and rep["cycle_strength"] > 0.5 and rep["cycles_in_history"] == pytest.approx(104.3, abs=0.1)
    walk = np.cumsum(np.random.default_rng(4).normal(0, 1, 730))
    _, rep2 = pl.prepare_series(pd.Series(walk, index=idx))
    assert rep2["period"] is None and rep2["cycle_strength"] is None


def test_a_history_with_under_three_cycles_says_so():
    idx = pd.date_range("2023-01-01", periods=400, freq="D")
    annual = 10 * np.sin(2 * np.pi * np.arange(400) / 365) + np.random.default_rng(5).normal(0, 0.3, 400)
    _, rep = pl.prepare_series(pd.Series(annual, index=idx))
    assert any("fewer than three" in n for n in rep["notes"]) or rep["period"] is None
