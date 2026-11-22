"""Round 3 clock and calendar (A1-01, A1-03, A1-11, A4-13): dates are never re-dated by position, closed days are not
invented, sub-daily repeated hours are not summed silently, and the saved series carries its clock.

Motivation vs Logic: the live retail run (rep4) placed 504 values by position on a weekday grid and forecast 2011-11-07
to 2011-11-24 while the data ran to 2011-12-08 (A1-01). Synthetic fixtures below are labelled as such; they are not
live results.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from prophet_agent.sandbox import prophet_lib as pl


def _rows(dates, values):
    return pd.DataFrame({"d": list(dates), "v": [str(v) for v in values]})


# A1-01: a business-day grid must not fold weekend sales into Friday.
def test_weekend_rows_are_refused_on_a_business_day_grid():
    # Synthetic: Fri 100, Sat 10, Sun 20, Mon 30 (the repro in the A1-01 finding gave Fri 130 with no warning).
    df = _rows(["2024-01-05", "2024-01-06", "2024-01-07", "2024-01-08"], [100, 10, 20, 30])
    with pytest.raises(ValueError, match="Saturday") as err:
        pl.build_series(df, "v", "d", freq="B")
    assert "freq='D'" in str(err.value)


def test_a_daily_grid_keeps_every_observed_date_apart():
    df = _rows(["2024-01-05", "2024-01-06", "2024-01-07", "2024-01-08"], [100, 10, 20, 30])
    s = pl.build_series(df, "v", "d", freq="D")
    assert list(s) == [100.0, 10.0, 20.0, 30.0]
    assert s.index[-1] == pd.Timestamp("2024-01-08") and s.index[0] == pd.Timestamp("2024-01-05")


def test_prepare_series_refuses_weekend_rows_on_a_business_day_grid():
    df = _rows(["2024-01-05", "2024-01-06", "2024-01-07", "2024-01-08"], [100, 10, 20, 30])
    with pytest.raises(ValueError, match="freq='D'"):
        pl.prepare_series(df, "v", "d", freq="B")


def test_a_series_ends_on_its_last_observed_date():
    # Synthetic daily data with Sunday trading; the last observed row is a Thursday.
    dates = pd.date_range("2011-11-27", "2011-12-08", freq="D")
    df = _rows(dates.strftime("%Y-%m-%d"), np.arange(len(dates)) + 1.0)
    s = pl.build_series(df, "v", "d", freq="D")
    assert s.index[-1] == pd.Timestamp("2011-12-08") and len(s) == len(dates)


# A4-13: the clock is saved with the series, and an old file without it still loads.
def test_save_series_writes_step_and_calendar_kind(tmp_path):
    bdays = pd.bdate_range("2024-01-01", periods=10)
    path = str(tmp_path / "b.json")
    pl.save_series(pd.Series(np.arange(10.0), index=bdays), path)
    obj = json.loads(Path(path).read_text(encoding="utf-8"))
    assert obj["step"] == "B" and obj["calendar"] == "business"
    assert obj["series"] == [float(x) for x in range(10)] and len(obj["dates"]) == 10     # backward compatible shape

    days = pd.date_range("2024-01-01", periods=10, freq="D")
    pl.save_series(pd.Series(np.arange(10.0), index=days), path)
    obj = json.loads(Path(path).read_text(encoding="utf-8"))
    assert obj["step"] == "D" and obj["calendar"] == "calendar"


def test_load_series_still_reads_a_file_without_clock_keys(tmp_path):
    path = tmp_path / "old.json"
    path.write_text(json.dumps({"series": [1.0, 2.0], "dates": ["2024-01-01", "2024-01-02"]}), encoding="utf-8")
    back = pl.load_series(str(path))
    assert list(back.values) == [1.0, 2.0]


# A1-11: a Sydney-style fall-back day (02:00 happens twice) is neither summed nor dropped.
_FALL_BACK = pd.DataFrame({"t": ["2024-04-07 01:00:00+11:00", "2024-04-07 02:00:00+11:00",
                                 "2024-04-07 02:00:00+10:00", "2024-04-07 03:00:00+10:00"],
                           "v": ["10", "10", "10", "10"]})


def test_fall_back_hour_with_offsets_is_four_distinct_hours():
    s = pl.build_series(_FALL_BACK, "v", "t", freq="h")
    assert list(s) == [10.0, 10.0, 10.0, 10.0]
    assert list(pd.Series(s.index).diff().dropna()) == [pd.Timedelta(hours=1)] * 3   # one hour apart on a UTC clock


def test_prepare_series_keeps_the_fall_back_hours_apart():
    s, rep = pl.prepare_series(_FALL_BACK, "v", "t")
    assert len(s) == 4 and s.sum() == pytest.approx(40.0) and rep["duplicate_dates"] == 0


def test_repeated_wall_clock_hour_without_offsets_is_refused_unless_agg_is_named():
    df = pd.DataFrame({"t": ["2024-04-07 01:00", "2024-04-07 02:00", "2024-04-07 02:00", "2024-04-07 03:00"],
                       "v": ["10", "10", "10", "10"]})
    with pytest.raises(ValueError, match="share a timestamp"):
        pl.build_series(df, "v", "t", freq="h")
    s = pl.build_series(df, "v", "t", freq="h", agg="sum")       # named explicitly: the caller has chosen
    assert list(s) == [10.0, 20.0, 10.0]


# A1-03: structurally closed days are not interpolated and add nothing to the total.
def _saturday_closed(n=120, value=100.0):
    idx = pd.date_range("2024-01-01", periods=n, freq="D")       # synthetic: Saturdays never trade
    keep = idx.day_of_week != 5
    return idx, keep, _rows(idx[keep].strftime("%Y-%m-%d"), [value] * int(keep.sum()))


def test_a_weekday_pattern_without_closed_is_refused_with_the_fix():
    _, _, df = _saturday_closed()
    with pytest.raises(ValueError, match="closed=\\['Saturday'\\]"):
        pl.prepare_series(df, "v", "d")


def test_closed_saturdays_stay_missing_and_invent_no_value():
    idx, keep, df = _saturday_closed()
    s, rep = pl.prepare_series(df, "v", "d", closed=["Saturday"])
    sat = s.index.day_of_week == 5
    assert s[sat].isna().all()                                    # closed days stay NaN, not interpolated
    assert s.dropna().sum() == pytest.approx(100.0 * int(s.notna().sum()))     # every kept value is observed, none invented
    assert rep["filled_sum_share"] == 0
    assert rep["closed_periods"] == int(sat.sum())


def test_a_real_gap_on_an_open_day_is_still_interpolated():
    idx, keep, df = _saturday_closed(n=60)
    tuesdays = idx[(idx.day_of_week == 1)]
    df = df[df["d"] != tuesdays[3].strftime("%Y-%m-%d")]          # one Tuesday missing: a genuine gap
    s, rep = pl.prepare_series(df, "v", "d", closed=["Saturday"])
    assert rep["filled_periods"] == 1 and rep["filled_sum_share"] > 0
    assert s[tuesdays[3]] == pytest.approx(100.0)


# A4-13 and A3-05: holidays given as dates are closed days, and say so in the report.
def test_closed_dates_are_treated_like_closed_weekdays():
    idx = pd.bdate_range("2024-01-01", periods=30)               # synthetic trading calendar
    holiday = idx[10]
    df = _rows([d.strftime("%Y-%m-%d") for d in idx if d != holiday], [50.0] * 29)
    s, rep = pl.prepare_series(df, "v", "d", closed=[holiday.strftime("%Y-%m-%d")])
    assert pd.isna(s[holiday]) and rep["filled_sum_share"] == 0 and rep["closed_periods"] == 1


def test_closed_days_dropped_then_saved_record_the_trading_calendar(tmp_path):
    # Synthetic: Saturdays never trade. The closed days are dropped with dropna(); the dates of the rest survive the save.
    idx, keep, df = _saturday_closed(n=28)
    s, rep = pl.prepare_series(df, "v", "d", closed=["Saturday"])
    path = str(tmp_path / "trading.json")
    pl.save_series(s.dropna(), path)
    obj = json.loads(Path(path).read_text(encoding="utf-8"))
    assert obj["calendar"] == "trading" and obj["step"] == "D"
    assert len(obj["series"]) == len(obj["dates"]) == int(keep.sum())
    assert all(pd.Timestamp(d).day_of_week != 5 for d in obj["dates"])
