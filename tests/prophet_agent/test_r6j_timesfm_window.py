"""Round 6j (2026-11-23): the raw TimesFM forecast says which calendar period it covers.

Live retail rep8 (build 008917e): the planner answered from timesfm_forecast, whose output had no dates, so the answer listed
twenty values with no window. The result now carries the forecast's dates and a window (first, last, steps) when the history's
dates are known and match its length. All series here are synthetic.
"""
from __future__ import annotations

import importlib
from datetime import date, timedelta

# The package exports a function named timesfm_tool, so the module is imported by its full name.
T = importlib.import_module("prophet_agent.tools.timesfm_tool")


def _shop_days(start: date, calendar_days: int, shut: set[int]) -> list[str]:
    out = []
    for i in range(calendar_days):
        d = start + timedelta(days=i)
        if d.weekday() not in shut:
            out.append(d.isoformat())
    return out


def _result(horizon: int) -> dict:
    return {"point": [1.0] * horizon, "quantiles": {"0.1": [0.5] * horizon, "0.9": [1.5] * horizon}, "config": {}}


def test_a_history_with_known_dates_gets_its_forecast_window():
    hist = _shop_days(date(2011, 1, 3), 170, {5})
    r = _result(20)
    T._attach_window(r, hist, len(hist), 20)
    assert r["window"]["steps"] == 20
    assert r["window"]["first"] == r["future_dates"][0] and r["window"]["last"] == r["future_dates"][-1]
    assert all(date.fromisoformat(d).weekday() != 5 for d in r["future_dates"])


def test_no_window_when_the_dates_do_not_match_the_history_length():
    hist = _shop_days(date(2011, 1, 3), 170, {5})
    r = _result(20)
    T._attach_window(r, hist, len(hist) + 1, 20)
    assert "window" not in r and "future_dates" not in r


def test_no_window_without_dates():
    r = _result(3)
    T._attach_window(r, None, 5, 3)
    assert "window" not in r


def test_the_compact_view_the_planner_reads_shows_the_window():
    r = _result(3)
    r["window"] = {"first": "2011-12-09", "last": "2011-12-13", "steps": 3}
    out = T._compact(r)
    assert out["window"] == {"first": "2011-12-09", "last": "2011-12-13", "steps": 3}


def test_the_compact_view_has_no_window_key_when_there_is_none():
    assert "window" not in T._compact(_result(3))
