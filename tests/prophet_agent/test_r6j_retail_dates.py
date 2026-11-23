"""Round 6j (2026-11-23): a daily shop that is shut on one weekday still gets its forecast dates.

Live retail e6caf00e (2026-11-23): the history was 633 trading days, open Monday to Sunday and shut on Saturdays. The
forecast's dates were null, so the answer could not state the window it covered. The date helper only knew weekends. A
history with a shut Saturday is now continued on its own rhythm, and a history that does not have one still gets no dates.
All series here are synthetic.
"""
from __future__ import annotations

from datetime import date, timedelta

import pandas as pd

from prophet_agent.forecast.calendar import next_trading_days
from prophet_agent.tools import report_tool as R


def _days(start: date, calendar_days: int, shut: set[int]) -> list[str]:
    """Consecutive calendar days from ``start``, leaving out the weekdays in ``shut`` (Monday is 0, Sunday is 6)."""
    out = []
    for i in range(calendar_days):
        d = start + timedelta(days=i)
        if d.weekday() not in shut:
            out.append(d.isoformat())
    return out


def test_a_shop_shut_on_saturdays_gets_its_next_dates_skipping_saturdays():
    hist = _days(date(2011, 1, 3), 170, {5})          # a Monday start, so the history runs through a Saturday-shut year
    ahead = R._future_dates(hist, 30)
    expected = [d.isoformat() for d in next_trading_days(date.fromisoformat(hist[-1]), 30, None, frozenset({5}))]
    assert ahead == expected
    assert len(ahead) == 30
    assert all(date.fromisoformat(d).weekday() != 5 for d in ahead)


def test_the_retail_shape_names_saturday_as_the_one_shut_day():
    hist = pd.DatetimeIndex(pd.to_datetime(_days(date(2009, 12, 1), 760, {5})))   # two years of a Saturday-shut shop
    assert R._closed_weekdays(hist) == frozenset({5})


def test_a_weekday_only_history_is_still_shut_on_both_weekend_days():
    hist = pd.DatetimeIndex(pd.to_datetime(_days(date(2025, 1, 6), 120, {5, 6})))
    assert R._closed_weekdays(hist) == frozenset({5, 6})


def test_a_history_open_every_day_names_no_shut_day():
    hist = pd.DatetimeIndex(pd.to_datetime(_days(date(2025, 1, 1), 120, set())))
    assert R._closed_weekdays(hist) is None


def test_an_intraday_history_names_no_shut_day():
    hist = pd.date_range("2025-01-01", periods=200, freq="h")
    assert R._closed_weekdays(hist) is None


def test_a_shut_day_history_with_holiday_and_shutdown_gaps_still_gets_its_dates():
    # Two shutdowns (five and twelve days) and a holiday skip the open days too. These are known gaps, so the rhythm holds.
    days = _days(date(2011, 1, 3), 170, {5})
    shut_for_a_week = days[:60] + days[65:90] + days[102:]
    ahead = R._future_dates(shut_for_a_week, 10)
    assert ahead is not None and len(ahead) == 10
    assert all(date.fromisoformat(d).weekday() != 5 for d in ahead)


def test_scattered_missing_open_days_get_no_dates():
    # Random missing days are not a calendar: more than a tenth of the gaps skip an open day, so no dates are invented.
    days = _days(date(2011, 1, 3), 170, {5})
    scattered = [d for i, d in enumerate(days) if i % 7 not in (0, 1)]
    assert R._future_dates(scattered, 5) is None


def test_one_stray_saturday_does_not_reopen_a_shut_saturday_shop():
    days = _days(date(2011, 1, 3), 170, {5})
    days = sorted(set(days) | {"2011-02-05"})         # a Saturday traded once
    hist = pd.DatetimeIndex(pd.to_datetime(days))
    assert R._closed_weekdays(hist) == frozenset({5})
