"""R6m, audit item 2: infer_calendar must read the same calendar whatever type the dates arrive as.

A probe (synthetic) gave a TypeError for a weekday series of datetime objects, which is what pandas timestamps are. The
cause: a datetime passes isinstance(d, date), so it was kept as a datetime and compared with the date holidays. The
trading days below are the NYSE weekdays of 2025, so the calendar it follows is NYSE whatever the input type.
"""
from __future__ import annotations

import datetime as dt

import numpy as np

from prophet_agent.forecast.calendar import infer_calendar


def _nyse_trading_days_2025():
    import holidays
    closed = set(holidays.financial_holidays("NYSE", years=[2025]))
    days = []
    d = dt.date(2025, 1, 1)
    while d.year == 2025:
        if d.weekday() < 5 and d not in closed:
            days.append(d)
        d += dt.timedelta(days=1)
    return days


def test_the_calendar_is_read_from_plain_dates():
    assert infer_calendar(_nyse_trading_days_2025())["calendar"] == "NYSE"


def test_the_same_calendar_is_read_from_datetimes():
    days = _nyse_trading_days_2025()
    out = infer_calendar([dt.datetime(d.year, d.month, d.day) for d in days])
    assert out["calendar"] == "NYSE"
    assert out["absent_share"] == infer_calendar(days)["absent_share"]


def test_the_same_calendar_is_read_from_numpy_dates_and_iso_strings():
    days = _nyse_trading_days_2025()
    assert infer_calendar(np.array(days, dtype="datetime64[D]"))["calendar"] == "NYSE"
    assert infer_calendar([d.isoformat() for d in days])["calendar"] == "NYSE"
