"""Round 6k (2026-11-23): repeated timestamps do not change the date kind of a series.

Audit (forecast, season.py:52-54): a median over gaps that include zeros read a daily series with each date repeated as
hourly. The gaps are now measured between distinct dates. All series here are synthetic.
"""
import numpy as np

from prophet_agent.forecast import season as S


def _days(n: int) -> np.ndarray:
    return np.datetime64("2011-01-03") + np.arange(n)


def test_a_daily_series_with_repeated_dates_is_still_daily():
    assert S._kind_from_dates(np.repeat(_days(400), 2)) == "daily"


def test_a_daily_series_with_no_repeats_is_daily():
    assert S._kind_from_dates(_days(400)) == "daily"


def test_a_business_series_with_repeated_dates_is_still_business():
    d = _days(400)
    weekdays = d[((d.astype("datetime64[D]").astype(np.int64) + 3) % 7) < 5]
    assert S._kind_from_dates(np.repeat(weekdays, 2)) == "business"


def test_a_single_repeated_weekend_date_gives_no_kind():
    # A Saturday, so the business check does not apply and the gap path is taken: one distinct date has no gap.
    assert S._kind_from_dates(np.array(["2011-01-08"] * 5, dtype="datetime64[s]")) is None
