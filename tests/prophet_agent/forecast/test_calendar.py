"""The trading calendar is read from the data: a NYSE-style history skips Thanksgiving in its forecast dates."""
from datetime import date

import pandas as pd

from prophet_agent.forecast.calendar import infer_calendar, next_trading_days


def _nyse_days(start: str, end: str) -> list[date]:
    import holidays
    days = pd.bdate_range(start, end)
    hols = holidays.financial_holidays("NYSE", years=range(days[0].year, days[-1].year + 1))
    return [d.date() for d in days if d.date() not in hols]


def test_a_history_that_skips_nyse_holidays_is_read_as_nyse_and_skips_thanksgiving():
    cal = infer_calendar(_nyse_days("2021-01-04", "2026-11-17"))
    assert cal["calendar"] == "NYSE" and cal["absent_share"] >= 0.95
    ahead = next_trading_days(date(2026, 11, 17), 10, cal["calendar"])
    assert date(2026, 11, 26) not in ahead and ahead[0] == date(2026, 11, 18) and len(ahead) == 10


def test_plain_weekdays_when_no_calendar_is_followed():
    days = [d.date() for d in pd.bdate_range("2022-01-03", "2026-11-17")]   # every weekday present: no holidays skipped
    cal = infer_calendar(days)
    assert cal["calendar"] is None
    assert date(2026, 11, 26) in next_trading_days(date(2026, 11, 17), 10, None)


def test_the_report_dates_follow_the_series_calendar():
    from prophet_agent.tools.report_tool import _future_dates
    hist = [d.isoformat() for d in _nyse_days("2021-01-04", "2026-11-17")]
    ahead = _future_dates(hist, 10)
    assert "2026-11-26" not in ahead and ahead[0] == "2026-11-18"


def test_a_simple_rule_can_be_the_centre_line_with_timesfms_band_recentred():
    from prophet_agent.tools.report_tool import _centre, _method_note
    fc = {"point": [10.0, 11.0], "quantiles": {"0.1": [9.0, 9.5], "0.5": [10.0, 11.0], "0.9": [11.0, 12.5]}, "config": {}}
    out = _centre(fc, [5.0, 8.0, 12.0], "naive")
    assert out["point"] == [12.0, 12.0] and out["timesfm_point"] == [10.0, 11.0]
    assert out["quantiles"]["0.1"] == [11.0, 10.5] and out["quantiles"]["0.9"] == [13.0, 13.5]   # widths kept
    assert _centre(fc, [5.0, 8.0, 12.0], "drift")["point"] == [15.5, 19.0]
    assert "more accurate than TimesFM" in _method_note({}, out["centre"])


def test_the_pack_states_how_far_the_band_reaches_from_the_centre():
    from prophet_agent.tools.report_tool import range_vs_centre
    fc = {"point": [100.0, 100.0, 174.35], "quantiles": {"0.1": [99.2, 98, 168.3], "0.9": [100.8, 102, 180.0]}}
    r = range_vs_centre(fc)
    assert r["first_step"] == {"below_pct": -0.8, "above_pct": 0.8}
    assert r["last_step"] == {"below_pct": -3.5, "above_pct": 3.2}
    assert range_vs_centre({"point": [0.0], "quantiles": {"0.1": [-1], "0.9": [1]}}) is None
