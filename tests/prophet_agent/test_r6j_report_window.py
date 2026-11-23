"""Round 6j (2026-11-23): the forecast report names the period its forecast covers.

Live retail e6caf00e rep9: the forecast carried its dates and the table had a date column, but no line of the observation
named the window, so the answer gave the total without its period. The window is now a field of the observation.
All series here are synthetic.
"""
from prophet_agent.tools import report_tool as R


def test_a_forecast_with_dates_gets_its_window():
    fc = {"future_dates": ["2011-12-09", "2011-12-10", "2011-12-12"]}
    assert R._window(fc) == {"first": "2011-12-09", "last": "2011-12-12", "steps": 3}


def test_a_forecast_without_dates_gets_no_window():
    assert R._window({"point": [1.0, 2.0]}) is None
    assert R._window({"future_dates": None}) is None
    assert R._window({"future_dates": []}) is None
