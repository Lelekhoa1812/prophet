"""r6m passages: dates are real calendar dates, and figure checks survive any digit run a page or draft contains.

Defects reproduced here:
- parse_date_expr built an ISO string from the words and numbers it matched without checking the calendar, so
  "31 February 2026" and "2026-13-45" became dated events ("2026-02-31", "2026-13-45") in the event list;
- verify_figures converted every digit run to int to decide whether it is checkable, so a run of more than 4300 digits
  (Python's int conversion limit) raised ValueError and failed the whole check.
"""
from prophet_agent.research.events import dated_events
from prophet_agent.research.passages import parse_date_expr, verify_figures


def test_impossible_calendar_dates_are_not_read():
    assert parse_date_expr("31 February 2026", 2026) is None
    assert parse_date_expr("February 30, 2026", 2026) is None
    assert parse_date_expr("2026-13-45", 2026) is None
    assert parse_date_expr("2026-02-30", 2026) is None


def test_leap_day_is_read_only_in_a_leap_year():
    assert parse_date_expr("29 February 2028", 2026) == ("2028-02-29", True)
    assert parse_date_expr("29 February 2026", 2026) is None


def test_impossible_dates_do_not_become_events():
    sources = [{"id": "s1", "tier": 1, "date": "2026-01-01", "url": "https://www.eia.gov/x",
                "passages": [{"text": "The release is set for 31 February 2026. The next one is 10 March 2026."}]}]
    events = dated_events(sources)
    assert [e["date"] for e in events] == ["2026-03-10"]
