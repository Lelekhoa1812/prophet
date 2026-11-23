"""Round 6m, goal audit item 4: a month followed by a count that is not a day is not a date label.

Synthetic wording only.
"""
from __future__ import annotations

from prophet_agent.runtime import grounding


# Hidden before the fix: "May 40" matched the month-then-day label, so a count of 40 cases was blanked.
def test_a_month_followed_by_a_number_that_is_not_a_day_keeps_the_number():
    assert grounding.figures("Sales in May 40 cases were recorded.") == ["40"]
