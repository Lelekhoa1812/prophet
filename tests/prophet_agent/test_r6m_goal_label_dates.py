"""Round 6m, goal audit item 4: an ISO-looking measured range is not a date label.

Synthetic wording only.
"""
from __future__ import annotations

from prophet_agent.runtime import grounding


# Hidden before the fix: "1000-15" matched the year-month date pattern, so both figures were blanked.
def test_a_measured_range_is_not_read_as_a_year_and_month():
    assert grounding.figures("The range was 1000-15 units.") == ["1000", "15"]
