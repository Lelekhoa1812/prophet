"""Round 6m, goal audit item 4: a measured percentage range is not an interval level without its band word.

Synthetic wording only.
"""
from __future__ import annotations

from prophet_agent.runtime import grounding


# Hidden before the fix: any "10-90%" pair was a band label, so a measured percentage range vanished.
def test_a_measured_percentage_range_with_no_band_word_is_kept():
    assert grounding.figures("Demand rose 10-90% across stores.") == ["10", "90%"]
