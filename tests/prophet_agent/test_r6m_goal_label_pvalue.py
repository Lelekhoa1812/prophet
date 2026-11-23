"""Round 6m, goal audit item 4: a measured p-value is a figure, not a significance threshold.

Before the fix, "p = 0.05" was blanked like the threshold "p < 0.05", so a reported p-value that equals a conventional level
was never checked. Only an inequality states a decision rule. Every figure here is synthetic wording.
"""
from __future__ import annotations

from prophet_agent.runtime import grounding


def test_a_measured_p_value_equal_to_a_conventional_level_is_kept():
    assert grounding.figures("The difference was p = 0.05 between arms.") == ["0.05"]
    assert grounding.figures("The gap was p = 0.01.") == ["0.01"]


def test_a_significance_threshold_written_as_an_inequality_is_still_a_label():
    assert grounding.figures("A threshold of p < 0.05 applies.") == []
    assert grounding.figures("Results at p ≤ 0.05 held.") == []
