"""Round 6m, goal audit item 2: an answer that names a bound's assumption in one sentence and refers back to it in the next.

Stored mission 139db236 (read 2026-11-23): "The interval widens steadily because daily moves are treated as independent; the
total 20-day range under that assumption is roughly $1,766-$1,864 (vs. $1,815 if every day hits the centre)." The assumption is
named in the sentence before the range, and "that assumption" refers back to it. The range was blocked as unnamed.

Synthetic wording only. Nothing here calls the model or the sandbox.
"""
from __future__ import annotations

from prophet_agent.runtime import grounding

LO, HI = 1765.97, 1864.33
ANSWER_139DB236 = (
    "The interval widens steadily because daily moves are treated as independent; the total 20-day range under that "
    "assumption is roughly $1,766-$1,864 (vs. $1,815 if every day hits the centre)."
)


def test_a_range_under_that_assumption_takes_the_assumption_named_before_it():
    assert grounding.bound_misnamed(ANSWER_139DB236, LO, HI, together=False) is None


def test_a_range_under_this_assumption_takes_the_assumption_named_before_it():
    text = ("Daily moves are treated as independent. The total 20-day range under this assumption is roughly "
            "$1,766-$1,864.")
    assert grounding.bound_misnamed(text, LO, HI, together=False) is None


def test_a_range_under_that_assumption_is_still_blocked_when_the_sentence_before_names_no_assumption():
    text = ("The interval widens steadily as the horizon grows. The total 20-day range under that assumption is roughly "
            "$1,766-$1,864.")
    assert grounding.bound_misnamed(text, LO, HI, together=False) is not None


def test_an_anaphor_takes_only_the_assumption_of_the_sentence_directly_before_it():
    # The independence is two sentences back; the sentence before the range names nothing, so the range stays unnamed.
    text = ("Daily moves are treated as independent. The interval widens steadily as the horizon grows. The total 20-day "
            "range under that assumption is roughly $1,766-$1,864.")
    assert grounding.bound_misnamed(text, LO, HI, together=False) is not None


def test_a_bare_range_with_no_assumption_anywhere_is_still_blocked():
    text = "The total 20-day range is roughly $1,766-$1,864."
    assert grounding.bound_misnamed(text, LO, HI, together=False) is not None


def test_the_together_bound_is_named_by_the_sentence_before_an_anaphor_too():
    text = ("All days miss the same way at once. The total 20-day range under that assumption is roughly "
            "$1,766-$1,864.")
    assert grounding.bound_misnamed(text, LO, HI, together=True) is None
