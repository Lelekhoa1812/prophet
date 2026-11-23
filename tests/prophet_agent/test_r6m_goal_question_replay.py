"""Round 6m, goal audit item 2: a stored answer is not replayed for a question whose bound, date, horizon or unit changed.

Before the fix, _already_asked treated a question as asked before when its wording was about 75% alike, or when its header
matched, whatever the percentages and time units in it. So "the 90% interval?" was dropped as "not asked again" and the person's
answer to "the 80% interval?" stood in for it. A question about another unit did the same. A changed level or unit is a new
question. Dates are not covered here (live run 04 is the same decision with a moved date). Every question here is synthetic.
"""
from __future__ import annotations

from prophet_agent.runtime.goal import _already_asked


def _history(question, header):
    return [{"kind": "ask_user", "answers": {header: "x"}, "questions": [{"question": question, "header": header}]}]


def _replayed(new_question, new_header, old_question, old_header):
    _, repeated = _already_asked([{"question": new_question, "header": new_header}],
                                 _history(old_question, old_header))
    return bool(repeated)


def test_a_question_about_a_different_interval_level_is_not_replayed():
    assert _replayed("Should the total range use the 90% interval?", "Interval",
                     "Should the total range use the 80% interval?", "Interval") is False


def test_a_question_about_a_different_range_is_not_replayed_even_when_the_wording_is_close():
    assert _replayed("Which bound should the total range state: the 90% range or the 95% range?", "Range",
                     "Which bound should the total range state: the 80% range or the 90% range?", "Range") is False


def test_a_question_about_a_different_unit_is_not_replayed_under_the_same_header():
    assert _replayed("What horizon in days should the forecast cover?", "Horizon",
                     "What horizon in weeks should the forecast cover?", "Horizon") is False


def test_the_same_question_worded_the_same_is_still_replayed():
    assert _replayed("Should the 2025-06-03 spike be ignored as a data error?", "Spike",
                     "Should the 2025-06-03 spike be ignored as a data error?", "Spike") is True


def test_a_rephrase_that_keeps_the_same_figures_and_units_is_still_replayed():
    assert _replayed("Should the 2025-06-03 spike be treated as a data error?", "Spike",
                     "Should the 2025-06-03 spike be ignored as a data error?", "Spike") is True
