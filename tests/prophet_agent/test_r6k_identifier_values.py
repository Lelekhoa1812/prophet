"""Round 6k (2026-11-23): a measured count after a label noun is a figure, not an identifier.

Audit (runtime/grounding.py:88-89, verified): the identifier pattern blanked "Customers 1,200" whole, so a wrong measured
count was never checked against the evidence. A list is comma-then-space; a thousands group or a decimal is a value.
All figures here are synthetic.
"""
import pytest

from prophet_agent.runtime.grounding import _drop_labels


@pytest.mark.parametrize("text, kept", [
    ("Customers 1,200 bought in the period.", "1,200"),
    ("Orders 3,000 units shipped.", "3,000"),
    ("Items 2,400 sold.", "2,400"),
    ("Store 3.5 was the best.", "3.5"),
])
def test_a_measured_count_after_a_label_noun_is_kept_for_checking(text, kept):
    assert kept in _drop_labels(text)


@pytest.mark.parametrize("text, gone", [
    ("Week 12 was strong.", "12"),
    ("Days 3, 10, 17, 24 were low.", "24"),
    ("Customer IDs 41-48 churned.", "41-48"),
    ("Rows 60 through 66 matched.", "66"),
])
def test_a_label_number_is_still_blanked(text, gone):
    assert gone not in _drop_labels(text)
