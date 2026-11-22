"""Round 4 finding A4-10: a ledger entry can carry the hypothesis it tested.

Motivation vs Logic: none of the six audited ledgers recorded a falsifiable expectation, so the method's "hypotheses, not
habits" step was invisible. ``Ledger.add`` now takes ``expected`` (what the analyst predicted before looking) and
``outcome`` (confirmed, refuted or not tested). Both are optional, so entries without them keep their old shape and stay
valid. An outcome needs an expectation to test against.
"""
from __future__ import annotations

import pytest

from prophet_agent.sandbox import prophet_stats as ps


def test_an_entry_with_expected_and_outcome_keeps_both():
    led = ps.Ledger()
    led.add("shift", "level moved in 2024", "keep the break", expected="the level shifted in 2024", outcome="confirmed")
    entry = led.to_dict()[0]
    assert entry["expected"] == "the level shifted in 2024"
    assert entry["outcome"] == "confirmed"


def test_an_entry_without_them_has_no_such_keys_so_old_ledgers_are_unchanged():
    led = ps.Ledger()
    led.add("stationarity", "trend found", "difference once")
    entry = led.to_dict()[0]
    assert "expected" not in entry and "outcome" not in entry


@pytest.mark.parametrize("outcome", ["done", "true", "", "Confirmed "])
def test_an_outcome_outside_the_three_words_is_refused(outcome):
    led = ps.Ledger()
    with pytest.raises(ValueError):
        led.add("weekly", "cycle", "keep", expected="a weekly cycle", outcome=outcome)


@pytest.mark.parametrize("outcome", ["confirmed", "refuted", "not tested"])
def test_each_of_the_three_outcomes_is_accepted(outcome):
    led = ps.Ledger()
    led.add("weekly", "cycle", "keep", expected="a weekly cycle", outcome=outcome)
    assert led.to_dict()[0]["outcome"] == outcome


def test_an_outcome_without_an_expectation_is_refused():
    led = ps.Ledger()
    with pytest.raises(ValueError):
        led.add("weekly", "cycle", "keep", outcome="confirmed")


def test_a_blank_expectation_is_refused():
    led = ps.Ledger()
    with pytest.raises(ValueError):
        led.add("weekly", "cycle", "keep", expected="   ")


def test_expected_and_outcome_survive_save_and_open_without_shrinking(tmp_path):
    path = str(tmp_path / "ledger.json")
    led = ps.Ledger()
    led.add("a", "f", "d", expected="x", outcome="refuted")
    led.save(path)
    again = ps.Ledger.open(path)
    assert again.to_dict()[0]["outcome"] == "refuted"
    again.add("b", "g", "e")
    again.save(path)
    assert len(ps.Ledger.open(path).to_dict()) == 2
