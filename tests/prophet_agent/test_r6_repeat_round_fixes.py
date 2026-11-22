"""Repeat round on 4b5df07 (2026-11-23): three partial scenarios, three causes, three fixes.

Every series and figure here is synthetic. Live evidence is quoted from the stored missions in data/prophet.sqlite.
"""
from __future__ import annotations

from prophet_agent.runtime import contracts
from prophet_agent.runtime.grounding import snap, snap_note


def _find(checks, name):
    return next(c for c in checks if c["name"] == name)


# 1. Cycle claim: the rewrite must be told the number to write (retail business user 50074f99).

def test_a_cycle_that_disagrees_with_health_names_the_measured_period_to_write():
    # The forecast's health record measured 6 steps; the answer says 3, the backtest's season-check figure.
    history = [{"kind": "tool", "tool": "timesfm_forecast", "args": {},
                "result": {"point": [1.0, 2.0], "health": {"period": 6, "cycle_strength": 0.48}}}]
    answer = "Sales run on a 3-step cycle, so a short low run is normal."
    checks = contracts.check_answer(answer, history)
    c = _find(checks, "answer_cycle_matches_health")
    assert c["ok"] is False
    assert "write 6 steps as the cycle" in c["detail"]
    assert "backtest's season check" in c["detail"]


def test_a_cycle_that_matches_health_still_passes():
    history = [{"kind": "tool", "tool": "timesfm_forecast", "args": {},
                "result": {"point": [1.0, 2.0], "health": {"period": 6, "cycle_strength": 0.48}}}]
    checks = contracts.check_answer("The series has a 6-step cycle.", history)
    assert _find(checks, "answer_cycle_matches_health")["ok"] is True


# 2. Figures cut down from a printed decimal (Beijing 82fb34d9: 198.7733 written as 198).

def test_a_whole_number_cut_down_from_a_printed_decimal_is_rounded_to_it():
    history = [{"kind": "tool", "tool": "timesfm_forecast", "args": {},
                "result": {"q90": [198.7733, 203.7077], "q90_b": [215.626, 215.7937], "q90_c": [191.582, 191.6463]}}]
    text = "The upper bands reach 198 and 203, then 215 and 191."
    fixed_text, fixed = snap(text, history, "")
    assert fixed_text == "The upper bands reach 199 and 204, then 216 and 192."
    assert fixed == [("198", "199"), ("203", "204"), ("215", "216"), ("191", "192")]
    assert "4 figure(s)" in snap_note(fixed)


def test_a_whole_number_is_not_changed_when_no_printed_value_rounds_to_it_uniquely():
    # 198.2 is within half a unit of 198, so the figure already matches and is not a truncation.
    history = [{"kind": "tool", "tool": "t", "args": {}, "result": {"a": 198.2, "b": 198.4}}]
    assert snap("It is 198.", history, "") == ("It is 198.", [])
    # A genuinely absent whole number, with no printed value to cut from, is left alone.
    history = [{"kind": "tool", "tool": "t", "args": {}, "result": {"a": 150.7}}]
    assert snap("It is 198.", history, "") == ("It is 198.", [])
