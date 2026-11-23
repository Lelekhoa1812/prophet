"""Round 6c (2026-11-23): a same-period lag comparison is not a cycle claim.

Live retail business user (cf7f4aa6, build f84cce2): "vs. same period 260 days earlier = +150.8 %" was read as a
260-step cycle, blocked against the measured 5-step health period, and the rewrite could not clear it. Every series
here is synthetic.
"""
from __future__ import annotations

from prophet_agent.runtime import contracts, grounding


def _find(checks, name):
    return next(c for c in checks if c["name"] == name)


def test_a_year_over_year_lag_is_not_read_as_a_cycle():
    text = "Compared to the same period 260 days earlier, sales are up about 151% (p = 0.016)."
    assert grounding.claimed_periods(text) == []


def test_a_lag_with_no_same_word_is_not_a_cycle_either():
    assert grounding.claimed_periods("Sales were 12% higher 260 days earlier.") == []


def test_a_real_cycle_claim_is_still_read():
    assert [n for n, _ in grounding.claimed_periods("The series has a 5-day weekly cycle (strength 0.41).")] == [5]
    assert {n for n, _ in grounding.claimed_periods("A seasonal period of 7 steps.")} == {7}


def test_the_retail_business_answer_passes_the_cycle_check_with_a_five_step_health_record():
    history = [{"kind": "tool", "tool": "timesfm_forecast", "args": {},
                "result": {"point": [1.0, 2.0], "health": {"period": 5, "cycle_strength": 0.41}}}]
    text = ("The series has a 5-day weekly cycle (strength 0.41). A change test found +0.9% (p = 0.95) over the last "
            "20 days; compared to the same period 260 days earlier sales are up about 151% (p = 0.016).")
    assert _find(contracts.check_answer(text, history), "answer_cycle_matches_health")["ok"] is True
