"""Round 6l, goal audit item 2: the repair pass re-checks the answer's claims on the text it returns.

The claims were checked before the repair rewrote unsupported figures. A rewrite can change a claim the stored output supports
(here, the cycle), and the answer then left as "met" without a second check. The claim checker is replaced by a stub that
blocks a 2-step cycle, because the real cycle check needs a full forecast record and its own ledger.

Synthetic figures only.
"""
from __future__ import annotations

import pytest

from prophet_agent.runtime import contracts
from prophet_agent.runtime.goal import GoalConfig, goal_program
from prophet_agent.runtime.store import Store

from tests.prophet_agent.test_goal_loop import WritingBrain, call, finish, met, run, _tool


@pytest.fixture()
def store(tmp_path):
    s = Store(tmp_path / "repair_claims.sqlite")
    yield s
    s.close()


@pytest.fixture()
def cycle_check(monkeypatch):
    def fake(answer, history, *, run_root=None, goal=""):
        ok = "2-step" not in answer
        return [{"name": "answer_cycle_matches_health", "ok": ok, "severity": "block",
                 "detail": "the stored cycle is 6 steps" if not ok else "the cycle matches"}]
    monkeypatch.setattr(contracts, "check_answer", fake)


RESULT = {"point": [1.0, 2.0], "n": 6}
ANSWER = "The series has a 6-step cycle, and sales reached 999 units."


def test_a_repair_that_breaks_a_checked_claim_is_not_returned_as_met(store, cycle_check):
    # The first write is the composed answer (figure 999 is not in the evidence); the second is the repair, which now says 2.
    brain = WritingBrain(plans=[call("cycle_probe"), finish()], evals=[met(ANSWER)],
                         writes=[ANSWER, "The series has a 2-step cycle."])
    program = goal_program("cycle", {"cycle_probe": _tool("cycle_probe", lambda args, key: RESULT)}, brain,
                           GoalConfig(max_iterations=6, tool_budget=1))
    _, out = run(store, program)
    result = out["result"]
    assert result["status"] == "partial", result
    assert result["contract_failures"] == ["answer_cycle_matches_health"]
    assert result["answer"] == "The series has a 2-step cycle."


def test_a_repair_that_keeps_the_checked_claims_is_still_met(store, cycle_check):
    brain = WritingBrain(plans=[call("cycle_probe"), finish()], evals=[met(ANSWER)],
                         writes=[ANSWER, "The series has a 6-step cycle."])
    program = goal_program("cycle", {"cycle_probe": _tool("cycle_probe", lambda args, key: RESULT)}, brain,
                           GoalConfig(max_iterations=6, tool_budget=1))
    _, out = run(store, program)
    assert out["result"]["status"] == "met", out["result"]
    assert out["result"]["answer"] == "The series has a 6-step cycle."
