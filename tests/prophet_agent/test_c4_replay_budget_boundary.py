"""Round 4 C4, audit A5-07: a part that ends at the tool-allowance edge must replay the same branch after a top-up.

Motivation vs Logic
Motivation: the allowance decides whether the loop evaluates at a step (``out_of_calls``), but that value was recomputed on
each run. A part that ran out of calls at step 2 stored an evaluation at step 2. After a top-up the recomputed value was
False, the loop skipped the evaluation, asked for step 3, and found the stored ``eval:2`` there: "replay diverged".
Logic: the planner's record stores the allowance decision it was made under (``out_of_calls``), and the evaluation gate reads
that stored value. The checkpoint at the end of a part is decided by the allowance now, so a topped-up part goes on. The
repair allowance is derived from the stored history (one entry per grant), not from an in-place change to the config.

The test runs a mission to a checkpoint at a spent allowance, then resumes it after a top-up with a brain that is asked only
for the new decisions. A replay that takes another branch fails with "replay diverged".
"""
from __future__ import annotations

import pytest

from prophet_agent.runtime.goal import GoalConfig, goal_program
from prophet_agent.runtime.runner import Runtime, Tool
from prophet_agent.runtime.store import Store

GOAL = "forecast the daily sales for the next 30 days"   # a forecast goal: the premature-finish gate applies


@pytest.fixture()
def store(tmp_path):
    s = Store(tmp_path / "boundary.sqlite")
    yield s
    s.close()


class Queue:
    """Answers each decision from its own queue; a decision that is not scripted fails the test."""

    def __init__(self, plans, evals):
        self.plans, self.evals = list(plans), list(evals)
        self.asked = []

    def decide(self, messages, schema, name):
        self.asked.append(name)
        queue = self.plans if name == "plan" else self.evals
        if not queue:
            raise AssertionError(f"the brain was asked for {name} but no decision is scripted")
        return queue.pop(0)


def _call(k):
    return {"action": "call_tool", "tool": "a", "args": {"k": k}, "rationale": "work"}


def test_resume_across_a_topup_takes_the_same_branch_and_finishes(store, monkeypatch):
    # The scripted path ends at the third rejected finish. This test is about replay across a top-up, so it pins the
    # rejection limits it was written for; the pivot and the later write-up have their own tests.
    import prophet_agent.runtime.goal as goal_mod
    monkeypatch.setattr(goal_mod, "MAX_REJECTED_FINISHES", 3)
    monkeypatch.setattr(goal_mod, "PIVOT_AFTER_REJECTIONS", 99)
    counter = {"a": 0}

    def a(args, key):
        counter["a"] += 1
        return {"ok": True, "k": args["k"]}

    tools = {"a": Tool("a", a, idempotent=True)}
    mid = store.create_mission(GOAL, {"max_tool_calls": 2, "max_seconds": 900})

    # Part 1: two calls spend the allowance. At the spent allowance the planner is forced to finish, and the finish is
    # evaluated (the evaluation is stored at step 6), which ends the part in a checkpoint.
    first_brain = Queue(plans=[_call(1), _call(2), _call(1)],
                        evals=[{"verdict": "not_met", "reason": "more work is needed"}])
    first = Runtime(store, owner="p1").run(mid, goal_program(GOAL, tools, first_brain, GoalConfig(max_iterations=10, tool_budget=2)))
    assert first["status"] == "checkpoint", first
    assert counter["a"] == 2

    # Part 2: the person continues with a larger allowance. At step 2 the stored finish is now NOT forced, so without the
    # stored allowance decision the finish would be refused here without its evaluation, and step 3 would be asked for
    # where the ledger holds the evaluation (replay diverged). The brain is asked for the new decisions only.
    store.set_budget(mid, {"max_tool_calls": 4, "max_seconds": 900, "segment": 1})
    second_brain = Queue(plans=[_call(3), {"action": "finish", "rationale": "done"}, {"action": "finish", "rationale": "done"},
                                {"action": "finish", "rationale": "done"}, {"action": "finish", "rationale": "done"}],
                         evals=[{"verdict": "not_met", "reason": "still open"}, {"verdict": "not_met", "reason": "still open"}])
    second = Runtime(store, owner="p2").run(mid, goal_program(GOAL, tools, second_brain,
                                                              GoalConfig(max_iterations=10, tool_budget=4, segment=1)))
    assert second["status"] != "failed", second
    assert "diverged" not in str(second), second
    assert second_brain.asked == ["plan", "plan", "plan", "plan", "eval", "plan", "eval"], second_brain.asked
    assert counter["a"] == 3, "the replayed calls ran again or the new call did not run once"
