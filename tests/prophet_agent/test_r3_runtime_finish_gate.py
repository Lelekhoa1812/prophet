"""Round 3, runtime goal 5 (A5-04): a premature finish is refused before the evaluator runs, and the refusal names the next item.

Before this, every finish went to the evaluator, which then judged an unfinished forecast and wrote a verdict the planner read
as a finding. A finish with checklist items still to do is now refused in the run itself, with the next item named, and the
evaluator is not asked. The refusals are capped, so a goal that the checklist can never close still reaches the evaluator.
"""
from __future__ import annotations

import pytest

from prophet_agent.runtime.goal import GoalConfig, goal_program
from prophet_agent.runtime.runner import Runtime, Tool
from prophet_agent.runtime.store import Store

GOAL = "forecast the daily sales for the next 30 days"


@pytest.fixture()
def store(tmp_path):
    s = Store(tmp_path / "finish.sqlite")
    yield s
    s.close()


class Scripted:
    def __init__(self, plans, evals=()):
        self.plans, self.evals = list(plans), list(evals)
        self.calls = {"plan": 0, "eval": 0}

    def decide(self, messages, schema, name):
        self.calls[name] += 1
        queue = self.plans if name == "plan" else self.evals
        if not queue:
            raise AssertionError(f"brain asked for {name} but the script is exhausted")
        return queue.pop(0)


def finish(why="the work is done"):
    return {"action": "finish", "rationale": why}


def met(answer="ok"):
    return {"verdict": "met", "reason": "enough", "answer": answer}


def _inspect():
    return Tool("inspect_data", lambda args, key: {"ok": True, "columns": ["day", "sales"]}, idempotent=True)


def test_a_premature_finish_is_refused_before_the_evaluator_is_asked(store):
    # The evaluator has no script for the first two finishes: if it were asked, the run would stop with an AssertionError.
    brain = Scripted([finish(), finish(), finish()], [met()])
    mid = store.create_mission(GOAL, {"max_tool_calls": 20})
    out = Runtime(store, owner="r1").run(mid, goal_program(GOAL, {"inspect_data": _inspect()}, brain,
                                                           GoalConfig(max_iterations=10)))
    refusals = [h for h in out["result"]["evidence"] if h.get("kind") == "finish_refused"]
    assert refusals, "the first finish is not met: the checklist still has items to do"
    assert "inspect the data" in refusals[0]["next"], refusals[0]
    assert "still to do" in refusals[0]["note"], refusals[0]["note"]


def test_refusals_are_capped_so_an_unclosable_goal_still_reaches_the_evaluator(store):
    brain = Scripted([finish(), finish(), finish()], [met()])
    mid = store.create_mission(GOAL, {"max_tool_calls": 20})
    out = Runtime(store, owner="r1").run(mid, goal_program(GOAL, {"inspect_data": _inspect()}, brain,
                                                           GoalConfig(max_iterations=10)))
    refused = sum(1 for h in out["result"]["evidence"] if h.get("kind") == "finish_refused")
    assert refused == 2, "two refusals, then the third finish goes to the evaluator"
    assert brain.calls["eval"] == 1


def test_a_resumed_run_refuses_the_same_finish_without_asking_the_brain(store):
    brain = Scripted([finish(), finish(), finish()], [met()])
    mid = store.create_mission(GOAL, {"max_tool_calls": 20})
    first = Runtime(store, owner="r1").run(mid, goal_program(GOAL, {"inspect_data": _inspect()}, brain,
                                                             GoalConfig(max_iterations=10)))

    class Silent:
        def decide(self, messages, schema, name):
            raise AssertionError(f"a resumed run asked for {name}")

    again = Runtime(store, owner="r2").run(mid, goal_program(GOAL, {"inspect_data": _inspect()}, Silent(),
                                                             GoalConfig(max_iterations=10)))
    assert again["result"]["evidence"] == first["result"]["evidence"]


def test_a_finish_at_a_part_boundary_is_judged_the_same_way_on_resume(store):
    # Live evidence (api test_a_checkpointed..., 2026-11-22): part 1 ended on a finish that was the part's last step, so the
    # evaluator ran. On resume the same step was no longer the last of the run and the gate refused it, so the replay
    # asked for a plan where the ledger held an eval ("replay diverged at step 2"). A part boundary is stable across parts.
    brain = Scripted([finish(), finish()], [{"verdict": "not_met", "reason": "need more"}, met("done")])
    mid = store.create_mission(GOAL, {"max_tool_calls": 20})
    Runtime(store, owner="r1").run(mid, goal_program(GOAL, {"inspect_data": _inspect()}, brain,
                                                     GoalConfig(max_iterations=1, segment=0)))
    again = Runtime(store, owner="r2").run(mid, goal_program(GOAL, {"inspect_data": _inspect()}, brain,
                                                             GoalConfig(max_iterations=1, segment=1)))
    assert again["status"] != "failed", again.get("error")
    assert again["result"]["status"] == "met"
    assert brain.calls["eval"] == 2, "each part's boundary finish is judged once, and the resume replays it"
