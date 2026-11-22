"""Round 3 replay: one mission that uses every new path (a repeat, a refused argument list, a paged observation, a refused finish)
is run once, then resumed by another owner with a brain that must not be asked anything. The resumed run must take the same path
and must not run a tool again. This is the invariant the brief sets: every model- or history-dependent decision is stored, so a
resumed run reads it back rather than deciding again.
"""
from __future__ import annotations

import pytest

from prophet_agent.runtime.goal import GoalConfig, goal_program
from prophet_agent.runtime.runner import Runtime, Tool
from prophet_agent.runtime.store import Store

GOAL = "forecast the daily sales for the next 30 days"


@pytest.fixture()
def store(tmp_path):
    s = Store(tmp_path / "replay.sqlite")
    yield s
    s.close()


class Scripted:
    def __init__(self, plans, evals):
        self.plans, self.evals = list(plans), list(evals)

    def decide(self, messages, schema, name):
        queue = self.plans if name == "plan" else self.evals
        if not queue:
            raise AssertionError(f"brain asked for {name} but the script is exhausted")
        return queue.pop(0)


class Silent:
    def decide(self, messages, schema, name):
        raise AssertionError(f"a resumed run asked for {name}")


def call(tool, args):
    return {"action": "call_tool", "tool": tool, "args": args, "rationale": "work"}


def finish():
    return {"action": "finish", "rationale": "done"}


def _tools(counter):
    def a(args, key):
        counter["a"] += 1
        return {"ok": True, "k": args.get("k"), "text": "v" * 5000}

    def strict(args, key):
        counter["strict"] += 1
        if set(args) - {"k"}:
            raise ValueError(f"unknown arguments {sorted(set(args) - {'k'})}; allowed: k")
        return {"ok": True}

    return {"a": Tool("a", a, idempotent=True), "strict": Tool("strict", strict, idempotent=True)}


def _program(tools, brain):
    return goal_program(GOAL, tools, brain, GoalConfig(max_iterations=12))


def test_a_mixed_mission_resumes_on_the_same_path_without_running_a_tool_or_asking_the_brain(store):
    counter = {"a": 0, "strict": 0}
    tools = _tools(counter)
    brain = Scripted(
        plans=[call("a", {"k": 1}), call("a", {"k": 1}), call("strict", {"x": 1}),
               call("read_observation", {"step": 1, "page": 2}), finish(), finish(), finish()],
        evals=[{"verdict": "met", "reason": "enough", "answer": "Forecast written."}])
    mid = store.create_mission(GOAL, {"max_tool_calls": 20})
    first = Runtime(store, owner="r1").run(mid, _program(tools, brain))
    evidence = first["result"]["evidence"]

    assert counter == {"a": 1, "strict": 1}, "the repeat was not sent to the tool, and the refused call ran once"
    assert any(h.get("repeat_of") == 1 for h in evidence)
    assert any(h.get("rejected") is True for h in evidence)
    assert any(h.get("kind") == "finish_refused" for h in evidence)
    assert any(h.get("tool") == "read_observation" and "result" in h for h in evidence)

    again = Runtime(store, owner="r2").run(mid, _program(tools, Silent()))
    assert again["result"]["evidence"] == evidence, "the resumed run took a different path"
    assert again["result"]["status"] == first["result"]["status"]
    assert counter == {"a": 1, "strict": 1}, "a resumed run ran a tool again"
