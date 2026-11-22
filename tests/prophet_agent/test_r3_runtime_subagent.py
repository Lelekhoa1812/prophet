"""Round 3, runtime goal 6 (A5-05, A3-02): a sub-agent's report is typed.

Before this, a sub-agent that gave no written answer still reported its run status, so the lead read an empty report as work
done. Now an empty answer is "not_met", and the report carries findings and figures. A figure is sourced only when a number in
the sub-agent's own evidence supports it; the flag is computed here, not taken from the model.
"""
from __future__ import annotations

import pytest

from prophet_agent.runtime.goal import GoalConfig, goal_program
from prophet_agent.runtime.runner import Runtime, Tool
from prophet_agent.runtime.store import Store


@pytest.fixture()
def store(tmp_path):
    s = Store(tmp_path / "subagent.sqlite")
    yield s
    s.close()


class Router:
    """A lead and its sub-agents, each with its own script, found by the id the brief names."""

    def __init__(self, plans, evals):
        self.plans = {k: list(v) for k, v in plans.items()}
        self.evals = {k: list(v) for k, v in evals.items()}

    def decide(self, messages, schema, name):
        text = messages[-1]["content"]
        who = text.split("You are sub-agent ", 1)[1].split(" ", 1)[0] if "You are sub-agent " in text else "lead"
        queue = (self.plans if name == "plan" else self.evals).setdefault(who, [])
        if not queue:
            if name == "eval":
                return {"verdict": "met", "reason": "enough", "answer": f"{who} done"}
            raise AssertionError(f"{who} asked for a {name} but its script is exhausted")
        return queue.pop(0)


def call(tool, args):
    return {"action": "call_tool", "tool": tool, "args": args, "rationale": "work"}


def finish():
    return {"action": "finish", "rationale": "done"}


def delegate(task):
    return {"action": "delegate", "rationale": "split", "tasks": [{"task": task, "agent_type": "analyst"}]}


GOAL = "summarise the stock list for the store"


def _run(store, router):
    program = goal_program(GOAL, {"sandbox_python": Tool("sandbox_python", lambda a, k: {"ok": True, "demand": 12},
                                                          idempotent=True)}, router, GoalConfig(max_iterations=12))
    mid = store.create_mission(GOAL, {"max_tool_calls": 20})
    return mid, Runtime(store, owner="r1").run(mid, program)


def _subagent(out):
    return next(h for h in out["result"]["evidence"] if h.get("kind") == "subagent")


def test_a_sub_agent_with_no_written_answer_is_not_met(store):
    blank = {"verdict": "met", "reason": "done", "answer": ""}
    router = Router(plans={"lead": [delegate("count the lines"), finish()], "a1": [finish()] * 20},
                    evals={"a1": [dict(blank) for _ in range(20)]})
    _, out = _run(store, router)
    report = _subagent(out)
    assert report["status"] == "not_met", report
    assert "no written answer" in report["reason"], report["reason"]


def test_a_sub_agent_report_carries_findings_and_figures_marked_by_whether_a_tool_printed_them(store):
    router = Router(
        plans={"lead": [delegate("find the demand"), finish()],
               "a1": [call("sandbox_python", {"code": "print(12)"}), finish()]},
        evals={"a1": [{"verdict": "met", "reason": "done", "answer": "Demand is 12 units a day.",
                       "findings": ["demand is 12 units a day"],
                       "figures": [{"name": "demand", "value": "12"}, {"name": "growth", "value": "40"}]}]})
    _, out = _run(store, router)
    report = _subagent(out)
    assert report["status"] == "met", report
    assert report["findings"] == ["demand is 12 units a day"]
    by_name = {f["name"]: f for f in report["figures"]}
    assert by_name["demand"]["sourced"] is True, "12 is in the tool's printed output"
    assert by_name["growth"]["sourced"] is False, "40 was never printed by a tool"
