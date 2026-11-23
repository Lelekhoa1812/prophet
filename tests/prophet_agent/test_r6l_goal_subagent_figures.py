"""Round 6l, goal audit item 3: a sub-agent's figures are checked against what its tools printed, not against its brief.

Synthetic figures only. The task is the lead agent's own words, so a figure repeated from it must not count as sourced.
"""
from __future__ import annotations

import pytest

from prophet_agent.runtime.goal import GoalConfig, _one_agent
from prophet_agent.runtime.runner import Runtime, Tool
from prophet_agent.runtime.store import Store

TASK = "Check whether the July total is 4,500 units and report it."


@pytest.fixture()
def store(tmp_path):
    s = Store(tmp_path / "subagent_figures.sqlite")
    yield s
    s.close()


class Queue:
    def __init__(self, plans, evals):
        self.plans, self.evals = list(plans), list(evals)

    def decide(self, messages, schema, name):
        queue = self.plans if name == "plan" else self.evals
        if not queue:
            raise AssertionError(f"the brain was asked for {name} but no decision is scripted")
        return queue.pop(0)


def _run_subagent(store, tool_result, verdict):
    tools = {"a": Tool("a", lambda args, key: tool_result, idempotent=True)}
    brain = Queue(plans=[{"action": "call_tool", "tool": "a", "args": {}, "rationale": "look"},
                         {"action": "finish", "rationale": "done"}],
                  evals=[verdict])
    seen = {}
    spec = {"agent_type": "general", "task": TASK}

    def program(ctx):
        seen["obs"] = _one_agent(ctx, brain, spec, 1, tools, GoalConfig(max_iterations=10, tool_budget=4), "", [])
        return {"status": "done"}

    mid = store.create_mission(TASK, {"max_tool_calls": 4, "max_seconds": 900})
    Runtime(store, owner="p1").run(mid, program)
    return seen["obs"]


def test_a_figure_repeated_from_the_brief_without_any_tool_printing_it_is_not_sourced(store):
    verdict = {"verdict": "met", "answer": "The July total is 4,500 units.", "reason": "done",
               "findings": ["July total checked"], "figures": [{"name": "july total", "value": "4,500"}]}
    obs = _run_subagent(store, {"ok": True}, verdict)
    assert obs["figures"] and obs["figures"][0]["sourced"] is False, obs
    assert "4,500" in obs.get("unverified", []), obs


def test_a_figure_a_tool_printed_is_still_sourced(store):
    verdict = {"verdict": "met", "answer": "The July total is 4,500 units.", "reason": "done",
               "findings": ["July total checked"], "figures": [{"name": "july total", "value": "4,500"}]}
    obs = _run_subagent(store, {"july_total": 4500}, verdict)
    assert obs["figures"][0]["sourced"] is True, obs
    assert "4,500" not in obs.get("unverified", []), obs
