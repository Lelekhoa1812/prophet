"""Round 4 C4, audit A4-14: a sub-agent's own figure list is reconciled before the lead reads its report.

Motivation vs Logic
Motivation: a sub-agent reports figures in its verdict, each marked ``sourced`` or not against what its tools printed. The
lead was told only about unsourced figures that appear in the sub-agent's answer text. A figure the sub-agent listed as
unsourced, but left out of its prose, reached the lead without the warning the report is meant to carry.
Logic: the report's ``unverified`` list is the union of the answer's unsourced figures and the figures the sub-agent
itself marked unsourced. Nothing is dropped and nothing is reworded.
"""
from __future__ import annotations

import pytest

from prophet_agent.runtime.goal import GoalConfig, _one_agent
from prophet_agent.runtime.runner import Runtime, Tool
from prophet_agent.runtime.store import Store


@pytest.fixture()
def store(tmp_path):
    s = Store(tmp_path / "subagent.sqlite")
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


def test_a_figure_the_subagent_marked_unsourced_reaches_the_lead_even_if_the_answer_omits_it(store):
    tools = {"a": Tool("a", lambda args, key: {"ok": True}, idempotent=True)}
    brain = Queue(plans=[{"action": "finish", "rationale": "done"}],
                  evals=[{"verdict": "met", "answer": "The sales were steady.", "reason": "done",
                          "findings": ["sales were steady"],
                          "figures": [{"name": "peak", "value": "99.9"}]}])
    seen = {}
    spec = {"agent_type": "general", "task": "check the sales level"}

    def program(ctx):
        seen["obs"] = _one_agent(ctx, brain, spec, 1, tools, GoalConfig(max_iterations=10, tool_budget=4), "", [])
        return {"status": "done"}

    mid = store.create_mission("check the sales level", {"max_tool_calls": 4, "max_seconds": 900})
    Runtime(store, owner="p1").run(mid, program)
    obs = seen["obs"]
    assert obs["kind"] == "subagent", obs
    assert obs["figures"] and obs["figures"][0]["sourced"] is False, obs
    assert "99.9" in obs.get("unverified", []), obs
