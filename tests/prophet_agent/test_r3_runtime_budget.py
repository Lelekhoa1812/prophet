"""Round 3, runtime goal 1 (A5-01, A3-01, A5-08): one budget ledger.

Measured 2026-11-22 (live Beijing panel, data/live_panel_beijing_rep1.json): four panel analysts spent 44 tool calls, the
planner's count reached 45 against an allowance of 20, and the lead was forced to finish with "tool budget used" before it
had made one forecast. Lead calls and delegated calls now have their own counts: the lead's allowance is spent by its own
calls only, and delegated work has a separate cap.
"""
from __future__ import annotations

import pytest

from prophet_agent.runtime.goal import GoalConfig, _budget_note, _calls_used, goal_program
from prophet_agent.runtime.runner import Runtime, Tool
from prophet_agent.runtime.store import Store


@pytest.fixture()
def store(tmp_path):
    s = Store(tmp_path / "budget.sqlite")
    yield s
    s.close()


def _tool(name, fn=None):
    return Tool(name, fn or (lambda args, key: {"ok": True, "stdout": str(args)}), idempotent=True)


class Router:
    """One scripted brain for the lead and its sub-agents. A sub-agent's decisions are read from its own queue, found by
    the id its brief names ("You are sub-agent a1"); the lead's are read from the "lead" queue."""

    def __init__(self, plans=None, evals=None):
        self.plans = {k: list(v) for k, v in (plans or {}).items()}
        self.evals = {k: list(v) for k, v in (evals or {}).items()}
        self.calls = {"plan": 0, "eval": 0}

    def decide(self, messages, schema, name):
        self.calls[name] += 1
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


def finish(why="done"):
    return {"action": "finish", "rationale": why}


def delegate(n):
    return {"action": "delegate", "rationale": "split the work",
            "tasks": [{"task": f"analyse slice {k}", "agent_type": "analyst"} for k in range(n)]}


def _cell(k):
    return call("sandbox_python", {"code": f"print({k})"})


def _sandbox():
    return _tool("sandbox_python", lambda args, key: {"ok": True, "run_id": "a" * 32, "stdout": args["code"]})


def test_delegated_calls_do_not_force_the_lead_to_finish(store):
    # Four analysts each make ten calls (40 in all) against a lead allowance of 20. The lead still makes its own three
    # calls and then finishes; it is never told "tool budget used".
    plans = {f"a{k}": [_cell(10 * k + j) for j in range(10)] + [finish()] for k in range(1, 5)}
    plans["lead"] = [delegate(4), _cell(900), _cell(901), _cell(902), finish()]
    brain = Router(plans=plans)
    program = goal_program("analyse the daily sales", {"sandbox_python": _sandbox()}, brain,
                           GoalConfig(max_iterations=12, tool_budget=20))
    mid = store.create_mission("analyse the daily sales", {"max_tool_calls": 20})
    out = Runtime(store).run(mid, program)
    evidence = out["result"]["evidence"]
    assert not any(h.get("kind") == "finish" and h.get("rationale") == "tool budget used" for h in evidence), (
        "the lead was forced to finish by its delegates' calls")
    assert out["result"]["status"] == "met", out["result"]
    assert _calls_used(evidence) == 3, "the lead's own calls, not the sub-agents' forty"
    assert store.count_tool_calls(mid, lane="lead") == 3
    assert store.count_tool_calls(mid, lane="delegated") == 40


def test_a_delegation_past_its_own_cap_is_stopped_and_the_lead_keeps_its_allowance(store):
    plans = {f"a{k}": [_cell(10 * k + j) for j in range(10)] + [finish()] for k in range(1, 5)}
    plans["lead"] = [delegate(4), finish()]
    brain = Router(plans=plans)
    program = goal_program("analyse the daily sales", {"sandbox_python": _sandbox()}, brain,
                           GoalConfig(max_iterations=12, tool_budget=20))
    mid = store.create_mission("analyse the daily sales", {"max_tool_calls": 20, "max_delegated_calls": 5})
    out = Runtime(store).run(mid, program)
    assert out["status"] == "checkpoint" and out["which"] == "max_delegated_calls", (
        "a sub-agent work grant is a separate cap; exceeding it stops the delegated work, resumably")
    assert store.count_tool_calls(mid, lane="lead") == 0


def test_the_planner_note_counts_only_the_lead_calls():
    # Audit A3-01 measured case: a researcher's 13 calls and the lead's 3 leave 17 of 20 for the lead.
    research = {"kind": "subagent", "agent_id": "a1", "agent_type": "researcher", "status": "met", "answer": "x",
                "evidence": [{"kind": "tool", "tool": "web_search", "args": {"q": k}, "result": {"ok": True}}
                             for k in range(13)]}
    lead = [{"kind": "tool", "tool": "sandbox_python", "args": {"code": str(k)}, "result": {"ok": True}} for k in range(3)]
    history = lead + [research]
    assert _calls_used(history) == 3
    note = _budget_note(0, GoalConfig(max_iterations=50, tool_budget=20), history)
    assert "17 steps left" in note, note


def test_the_store_keeps_lead_and_delegated_calls_apart(store):
    mid = store.create_mission("g", {})
    store.tool_start(mid, 1, "sandbox_python", {"code": "a"}, "k-lead", lane="lead")
    store.tool_start(mid, 2, "sandbox_python", {"code": "b"}, "k-sub", lane="delegated")
    store.tool_start(mid, 3, "inspect_data", {"file": "x"}, "k-sub2", lane="delegated")
    assert store.count_tool_calls(mid) == 3
    assert store.count_tool_calls(mid, lane="lead") == 1
    assert store.count_tool_calls(mid, lane="delegated") == 2
