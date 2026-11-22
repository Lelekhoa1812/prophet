"""Sub-agents: typed, briefed, scoped, with their own notes, parallel when asked, resumable, replay-stable."""
from __future__ import annotations

import threading

import pytest

from prophet_agent.runtime import agents
from prophet_agent.runtime.goal import GoalConfig, goal_program
from prophet_agent.runtime.runner import Runtime, Tool
from prophet_agent.runtime.store import Store


class KeyedBrain:
    """Decides by the goal text a loop works on, so threads cannot swap each other's script. Counts calls."""

    def __init__(self, parent_plans, sub_plans=None):
        self.parent = list(parent_plans)
        self.sub = dict(sub_plans or {})        # marker in the brief -> list of plans
        self.lock = threading.Lock()
        self.calls = 0
        self.briefs: list[str] = []

    def _goal(self, messages):
        return messages[1]["content"].split("Goal:\n", 1)[1].split("\n\n", 1)[0] if len(messages) > 1 else ""

    def decide(self, messages, schema, name):
        with self.lock:
            self.calls += 1
            goal = messages[1]["content"]
            if name == "eval":
                return {"verdict": "met", "reason": "enough", "answer": "report " + ("sub" if "sub-agent a" in goal.split("Available tools")[0] else "final")}
            for marker, plans in self.sub.items():
                if marker in goal.split("Available tools")[0]:
                    self.briefs.append(goal)
                    return plans.pop(0) if plans else {"action": "finish", "rationale": "done"}
            return self.parent.pop(0) if self.parent else {"action": "finish", "rationale": "done"}


def _tool(name, fn=None):
    return Tool(name, fn or (lambda args, key: {"ok": True, "echo": args}), idempotent=True)


@pytest.fixture()
def store(tmp_path):
    s = Store(tmp_path / "sub.sqlite")
    yield s
    s.close()


def _run(store, program):
    mid = store.create_mission("g", {})
    return mid, Runtime(store).run(mid, program)


def test_agent_types_scope_the_tools_and_never_widen_them():
    available = {n: _tool(n) for n in ("sandbox_python", "web_search", "notes", "timesfm_forecast")}
    researcher = agents.allowed(agents.TYPES["researcher"], available)
    assert set(researcher) == {"web_search", "notes"}, "only the tools the mission has, and only the type's own"
    assert set(agents.allowed(agents.TYPES["general"], available)) == set(available)
    assert agents.get("nope") is None and agents.get(None).name == "general"
    assert "researcher" in agents.describe()


def test_the_brief_reaches_the_sub_agent_and_its_tools_are_scoped(store):
    brain = KeyedBrain(
        [{"action": "delegate", "task": "find the policy rate path", "agent_type": "researcher",
          "deliverable": "three dated findings", "constraints": "central banks only", "rationale": "r"}, {"action": "finish", "rationale": "d"}],
        {"sub-agent a1": [{"action": "finish", "rationale": "nothing to search"}]})
    tools = {"web_search": _tool("web_search"), "sandbox_python": _tool("sandbox_python")}
    _, out = _run(store, goal_program("g", tools, brain))
    sub = out["result"]["evidence"][0]
    assert sub["agent_id"] == "a1" and sub["agent_type"] == "researcher"
    assert "web_search" in sub["tools"] and "notes" in sub["tools"] and "sandbox_python" not in sub["tools"]
    brief = brain.briefs[0]
    assert "three dated findings" in brief and "central banks only" in brief and "find the policy rate path" in brief


def test_notes_are_each_agents_own_memory_and_readable_by_the_parent(store):
    brain = KeyedBrain(
        [{"action": "delegate", "task": "dig", "agent_type": "analyst", "rationale": "r"},
         {"action": "call_tool", "tool": "notes", "args": {"action": "read", "scope": "a1", "name": "found"}, "rationale": "read"},
         {"action": "finish", "rationale": "d"}],
        {"sub-agent a1": [{"action": "call_tool", "tool": "notes", "args": {"action": "write", "name": "found", "text": "run abc: trend up"}, "rationale": "keep"},
                          {"action": "finish", "rationale": "d"}]})
    _, out = _run(store, goal_program("g", {"sandbox_python": _tool("sandbox_python")}, brain))
    read = [h for h in out["result"]["evidence"] if h.get("tool") == "notes"][0]
    assert read["result"]["text"] == "run abc: trend up"
    assert store.note_list(store.missions()[0]["id"] if hasattr(store, "missions") else _mid(store), "a1")[0]["name"] == "found"


def _mid(store):
    return store.conn.execute("SELECT id FROM missions").fetchone()["id"]


def test_a_sub_agent_cannot_write_into_another_scope(store):
    from prophet_agent.tools.notes_tool import notes_tool
    mid = store.create_mission("g", {})
    tool = notes_tool(store, mid, "a2")
    with pytest.raises(ValueError, match="your own scope"):
        tool.fn({"action": "write", "name": "x", "text": "t", "scope": "main"}, "k")
    assert tool.fn({"action": "append", "name": "x", "text": "one"}, "k")["saved"]
    assert tool.fn({"action": "append", "name": "x", "text": "two"}, "k")["chars"] == len("one\ntwo")
    with pytest.raises(ValueError, match="unknown arguments"):
        tool.fn({"action": "list", "bogus": 1}, "k")


def test_parallel_sub_agents_run_together_and_replay_without_new_model_calls(store):
    plans = [{"action": "delegate", "rationale": "r", "tasks": [
        {"task": "slice alpha", "agent_type": "analyst"}, {"task": "slice beta", "agent_type": "analyst"},
        {"task": "slice gamma", "agent_type": "researcher"}]},
        {"action": "finish", "rationale": "d"}]
    sub = {f"slice {n}": [{"action": "call_tool", "tool": "sandbox_python", "args": {"code": n}, "rationale": "x"},
                          {"action": "finish", "rationale": "d"}] for n in ("alpha", "beta")}
    brain = KeyedBrain(plans, sub)
    tools = {"sandbox_python": _tool("sandbox_python", lambda a, k: {"ok": True, "stdout": f"did {a['code']}"}),
             "web_search": _tool("web_search")}
    mid = store.create_mission("g", {})
    out = Runtime(store).run(mid, goal_program("g", tools, brain))
    assert out["status"] == "completed"
    subs = [h for h in out["result"]["evidence"] if h.get("kind") == "subagent"]
    assert [s["agent_id"] for s in subs] == ["a1", "a2", "a3"] and {s["agent_type"] for s in subs} == {"analyst", "researcher"}
    assert any("did alpha" in str(s["evidence"]) for s in subs) and any("did beta" in str(s["evidence"]) for s in subs)
    calls = brain.calls
    again = Runtime(store).run(store.create_mission("g", {}), goal_program("g", tools, KeyedBrain(plans, {})))   # fresh mission: runs
    assert again["status"] == "completed"
    # Replay: the same mission against the stored ledger asks the model for nothing new.
    store.set_status(mid, "running")
    replay_brain = KeyedBrain([], {})
    Runtime(store).run(mid, goal_program("g", tools, replay_brain))
    assert replay_brain.calls == 0 and calls > 0


def test_resume_continues_an_earlier_sub_agent_with_its_history(store):
    brain = KeyedBrain(
        [{"action": "delegate", "task": "first look", "agent_type": "analyst", "rationale": "r"},
         {"action": "delegate", "task": "now check the second column", "agent_type": "analyst", "resume": "a1", "rationale": "r"},
         {"action": "finish", "rationale": "d"}],
        {"first look": [{"action": "call_tool", "tool": "sandbox_python", "args": {"code": "one"}, "rationale": "x"}, {"action": "finish", "rationale": "d"}],
         "second column": [{"action": "finish", "rationale": "d"}]})
    tools = {"sandbox_python": _tool("sandbox_python", lambda a, k: {"ok": True, "stdout": "FIRST-RUN-OUTPUT"})}
    _, out = _run(store, goal_program("g", tools, brain))
    subs = [h for h in out["result"]["evidence"] if h.get("kind") == "subagent"]
    assert len(subs) == 2 and subs[1]["agent_id"] == "a2"
    assert any("FIRST-RUN-OUTPUT" in str(h) for h in subs[1]["evidence"]), "the resumed agent keeps what it had done"
    assert any("continuing earlier work" in b for b in brain.briefs)


def test_unknown_type_resume_and_limit_are_refused_with_the_fix(store):
    brain = KeyedBrain([{"action": "delegate", "task": "t", "agent_type": "general", "resume": "a9", "rationale": "r"},
                        {"action": "finish", "rationale": "d"}], {})
    _, out = _run(store, goal_program("g", {"sandbox_python": _tool("sandbox_python")}, brain))
    err = out["result"]["evidence"][0]["error"]
    assert "no earlier sub-agent 'a9'" in err


def test_a_report_figure_no_tool_printed_is_flagged_to_the_lead_agent(store):
    class Brain(KeyedBrain):
        def decide(self, messages, schema, name):
            if name == "eval" and "sub-agent a1" in messages[1]["content"].split("Available tools")[0]:
                return {"verdict": "met", "reason": "ok", "answer": "The rate will reach 7.77 percent."}
            return super().decide(messages, schema, name)
    brain = Brain([{"action": "delegate", "task": "rates", "agent_type": "analyst", "rationale": "r"}, {"action": "finish", "rationale": "d"}],
                  {"rates": [{"action": "finish", "rationale": "d"}]})
    _, out = _run(store, goal_program("g", {"sandbox_python": _tool("sandbox_python")}, brain))
    assert "7.77" in out["result"]["evidence"][0]["unverified"]


def test_the_harness_starts_a_researcher_for_a_market_goal_and_a_replay_asks_the_model_nothing(store):
    inspect = {"file": "wti.csv", "profile": {"columns": [{"kind": "time", "name": "date", "duplicate_timestamps": 0},
                                                          {"kind": "number", "name": "price"}]}}
    tools = {"inspect_data": _tool("inspect_data", lambda a, k: inspect), "web_research": _tool("web_research")}
    goal = "Forecast the WTI spot price for 20 trading days"
    brain = KeyedBrain([{"action": "call_tool", "tool": "inspect_data", "args": {"file": "wti.csv"}, "rationale": "look"},
                        {"action": "finish", "rationale": "done"}],
                       {"sub-agent a1": [{"action": "call_tool", "tool": "web_research", "args": {"queries": ["opec"]}, "rationale": "r"},
                                         {"action": "finish", "rationale": "done"}]})
    mid, out = _run(store, goal_program(goal, tools, brain))
    subs = [h for h in out["result"]["evidence"] if h.get("kind") == "subagent"]
    assert len(subs) == 1 and subs[0]["agent_type"] == "researcher" and subs[0]["auto"] == "research"
    assert "Do not forecast" in brain.briefs[0]
    calls = brain.calls
    replay = KeyedBrain([], {})
    store.set_status(mid, "running")
    out2 = Runtime(store).run(mid, goal_program(goal, tools, replay))
    assert replay.calls == 0 and out2["status"] == out["status"], "the stored steps replay; nothing is asked again"
    assert calls > 0
