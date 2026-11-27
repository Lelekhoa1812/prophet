"""Research lanes: many sub-agents at once, each with its own loop, a shared board, failure isolation and replay stability
(runtime/lanes.py; owner request 2026-11-27)."""
from __future__ import annotations

import threading

import pytest

from prophet_agent.runtime import lanes
from prophet_agent.runtime.goal import PLAN_SCHEMA, GoalConfig, goal_program
from prophet_agent.runtime.runner import Runtime, Tool
from prophet_agent.runtime.store import Store
from prophet_agent.nim.schema import validate


class LaneBrain:
    """Plans by the brief a loop works on, so concurrent lanes cannot swap scripts. Counts calls."""

    def __init__(self, lead, lane_plans=None, fail_marker=None):
        self.lead = list(lead)
        self.lane_plans = {k: list(v) for k, v in (lane_plans or {}).items()}
        self.fail_marker = fail_marker
        self.lock = threading.Lock()
        self.calls = 0
        self.seen: dict[str, list[str]] = {}

    def decide(self, messages, schema, name):
        with self.lock:
            self.calls += 1
            text = messages[1]["content"]
            head = text.split("Available tools")[0]
            task = head.split("Task: ", 1)[1].split("\n", 1)[0] if "Task: " in head else ""
            if self.fail_marker and self.fail_marker == task:
                raise ValueError("this lane's model broke")
            if name == "eval":
                return {"verdict": "met", "reason": "enough", "answer": "lane report" if "research lane" in head else "final",
                        "findings": ["a finding"]}
            for marker, plans in self.lane_plans.items():
                if marker == task:
                    self.seen.setdefault(marker, []).append(text)
                    return plans.pop(0) if plans else {"action": "finish", "rationale": "done"}
            return self.lead.pop(0) if self.lead else {"action": "finish", "rationale": "done"}


def _tool(name, fn=None):
    return Tool(name, fn or (lambda args, key: {"ok": True, "echo": args}), idempotent=True)


@pytest.fixture()
def store(tmp_path):
    s = Store(tmp_path / "lanes.sqlite")
    yield s
    s.close()


def _fan_out(n, **extra):
    return {"action": "lanes", "rationale": "broad question",
            "lanes": [{"task": f"angle {k}", "agent_type": "researcher", **extra} for k in range(n)]}


def test_the_plan_schema_accepts_up_to_fifty_lanes_and_requires_them():
    assert validate(_fan_out(50), PLAN_SCHEMA) is None
    assert validate(_fan_out(51), PLAN_SCHEMA) is not None
    assert validate({"action": "lanes", "rationale": "r"}, PLAN_SCHEMA) is not None


def test_twenty_lanes_run_on_a_bounded_pool_and_replay_without_new_model_calls(store):
    plans = {f"angle {k}": [{"action": "call_tool", "tool": "web_search", "args": {"q": f"q{k}"}, "rationale": "x"},
                            {"action": "finish", "rationale": "d"}] for k in range(20)}
    brain = LaneBrain([_fan_out(20), {"action": "finish", "rationale": "d"}], plans)
    tools = {"web_search": _tool("web_search")}
    mid = store.create_mission("g", {})
    out = Runtime(store).run(mid, goal_program("g", tools, brain, GoalConfig(lane_workers=6)))
    assert out["status"] == "completed"
    ev = out["result"]["evidence"]
    subs = [h for h in ev if h.get("kind") == "subagent"]
    assert len(subs) == 20 and len({s["agent_id"] for s in subs}) == 20 and all(s.get("lane") for s in subs)
    digest = next(h for h in ev if h.get("kind") == "lanes")
    assert len(digest["lanes"]) == 20 and digest["summary"].startswith("20 lanes")
    assert brain.calls > 20
    store.set_status(mid, "running")
    replay = LaneBrain([], {})
    Runtime(store).run(mid, goal_program("g", tools, replay, GoalConfig(lane_workers=6)))
    assert replay.calls == 0, "a resumed mission reads every lane's steps from the ledger"


def test_one_lane_failing_does_not_sink_the_others(store):
    brain = LaneBrain([_fan_out(3), {"action": "finish", "rationale": "d"}], {}, fail_marker="angle 1")
    out = Runtime(store).run(store.create_mission("g", {}), goal_program("g", {"web_search": _tool("web_search")}, brain))
    assert out["status"] == "completed"
    digest = next(h for h in out["result"]["evidence"] if h.get("kind") == "lanes")
    statuses = {row["task"]: row["status"] for row in digest["lanes"]}
    assert statuses["angle 1"] == "failed" and statuses["angle 0"] == statuses["angle 2"] == "met"
    assert "this lane's model broke" in next(r for r in digest["lanes"] if r["task"] == "angle 1")["error"]


def test_a_lane_reads_board_messages_before_it_plans_and_cannot_ask_the_person(store):
    mid = store.create_mission("g", {})
    store.board_post(mid, "lead", "all", "note", "the RBA cash rate file is already downloaded as rba.csv")
    brain = LaneBrain([_fan_out(1), {"action": "finish", "rationale": "d"}],
                      {"angle 0": [{"action": "ask_user", "rationale": "x", "questions": [
                          {"question": "Which?", "header": "Pick", "multiSelect": False,
                           "options": [{"label": "A", "description": "a"}, {"label": "B", "description": "b"}]}]},
                          {"action": "finish", "rationale": "d"}]})
    out = Runtime(store).run(mid, goal_program("g", {"web_search": _tool("web_search")}, brain))
    assert out["status"] == "completed"
    lane = next(h for h in out["result"]["evidence"] if h.get("kind") == "subagent")
    inbox = [h for h in lane["evidence"] if h.get("kind") == "board"]
    assert inbox and "rba.csv" in inbox[0]["messages"][0]["text"]
    refused = [h for h in lane["evidence"] if h.get("kind") == "ask_user"]
    assert refused and "cannot ask the person" in refused[0]["error"]
    assert "research lane a1" in brain.seen["angle 0"][0] and "angle 0" in brain.seen["angle 0"][0]


def test_the_board_posts_reads_and_claims_topics_once(store):
    mid = store.create_mission("g", {})

    class _Ctx:
        def __init__(self):
            self.store, self.mission_id = store, mid

    a1 = lanes.board_tool(_Ctx(), "a1", ["a2"])
    a2 = lanes.board_tool(_Ctx(), "a2", ["a1"])
    assert a1.fn({"action": "claim", "topic": "RBA minutes 2026"}, "k1")["claimed"] is True
    taken = a2.fn({"action": "claim", "topic": "rba  minutes, 2026"}, "k2")
    assert taken["claimed"] is False and taken["by"] == "a1"
    a1.fn({"action": "post", "to": "a2", "kind": "question", "text": "Do you have the 2025 decisions?"}, "k3")
    a1.fn({"action": "post", "to": "a2", "kind": "question", "text": "Do you have the 2025 decisions?"}, "k3")   # retried call
    got = a2.fn({"action": "read"}, "k4")
    questions = [m for m in got["messages"] if m["kind"] == "question"]
    assert len(questions) == 1 and questions[0]["sender"] == "a1"
    assert {"topic": "rba minutes 2026", "by": "a1"} in got["claimed_topics"]
    assert a1.fn({"action": "read"}, "k5")["messages"] == [m for m in a1.fn({"action": "read"}, "k5")["messages"]
                                                           if m["sender"] != "a1"]
    with pytest.raises(ValueError, match="to must be"):
        a1.fn({"action": "post", "to": "a9", "text": "x"}, "k6")


def test_an_identical_web_call_by_another_lane_is_answered_from_the_first(store):
    fetched = []

    def search(args, key):
        fetched.append(args["q"])
        return {"ok": True, "hits": ["page"]}

    same = [{"action": "call_tool", "tool": "web_search", "args": {"q": "rba cash rate"}, "rationale": "x"},
            {"action": "finish", "rationale": "d"}]
    brain = LaneBrain([_fan_out(3), {"action": "finish", "rationale": "d"}],
                      {f"angle {k}": [dict(p) for p in same] for k in range(3)})
    out = Runtime(store).run(store.create_mission("g", {}),
                             goal_program("g", {"web_search": _tool("web_search", search)}, brain, GoalConfig(lane_workers=1)))
    assert out["status"] == "completed"
    assert fetched == ["rba cash rate"], "the page is fetched once for three lanes"
    shared = [h for s in out["result"]["evidence"] if s.get("kind") == "subagent" for h in s["evidence"]
              if isinstance(h.get("result"), dict) and "shared_from_step" in h["result"]]
    assert len(shared) == 2


def test_a_lane_cannot_start_lanes():
    class Cfg:
        depth = 1
    assert "cannot start lanes" in lanes.run_lanes(None, None, _fan_out(2), {}, Cfg(), "", [], None, 100_000)[0]["error"]


def test_a_nested_planner_is_not_offered_lanes_or_delegate():
    """Round 20261128-063203: sub-agents planned "lanes" 924 times in VNUNEMP and 14 times in a row in SPX and EPL; each
    was refused and planned again."""
    from prophet_agent.runtime.goal import PLAN_SCHEMA_NESTED
    assert PLAN_SCHEMA_NESTED["properties"]["action"]["enum"] == ["call_tool", "ask_user", "finish"]
    assert validate(_fan_out(2), PLAN_SCHEMA_NESTED) is not None
    assert validate({"action": "call_tool", "tool": "web_search", "args": {}, "rationale": "r"}, PLAN_SCHEMA_NESTED) is None
    assert "lanes" in PLAN_SCHEMA["properties"]["action"]["enum"], "the lead keeps them"


def test_the_same_briefs_are_not_started_again(store):
    """Live AURATE re-sent the same six lane briefs fifteen times in a row."""
    done = {f"angle {k}": [{"action": "finish", "rationale": "d"}] for k in range(3)}
    brain = LaneBrain([_fan_out(2), _fan_out(3), _fan_out(2), {"action": "finish", "rationale": "d"}], done)
    out = Runtime(store).run(store.create_mission("g", {}), goal_program("g", {"web_search": _tool("web_search")}, brain))
    ev = out["result"]["evidence"]
    subs = [h for h in ev if h.get("kind") == "subagent"]
    assert sorted(s["task"] for s in subs) == ["angle 0", "angle 1", "angle 2"], "only the new brief ran the second time"
    digests = [h for h in ev if h.get("kind") == "lanes"]
    assert digests[1]["not_started_again"] and len(digests[1]["lanes"]) == 1
    assert "already ran" in digests[2]["error"]


def test_a_brief_can_name_mission_tools_outside_its_type(store):
    """Live AURATE: analyst lanes sent to download drivers reported "download_open_data is not available"."""
    plan = {"action": "lanes", "rationale": "drivers",
            "lanes": [{"task": "download and test the yield", "agent_type": "analyst",
                       "tools": ["download_open_data", "sandbox_python"]}]}
    lane = {"download and test the yield": [{"action": "call_tool", "tool": "download_open_data", "args": {"s": 1},
                                             "rationale": "x"}, {"action": "finish", "rationale": "d"}]}
    brain = LaneBrain([plan, {"action": "finish", "rationale": "d"}], lane)
    tools = {"download_open_data": _tool("download_open_data"), "sandbox_python": _tool("sandbox_python")}
    out = Runtime(store).run(store.create_mission("g", {}), goal_program("g", tools, brain))
    sub = next(h for h in out["result"]["evidence"] if h.get("kind") == "subagent")
    assert "download_open_data" in sub["tools"]
    assert any(e.get("tool") == "download_open_data" and "result" in e for e in sub.get("evidence") or [])
