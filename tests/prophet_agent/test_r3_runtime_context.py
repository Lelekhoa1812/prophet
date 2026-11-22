"""Round 3, runtime goal 7 (A5-03): deterministic context editing. The model summarises only at part boundaries.

Measured (six live runs, 2026-11-22): compaction took 1,728 s of 6,667 s, 26 percent of the run, and ran about every fourth tool
call, each a model call that folded a few observations into the memo. Now an observation that leaves the recent window becomes a
short stub (tool, arguments, key result fields, and its step, so read_observation can page the full result back). The stubs are
a pure function of the stored history. The model writes a memo only at a part boundary, and only when the view is over its limit.
"""
from __future__ import annotations

import pytest

from prophet_agent.runtime.goal import GoalConfig, goal_program
from prophet_agent.runtime.runner import Runtime, Tool
from prophet_agent.runtime.store import Store

SEG = 12          # steps per part
STEPS = 30        # long enough to cross two part boundaries (12 and 24)


@pytest.fixture()
def store(tmp_path):
    s = Store(tmp_path / "context.sqlite")
    yield s
    s.close()


class Long:
    """A brain with a long plan of identical calls. write() is the model's memo writer; the test counts its calls."""

    def __init__(self, plans):
        self.plans = list(plans)
        self.plan_messages = []
        self.writes = 0

    def decide(self, messages, schema, name):
        if name == "plan":
            self.plan_messages.append(messages[-1]["content"])
            if not self.plans:
                return {"action": "finish", "rationale": "done"}
            return self.plans.pop(0)
        return {"verdict": "met", "reason": "enough", "answer": "done"}

    def write(self, messages):
        # The model's free-text writer serves two jobs: the memo (compaction) and the answer composer. Only the memo counts here.
        if "New observations to fold in" in messages[-1]["content"]:
            self.writes += 1
            return "MODEL MEMO of the earlier parts"
        return "composed answer"


def _tool():
    def run(args, key):
        return {"ok": True, "rows": 100 + args["k"], "label": "series " + "z" * 1200, "k": args["k"]}
    return Tool("a", run, idempotent=True)


def _plans(n):
    return [{"action": "call_tool", "tool": "a", "args": {"k": k}, "rationale": "work"} for k in range(n)]


def _run(store, brain):
    program = goal_program("analyse the series", {"a": _tool()}, brain,
                           GoalConfig(max_iterations=SEG, view_chars=8000, keep_recent=4))
    mid = store.create_mission("analyse the series", {"max_tool_calls": 100})
    return mid, Runtime(store, owner="r1").run(mid, program)


def test_no_model_compaction_happens_between_part_boundaries(store):
    brain = Long(_plans(STEPS))
    mid, out = _run(store, brain)
    compact = [s for s in store.steps(mid) if s["name"].startswith("compact:")]
    for s in compact:
        step_index = int(s["name"].split(":")[2])     # compact:d0:<step>:<cut>
        assert step_index % SEG == 0, ("a model memo was written between part boundaries", s["name"])
    assert brain.writes == len(compact)
    assert brain.writes <= STEPS // SEG, "at most one model memo per part boundary"


def test_the_planner_sees_stubs_with_steps_and_no_model_memo_within_the_first_part(store):
    brain = Long(_plans(SEG - 1))          # SEG steps in all, the last a finish: no boundary is reached
    _run(store, brain)
    late = brain.plan_messages[SEG - 2]
    assert "read_observation" in late and "step 1" in late, "an old observation is a stub that names its step"
    assert "MODEL MEMO" not in late, "no model memo inside the first part"
    assert brain.writes == 0


def test_a_resumed_run_sees_the_same_compacted_view_and_asks_no_model(store):
    brain = Long(_plans(STEPS))
    mid, first = _run(store, brain)

    class Silent:
        writes = 0

        def decide(self, messages, schema, name):
            raise AssertionError(f"a resumed run asked for {name}")

        def write(self, messages):
            raise AssertionError("a resumed run asked the model to write a memo")

    again = Runtime(store, owner="r2").run(mid, goal_program(
        "analyse the series", {"a": _tool()}, Silent(), GoalConfig(max_iterations=SEG, view_chars=8000, keep_recent=4)))
    assert again["result"]["evidence"] == first["result"]["evidence"]
