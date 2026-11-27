"""Round 3, runtime goals 2 and 3: repeat detection (A5-09, A3-10) and argument errors not charged (A5-12).

Measured 2026-11-22: identical (tool, arguments) calls recurred 21 times in one electricity mission (panel_backtest 6,
sandbox_python 9), and the planner obeyed the note that "the same call again returns the same result". An argument error
("unknown arguments", "must be one of") was charged as a tool call and the same invalid call was sent again.
"""
from __future__ import annotations

import pytest

from prophet_agent.runtime.goal import GoalConfig, goal_program
from prophet_agent.runtime.runner import Runtime, Tool
from prophet_agent.runtime.store import Store


@pytest.fixture()
def store(tmp_path):
    s = Store(tmp_path / "repeat.sqlite")
    yield s
    s.close()


class Scripted:
    """Planner and evaluator decisions from two queues. Asking past the end of a queue is a failure, so a test shows
    exactly how many decisions the run needed."""

    def __init__(self, plans, evals=()):
        self.plans, self.evals = list(plans), list(evals)
        self.calls = {"plan": 0, "eval": 0}
        self.plan_messages = []

    def decide(self, messages, schema, name):
        self.calls[name] += 1
        if name == "plan":
            self.plan_messages.append(messages[-1]["content"])
            queue = self.plans
        else:
            queue = self.evals
        if not queue:
            raise AssertionError(f"brain asked for {name} but the script is exhausted")
        return queue.pop(0)


def call(tool, args):
    return {"action": "call_tool", "tool": tool, "args": args, "rationale": "work"}


def finish():
    return {"action": "finish", "rationale": "done"}


def met(answer="ok"):
    return {"verdict": "met", "reason": "enough", "answer": answer}


def counting(fn=None, idempotent=True):
    counter = {"n": 0}

    def run(args, key):
        counter["n"] += 1
        return fn(args) if fn else {"ok": True, "value": args.get("k", 0)}

    return Tool("a", run, idempotent=idempotent), counter


def run(store, program, budget=None):
    mid = store.create_mission("analyse", budget or {"max_tool_calls": 20})
    return mid, Runtime(store, owner="r1").run(mid, program)


# -- goal 2: repeat detection --------------------------------------------------------------------------------------------

def test_an_identical_call_is_not_run_again_and_points_to_the_earlier_result(store):
    tool, counter = counting()
    brain = Scripted([call("a", {"k": 1}), call("a", {"k": 1}), finish()], [met()])
    mid, out = run(store, goal_program("analyse", {"a": tool}, brain))
    assert counter["n"] == 1, "the second identical call must not run the tool again"
    first, second = out["result"]["evidence"][0], out["result"]["evidence"][1]
    assert second.get("repeat_of") == 1, second
    assert "same result" in second["note"] and "not run again" in second["note"], second["note"]
    assert store.count_tool_calls(mid) == 1, "a pointer is not a tool call, so it is not charged"
    assert first["result"] == {"ok": True, "value": 1}


def test_an_identical_web_call_in_one_mission_is_answered_with_the_earlier_result(store):
    # VN Unemployment (2026-11-27): identical searches ran back to back and the stall rule stopped the mission. Within one
    # mission the web does not change enough to be worth a second identical call, even after a sandbox cell.
    counter = {"n": 0}

    def search(args, key):
        counter["n"] += 1
        return {"ok": True, "hits": counter["n"]}

    tool = Tool("web_search", search, idempotent=True)
    brain = Scripted([call("web_search", {"q": "wti"}), call("web_search", {"q": "wti"}), finish()], [met()])
    run(store, goal_program("analyse", {"web_search": tool}, brain))
    assert counter["n"] == 1


def test_an_identical_failing_call_is_not_repeated_and_the_repeat_says_the_same_error(store):
    def broken(args, key):
        raise RuntimeError("upstream timed out")

    tool = Tool("a", broken, idempotent=True)
    brain = Scripted([call("a", {"k": 1}), call("a", {"k": 1}), finish()], [met()])
    mid, out = run(store, goal_program("analyse", {"a": tool}, brain))
    ev = out["result"]["evidence"]
    assert "upstream timed out" in ev[0]["error"]
    assert ev[1].get("repeat_of") == 1 and "upstream timed out" in ev[1]["error"], ev[1]


def test_the_repeat_note_is_stored_so_a_resumed_run_takes_the_same_path(store):
    tool, counter = counting()
    brain = Scripted([call("a", {"k": 1}), call("a", {"k": 1}), finish()], [met()])
    mid, first = run(store, goal_program("analyse", {"a": tool}, brain))

    class Silent:
        """A brain that must not be asked anything: every decision is already in the ledger."""
        def decide(self, messages, schema, name):
            raise AssertionError(f"a resumed run asked for {name}")

    again = Runtime(store, owner="r2").run(mid, goal_program("analyse", {"a": tool}, Silent()))
    assert again["result"]["evidence"] == first["result"]["evidence"]
    assert counter["n"] == 1


# -- goal 3: argument errors are not charged, and an identical one is caught by goal 2 -----------------------------

def _strict(args, key):
    if set(args) - {"k"}:
        raise ValueError(f"unknown arguments {sorted(set(args) - {'k'})}; allowed: k")
    return {"ok": True, "value": args.get("k", 0)}


def test_an_argument_error_is_not_charged_against_the_tool_allowance(store):
    tool = Tool("a", _strict, idempotent=True)
    brain = Scripted([call("a", {"x": 1}), call("a", {"k": 2}), finish()], [met()])
    mid, out = run(store, goal_program("analyse", {"a": tool}, brain, GoalConfig(tool_budget=1)),
                   budget={"max_tool_calls": 1})
    assert out["result"]["status"] == "met", (
        "the rejected call used no allowance, so the lead still has its one call for the valid one")
    assert out["result"]["evidence"][0].get("rejected") is True
    assert store.count_tool_calls(mid) == 1


def test_the_same_invalid_call_repeated_is_caught_by_the_repeat_rule(store):
    counter = {"n": 0}

    def invalid(args, key):
        counter["n"] += 1
        raise ValueError("unknown arguments ['x']; allowed: k")

    tool = Tool("a", invalid, idempotent=True)
    brain = Scripted([call("a", {"x": 1}), call("a", {"x": 1}), finish()], [met()])
    mid, out = run(store, goal_program("analyse", {"a": tool}, brain))
    assert counter["n"] == 1, "an identical invalid call is not sent to the tool a second time"
    assert out["result"]["evidence"][1].get("repeat_of") == 1
    assert store.count_tool_calls(mid) == 0
