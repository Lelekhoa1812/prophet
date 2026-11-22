"""Round 4 C4, audit A3-11: a repeated failure of one error class is probed at a streak of two, and a sandbox cell that imports a
tool name as a Python module is told that the name is a tool to call with call_tool.

Motivation vs Logic
Motivation: live retail rep4 sent four failing cells, three of them the same TypeError, before any probe; live runs also
wrote `import backtest_series` and `prophet_stats.panel_backtest(...)`, which fail because a tool is not a Python module.
Logic: the budget note reads the trailing failures and their error class (the exception name before the colon). Two of the
same class ask for the probe. The sandbox result is checked for "No module named 'X'" or "has no attribute 'X'" where X is a
tool name, and a hint is added to the observation (the stored ledger row is not changed, so replay is unaffected).
"""
from __future__ import annotations

import pytest

from prophet_agent.runtime.goal import GoalConfig, _budget_note, goal_program
from prophet_agent.runtime.runner import Runtime, Tool
from prophet_agent.runtime.store import Store

GOAL = "forecast the daily sales for the next 30 days"


def _failed_cell(message: str) -> dict:
    return {"kind": "tool", "tool": "sandbox_python", "args": {"code": "x"},
            "result": {"ok": False, "stderr_tail": f"Traceback (most recent call last):\n  File \"program.py\", line 3\n{message}"}}


def test_two_failures_of_one_error_class_get_the_probe_instruction():
    history = [_failed_cell("TypeError: list indices must be integers or slices, not str"),
               _failed_cell("TypeError: list indices must be integers or slices, not str")]
    note = _budget_note(2, GoalConfig(max_iterations=12), history)
    assert "print(type(x))" in note, note


def test_two_failures_of_different_classes_do_not_get_the_probe_instruction():
    history = [_failed_cell("KeyError: 'outlier_pct'"), _failed_cell("TypeError: list indices must be integers or slices, not str")]
    note = _budget_note(2, GoalConfig(max_iterations=12), history)
    assert "print(type(x))" not in note, note
    assert "Do not repeat them" in note, note


def test_a_tool_name_imported_as_a_module_gets_the_call_tool_hint():
    from prophet_agent.runtime.goal import tool_module_hint
    result = {"ok": False, "stderr_tail": "ModuleNotFoundError: No module named 'backtest_series'"}
    hint = tool_module_hint(result, {"backtest_series", "inspect_data"})
    assert hint is not None and "call_tool" in hint and "backtest_series" in hint, hint


def test_a_tool_name_used_as_a_module_attribute_gets_the_hint():
    from prophet_agent.runtime.goal import tool_module_hint
    result = {"ok": False, "stderr_tail": "AttributeError: module 'prophet_stats' has no attribute 'panel_backtest'"}
    assert tool_module_hint(result, {"panel_backtest"}) is not None


def test_an_ordinary_missing_package_gets_no_tool_hint():
    from prophet_agent.runtime.goal import tool_module_hint
    result = {"ok": False, "stderr_tail": "ModuleNotFoundError: No module named 'tabulate'"}
    assert tool_module_hint(result, {"backtest_series"}) is None


class Scripted:
    def __init__(self, plans, evals):
        self.plans, self.evals = list(plans), list(evals)

    def decide(self, messages, schema, name):
        queue = self.plans if name == "plan" else self.evals
        if not queue:
            raise AssertionError(f"brain asked for {name} but the script is exhausted")
        return queue.pop(0)


@pytest.fixture()
def store(tmp_path):
    s = Store(tmp_path / "hint.sqlite")
    yield s
    s.close()


def test_the_hint_reaches_the_planner_in_the_observation_and_not_in_the_ledger(store):
    def cell(args, key):
        return {"ok": False, "returncode": 1, "timed_out": False, "stdout": "", "stderr": "",
                "stderr_tail": "ModuleNotFoundError: No module named 'backtest_series'", "run_id": "r", "outputs": []}

    def other(args, key):
        return {"ok": True}

    tools = {"sandbox_python": Tool("sandbox_python", cell, idempotent=True),
             "backtest_series": Tool("backtest_series", other, idempotent=True)}
    brain = Scripted(plans=[{"action": "call_tool", "tool": "sandbox_python", "args": {"code": "import backtest_series"},
                             "rationale": "try"}, {"action": "finish", "rationale": "done"}],
                     evals=[{"verdict": "met", "reason": "enough", "answer": "Done."}])
    goal = "summarise the uploaded file"       # no forecast words, so no checklist gate runs
    mid = store.create_mission(goal, {"max_tool_calls": 20})
    out = Runtime(store, owner="h1").run(mid, goal_program(goal, tools, brain, GoalConfig(max_iterations=6)))
    obs = [h for h in out["result"]["evidence"] if h.get("kind") == "tool"]
    assert obs and "call_tool" in obs[0].get("hint", ""), obs
    stored = store.tool_results(mid, "sandbox_python")
    assert stored and all("hint" not in r for r in stored), "the hint was written into the ledger"
