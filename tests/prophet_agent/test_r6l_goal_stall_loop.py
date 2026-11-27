"""Round 6l, goal audit item 1: a loop of two calls taking turns (A, B, A) is a stall; ordinary progress is not.

Synthetic tools only. The window of 3 is the default GoalConfig.stall_window.
"""
from __future__ import annotations

import pytest

from prophet_agent.runtime import goal as G
from prophet_agent.runtime.goal import GoalConfig, goal_program
from prophet_agent.runtime.store import Store
from tests.prophet_agent.test_goal_loop import ScriptedBrain, _tool, call, run


@pytest.fixture()
def store(tmp_path):
    s = Store(tmp_path / "stall_loop.sqlite")
    yield s
    s.close()


def _obs(tool, result, args=None):
    return {"kind": "tool", "tool": tool, "args": args or {}, "result": result}


def test_a_then_b_then_a_with_the_same_outcome_is_a_stall():
    hist = [_obs("a", {"n": 1}), _obs("b", {"m": 2}), _obs("a", {"n": 1})]
    assert G._stalled(hist, 3) is True


def test_two_calls_taking_turns_for_a_longer_window_is_a_stall():
    hist = [_obs("a", {"n": 1}), _obs("b", {"m": 2}), _obs("a", {"n": 1}), _obs("b", {"m": 2})]
    assert G._stalled(hist, 4) is True


def test_a_then_b_then_a_whose_result_changed_is_progress():
    # The second A read a different result, so the world moved between the calls: not a loop.
    hist = [_obs("a", {"n": 1}), _obs("b", {"m": 2}), _obs("a", {"n": 2})]
    assert G._stalled(hist, 3) is False


def test_a_then_a_then_b_and_a_then_b_then_b_are_not_stalls():
    assert G._stalled([_obs("a", {"n": 1}), _obs("a", {"n": 1}), _obs("b", {"m": 2})], 3) is False
    assert G._stalled([_obs("a", {"n": 1}), _obs("b", {"m": 2}), _obs("b", {"m": 2})], 3) is False


def test_three_different_calls_are_not_a_stall():
    hist = [_obs("a", {"n": 1}), _obs("b", {"m": 2}), _obs("c", {"k": 3})]
    assert G._stalled(hist, 3) is False


def test_a_window_of_two_does_not_read_any_two_different_calls_as_a_loop():
    assert G._stalled([_obs("a", {"n": 1}), _obs("b", {"m": 2})], 2) is False


@pytest.mark.usefixtures("old_stop_gate")
def test_the_loop_stops_after_a_then_b_then_a_instead_of_running_on(store):
    # Three planned calls only: a fourth planner decision would exhaust the script and fail the mission, so a stop here is the
    # stall detector firing on A, B, A.
    brain = ScriptedBrain(plans=[call("a"), call("b"), call("a")], evals=[{"verdict": "not_met", "reason": "stuck"}])
    tools = {"a": _tool("a", lambda args, key: {"same": 1}), "b": _tool("b", lambda args, key: {"other": 2})}
    _, out = run(store, goal_program("g", tools, brain, GoalConfig(max_iterations=8, stall_window=3)))
    assert out["status"] == "stalled", "a stalled goal must not be recorded as a completed mission"
    assert out["result"]["status"] == "stalled" and out["result"]["iterations"] == 3
    assert brain.calls["plan"] == 3
