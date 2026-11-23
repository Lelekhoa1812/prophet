"""A refused argument list is not a stall.

Defect (docs/prophet/GOAL.md): a call whose arguments were refused before it ran is marked ``rejected`` and is not charged to
the allowance (round 3, A5-12). The stall rule nevertheless counted it: each refusal carries an error, so the same invalid call
sent over and over reached ERROR_STALL and stopped the mission as "stalled", which charges the planner for a slip it can fix.
"""
from __future__ import annotations

from prophet_agent.runtime.goal import ERROR_STALL, _stalled


def _refused(args: dict) -> dict:
    return {"kind": "tool", "tool": "sandbox_python", "args": args,
            "error": "unknown arguments ['colour']; allowed: code, carry", "rejected": True}


def _failed(args: dict) -> dict:
    return {"kind": "tool", "tool": "sandbox_python", "args": args, "error": "ZeroDivisionError: division by zero"}


def test_identical_refused_arguments_are_not_a_stall_by_error_run():
    history = [_refused({"code": "print(1)", "colour": "red"}) for _ in range(ERROR_STALL + 2)]
    assert _stalled(history, 3) is False


def test_identical_refused_arguments_are_not_a_stall_by_window():
    history = [_refused({"code": "print(1)", "colour": "red"}) for _ in range(3)]
    assert _stalled(history, 3) is False


def test_refusals_between_real_failures_do_not_reset_or_add_to_the_error_run():
    history = []
    for i in range(ERROR_STALL - 1):
        history.append(_failed({"code": f"1/{i}"}))
        history.append(_refused({"code": "x", "colour": "red"}))
    # ERROR_STALL - 1 identical-outcome real failures, with refusals between them: still short of the stall.
    assert _stalled(history, 3) is False


def test_identical_real_failures_still_stall():
    history = [_failed({"code": "1/0"}) for _ in range(ERROR_STALL)]
    assert _stalled(history, 3) is True
