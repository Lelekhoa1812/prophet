"""Round 5: a model call with too little of the segment's time left is a checkpoint, never a failed mission.

Live electricity rep4 (measured, 2026-11-23): at 899 s of a 900 s allowance a planner call started with a 1 s budget, the
transport raised Exhausted ("budget spent while reading") 1.5 s later, and the mission ended failed with no answer after 49
delegated calls. Round 4 (A5-06) had capped each call at the seconds left, floored at 1 s.
"""
from __future__ import annotations

import json

import pytest

from prophet_agent.nim.errors import Exhausted
from prophet_agent.runtime.goal import BUDGET_S, MIN_CALL_S, NimBrain
from prophet_agent.runtime.runner import BudgetExceeded


class Client:
    def __init__(self, fail=False):
        self.fail, self.calls = fail, []

    def complete(self, models, messages, **kw):
        self.calls.append(kw)
        if self.fail:
            raise Exhausted("nim budget spent before a usable answer; last: m: budget spent while reading")

        class R:
            text = json.dumps({"action": "finish"})
        return R()


def _brain(client, left):
    b = NimBrain(client, ["m"])
    b.bind(None, None, lambda: left)
    return b


def test_no_call_starts_with_less_than_the_minimum_left():
    client = Client()
    with pytest.raises(BudgetExceeded) as exc:
        _brain(client, 1.0).decide([], {}, "plan")
    assert exc.value.which == "max_seconds" and client.calls == []


def test_a_call_with_time_left_is_capped_at_the_time_left():
    client = Client()
    assert _brain(client, 200.0).decide([], {}, "plan") == {"action": "finish"}
    assert client.calls[0]["budget_s"] == 200.0


def test_a_call_cut_short_by_the_allowance_is_a_checkpoint():
    # remaining() is read again after the failure; a clock that has run down below the minimum means the allowance ran out
    ticks = iter([MIN_CALL_S + 5, 2.0])
    b = NimBrain(Client(fail=True), ["m"])
    b.bind(None, None, lambda: next(ticks))
    with pytest.raises(BudgetExceeded):
        b.write([])


def test_a_provider_failure_with_time_to_spare_is_still_a_failure():
    with pytest.raises(Exhausted):
        _brain(Client(fail=True), BUDGET_S + 100).decide([], {}, "plan")


def test_without_an_allowance_nothing_changes():
    client = Client()
    b = NimBrain(client, ["m"])
    b.decide([], {}, "plan")
    assert "budget_s" not in client.calls[0]
