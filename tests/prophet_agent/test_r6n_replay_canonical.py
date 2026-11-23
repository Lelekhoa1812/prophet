"""A step or tool result must be the same value on the first run and on a replay. Before the fix, the first run returned the
Python object the program made, while a replay returned the JSON round trip with default=str applied: a numpy int came back
as the string '3', tuples came back as lists, and int dict keys came back as strings. A resumed program could then fail
(for example n + 1 on '3'). The runner now canonicalises a result once, before the first return and before it is stored.
"""
from __future__ import annotations

import datetime as dt

import numpy as np
import pytest

from prophet_agent.runtime.runner import Ctx, Runtime, Tool, to_json_value
from prophet_agent.runtime.store import Store


class Crash(BaseException):
    """Simulates a process kill: not caught by the runtime, so the lease and status stay as they were."""


@pytest.fixture()
def store(tmp_path):
    s = Store(tmp_path / "prophet.sqlite")
    yield s
    s.close()


def _mission(store):
    return store.create_mission("forecast demand", {})


def _rerun(store, mid, program):
    """Run the same program again on a new runtime, as a restarted process would: the ledger answers every step."""
    store.set_status(mid, "created")
    return Runtime(store, owner="r2").run(mid, program)


def test_a_numpy_int_and_an_int_keyed_dict_are_identical_on_first_run_and_replay(store):
    mid = _mission(store)
    seen = []

    def program(ctx: Ctx):
        value = ctx.step("s", lambda c: {"n": np.int64(3), "pairs": (1, 2), "by_k": {1: "a", 2: "b"}})
        seen.append(value)
        return value

    assert Runtime(store, owner="r1").run(mid, program)["status"] == "completed"
    assert _rerun(store, mid, program)["status"] == "completed"
    first, replay = seen
    assert first == replay
    assert type(first["n"]) is int and first["n"] + 1 == 4, "the first run must give a real int, not a string"
    assert first["pairs"] == [1, 2] and first["by_k"] == {"1": "a", "2": "b"}


def test_a_replayed_numpy_int_can_be_used_in_arithmetic_after_a_resume(store):
    """The failure the defect report describes: a resumed program does arithmetic on a stored numpy value."""
    mid = _mission(store)
    results = []

    def program(ctx: Ctx):
        n = ctx.step("count", lambda c: np.int64(3))
        results.append(n + 1)
        return n + 1

    Runtime(store, owner="r1").run(mid, program)
    assert _rerun(store, mid, program)["result"] == 4
    assert results == [4, 4]


def test_a_tool_result_seen_before_a_crash_is_the_same_value_after_resume(store):
    """The tool's result is recorded, the step then crashes. On resume the step runs again and the call returns the
    recorded result, which must equal what the first run saw."""
    mid = _mission(store)
    seen = []

    def fetch(args, key):
        return {"n": np.int64(7), "ids": (4, 5), "by_k": {10: "x"}}

    tool = Tool("fetch", fetch)

    def program(ctx: Ctx):
        def step(c):
            value = c.call(tool, {"id": 1})
            seen.append(value)
            if len(seen) == 1:
                raise Crash()
            return value
        return ctx.step("s", step)

    with pytest.raises(Crash):
        Runtime(store, owner="r1").run(mid, program)
    store.conn.execute("UPDATE missions SET lease_until=0 WHERE id=?", (mid,))
    assert Runtime(store, owner="r2").run(mid, program)["status"] == "completed"
    first, resumed = seen
    assert first == resumed
    assert type(first["n"]) is int and first["ids"] == [4, 5] and first["by_k"] == {"10": "x"}


def test_a_numpy_array_is_returned_as_a_list_on_first_run_and_replay(store):
    mid = _mission(store)
    seen = []

    def program(ctx: Ctx):
        value = ctx.step("arr", lambda c: np.array([[1.5, 2.0], [3.0, 4.0]]))
        seen.append(value)
        return value

    Runtime(store, owner="r1").run(mid, program)
    _rerun(store, mid, program)
    assert seen[0] == seen[1] == [[1.5, 2.0], [3.0, 4.0]]
    assert type(seen[0]) is list


def test_a_value_with_no_json_form_is_refused_clearly_and_not_stored(store):
    mid = _mission(store)
    result = Runtime(store, owner="r1").run(mid, lambda ctx: ctx.step("when", lambda c: dt.date(2026, 1, 2)))
    assert result["status"] == "failed"
    assert "cannot be stored as JSON" in result["error"] and "date" in result["error"]
    assert store.step_record(mid, 0) is None, "a refused value must not be written to the ledger"


@pytest.mark.parametrize("value, expected", [
    (np.int64(3), 3),
    (np.float64(1.5), 1.5),
    (np.bool_(True), True),
    (np.str_("a"), "a"),
    ((1, (2, 3)), [1, [2, 3]]),
    ({2: "b", None: "n", True: "t"}, {"2": "b", "null": "n", "true": "t"}),
    ({"b": 1, "a": 2}, {"a": 2, "b": 1}),
])
def test_the_canonical_form_follows_the_json_rules(value, expected):
    canon = to_json_value(value)
    assert canon == expected
    assert type(canon) is type(expected)


def test_two_keys_that_become_the_same_json_key_are_refused():
    with pytest.raises(TypeError, match="same JSON key"):
        to_json_value({1: "a", "1": "b"})
