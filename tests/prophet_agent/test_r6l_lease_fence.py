"""Round 6l (2026-11-23): a runtime that lost its lease cannot write to the mission.

Audit (runtime store.py:409-437, HIGH): writes did not check the lease owner, so a runtime whose lease had been taken over
kept writing events, steps and results beside the new owner. Writes to a mission this process leased are now conditional
on the lease still being ours, read in the same transaction. A takeover after the program (the answer, the final status)
also ends as lease_lost, not as an exception out of run(). All missions here are synthetic.
"""
import time

import pytest

from prophet_agent.runtime.runner import Ctx, Runtime
from prophet_agent.runtime.store import LeaseLost, Store


@pytest.fixture()
def store(tmp_path):
    s = Store(tmp_path / "prophet.sqlite")
    yield s
    s.close()


def _takeover(store, mid):
    store.conn.execute("UPDATE missions SET lease_owner='intruder', lease_until=? WHERE id=?",
                       (time.time() + 3600, mid))


def test_writes_by_the_lease_owner_go_through(store):
    mid = store.create_mission("forecast", {})
    store.acquire_lease(mid, "me", ttl_s=3600)
    store.append_event(mid, "note", {"ok": True})
    store.set_status(mid, "running")
    store.note_put(mid, "lead", "plan", "step one")
    assert store.mission(mid)["status"] == "running"


@pytest.mark.parametrize("write", [
    lambda s, mid: s.append_event(mid, "x", {}),
    lambda s, mid: s.set_status(mid, "completed"),
    lambda s, mid: s.set_answer(mid, {"status": "met"}),
    lambda s, mid: s.set_budget(mid, {}),
    lambda s, mid: s.finish_step(mid, 0, "a", 1),
    lambda s, mid: s.note_put(mid, "lead", "n", "text"),
    lambda s, mid: s.open_question(mid, "key-1", {}),
    lambda s, mid: s.tool_start(mid, 0, "probe", {}, "key-1"),
])
def test_a_write_after_a_takeover_is_refused(store, write):
    mid = store.create_mission("forecast", {})
    store.acquire_lease(mid, "me", ttl_s=3600)
    _takeover(store, mid)
    with pytest.raises(LeaseLost):
        write(store, mid)


def test_a_tool_result_write_after_a_takeover_is_refused(store):
    mid = store.create_mission("forecast", {})
    store.acquire_lease(mid, "me", ttl_s=3600)
    store.tool_start(mid, 0, "probe", {}, "key-2")
    _takeover(store, mid)
    with pytest.raises(LeaseLost):
        store.tool_set("key-2", "done", {"v": 1})


def test_writes_to_a_mission_this_process_does_not_lease_are_not_fenced(store):
    # The service writes such as a cancel come from outside the runner; they keep their old behaviour.
    mid = store.create_mission("forecast", {})
    _takeover(store, mid)
    store.set_status(mid, "cancelled")
    assert store.mission(mid)["status"] == "cancelled"


def test_a_takeover_after_the_program_ends_as_lease_lost_not_an_exception(store):
    mid = store.create_mission("forecast", {})

    def program(ctx: Ctx):
        ctx.step("a", lambda c: 1)
        _takeover(store, mid)                 # the lease is lost after the last step, before the answer is written
        return {"status": "met", "answer": 2}

    result = Runtime(store, owner="old").run(mid, program)
    assert result == {"status": "lease_lost"}
    assert store.mission(mid)["answer"] is None, "the lost runtime wrote no answer"
