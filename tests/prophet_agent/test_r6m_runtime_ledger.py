"""Round 6m (2026-11-23): the step and tool-call ledgers keep the first finished result, and a sweep frees only turns that
really were abandoned.

Audit (runtime store.py): finish_step overwrote a finished step (ON CONFLICT DO UPDATE with no status guard), tool_set
let a late write downgrade a finished call, and interrupt_orphans marked a freshly created turn as interrupted. The
last one freed its notebook while the person's turn had not started, so a second turn could begin beside it.
All missions here are synthetic.
"""
import time

import pytest

from prophet_agent.runtime.runner import Ctx, Runtime, Tool
from prophet_agent.runtime.store import NotebookBusy, Store, idem_key


class Crash(BaseException):
    """Simulates a process kill: not caught by the runtime's ``except Exception`` handlers."""


@pytest.fixture()
def store(tmp_path):
    s = Store(tmp_path / "prophet.sqlite")
    yield s
    s.close()


def test_a_finished_step_is_not_overwritten_by_a_later_write(store):
    mid = store.create_mission("forecast", {})
    store.finish_step(mid, 0, "decide", "first")
    store.finish_step(mid, 0, "decide", "second")
    assert store.step_output(mid, 0) == "first", "the first finished result is the one replays return"


def test_a_late_write_cannot_downgrade_a_finished_tool_call(store):
    mid = store.create_mission("forecast", {})
    key = idem_key(mid, 0, "probe", {})
    store.tool_start(mid, 0, "probe", {}, key)
    store.tool_set(key, "done", {"ok": 1})
    store.tool_set(key, "failed", error="late write")
    rec = store.tool_get(key)
    assert rec.status == "done" and rec.result == {"ok": 1}


def test_a_turn_that_has_not_started_is_not_swept_as_orphaned(store):
    mid = store.create_mission("forecast", {})
    assert store.interrupt_orphans() == [], "a turn created a moment ago is about to start, not abandoned"
    assert store.mission(mid)["status"] == "created"


def test_a_turn_created_long_ago_and_never_started_is_swept_so_its_notebook_is_freed(store):
    mid = store.create_mission("forecast", {}, notebook_id="nb-old")
    store.conn.execute("UPDATE missions SET updated=? WHERE id=?", (time.time() - 3600, mid))
    assert store.interrupt_orphans() == [mid]
    store.create_mission("next question", {}, notebook_id="nb-old")       # the notebook is free again


def test_a_sweep_does_not_free_a_notebook_whose_turn_is_about_to_start(store):
    store.create_mission("first question", {}, notebook_id="nb-1")
    store.interrupt_orphans()
    with pytest.raises(NotebookBusy):
        store.create_mission("second question", {}, notebook_id="nb-1")


def test_each_finished_step_is_recorded_once_across_a_resume(store):
    mid = store.create_mission("forecast", {})
    runs = {"b": 0}

    def program(ctx: Ctx):
        ctx.step("a", lambda c: 1)

        def b(c):
            runs["b"] += 1
            if runs["b"] == 1:
                raise Crash()                       # a kill after the first step: nothing after it is recorded
            return 2
        return ctx.step("b", b)

    with pytest.raises(Crash):
        Runtime(store, owner="r1").run(mid, program)
    store.conn.execute("UPDATE missions SET lease_until=0 WHERE id=?", (mid,))   # the killed process's lease lapses
    assert Runtime(store, owner="r2").run(mid, program)["status"] == "completed"
    done = [e for e in store.events(mid) if e["kind"] == "step.done"]
    assert [e["payload"]["name"] for e in done] == ["a", "b"], "one step.done per step, none repeated on resume"
    assert runs["b"] == 2
