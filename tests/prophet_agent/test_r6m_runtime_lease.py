"""Round 6m (2026-11-23): a write to a mission is fenced by the lease the writing runtime holds.

Audit (runtime store.py, lease fence): the fence compared the lease owner in the database with one store-wide owner
recorded per mission. Two defects followed. (1) interrupt_orphans dropped that record for a lapsed lease, so a stalled
runtime's later writes were no longer checked at all. (2) When another runtime took the mission over through the same
store instance, the record moved to the new owner, and the stalled runtime's writes matched it and went through.
All missions here are synthetic.
"""
import pytest

from prophet_agent.runtime.runner import Ctx, Runtime, Tool
from prophet_agent.runtime.store import LeaseLost, Store


@pytest.fixture()
def path(tmp_path):
    return tmp_path / "prophet.sqlite"


def _lapse(store, mid):
    store.conn.execute("UPDATE missions SET lease_until=0 WHERE id=?", (mid,))


def test_a_stalled_runtime_stays_fenced_after_its_own_process_sweeps_the_lease(path):
    a, b = Store(path), Store(path)
    try:
        mid = a.create_mission("forecast", {})
        a.acquire_lease(mid, "a", ttl_s=3600)
        a.set_status(mid, "running")
        _lapse(a, mid)                                    # a's runtime stalls past its lease
        assert a.interrupt_orphans() == [mid]             # the sweep runs in a's own process
        b.acquire_lease(mid, "b", ttl_s=3600)             # another runtime takes the mission over
        b.finish_step(mid, 0, "decide", "from b")
        with pytest.raises(LeaseLost):
            a.finish_step(mid, 0, "decide", "from a")     # the stalled runtime must not overwrite it
        assert b.step_output(mid, 0) == "from b"
    finally:
        a.close()
        b.close()


def test_a_stalled_runtime_cannot_record_a_step_after_a_takeover_on_the_same_store(path):
    s = Store(path)
    try:
        mid = s.create_mission("forecast", {})

        def program(ctx: Ctx):
            def slow(c):
                # The lease lapses while this step runs, and another runtime on the same store takes the mission.
                _lapse(s, mid)
                s.acquire_lease(mid, "new", ttl_s=3600)
                return "from the stalled runtime"
            ctx.step("decide", slow)
            return {"status": "met", "answer": "from the stalled runtime"}

        result = Runtime(s, owner="old").run(mid, program)
        assert result == {"status": "lease_lost"}
        assert s.step_record(mid, 0) is None, "the stalled runtime recorded no step"
        assert s.mission(mid)["answer"] is None, "and wrote no answer"
    finally:
        s.close()


def test_a_stalled_tool_result_is_not_recorded_after_a_takeover_on_the_same_store(path):
    s = Store(path)
    try:
        mid = s.create_mission("forecast", {})

        def slow_tool(args, key):
            _lapse(s, mid)
            s.acquire_lease(mid, "new", ttl_s=3600)
            return {"ok": True}

        tool = Tool("probe", slow_tool, idempotent=False)
        result = Runtime(s, owner="old").run(mid, lambda ctx: ctx.step("x", lambda c: c.call(tool, {})))
        assert result == {"status": "lease_lost"}
        rows = s.conn.execute("SELECT status FROM tool_calls WHERE mission_id=?", (mid,)).fetchall()
        assert [r["status"] for r in rows] == ["started"], "the stalled runtime did not record the result"
    finally:
        s.close()


def test_a_takeover_seen_through_a_second_store_instance_is_still_fenced(path):
    a, b = Store(path), Store(path)
    try:
        mid = a.create_mission("forecast", {})

        def program(ctx: Ctx):
            def slow(c):
                _lapse(a, mid)
                b.acquire_lease(mid, "new", ttl_s=3600)
                return 1
            return ctx.step("x", slow)

        assert Runtime(a, owner="old").run(mid, program) == {"status": "lease_lost"}
        assert a.step_record(mid, 0) is None
    finally:
        a.close()
        b.close()


def test_a_service_write_after_the_lease_lapsed_is_still_allowed(path):
    # The service's cancel writes come from outside the runner and must keep going through once nobody holds the lease.
    a = Store(path)
    try:
        mid = a.create_mission("forecast", {})
        a.acquire_lease(mid, "a", ttl_s=3600)
        _lapse(a, mid)
        a.interrupt_orphans()
        a.set_status(mid, "cancelled")
        assert a.mission(mid)["status"] == "cancelled"
    finally:
        a.close()
