"""A cancel must be recorded and reported cleanly, whichever process holds the mission.

Before the fix, cancel() closed a mission itself whenever this service had no live thread for it. That raised LeaseLost
(unmapped, so a 500 at the API) when this service still had an old lease on record, and when another process was running
the mission it marked the mission cancelled while that run kept going. A cancel that arrived while a run was pausing was
also left as waiting_user with the request unactioned.
"""
from __future__ import annotations

import threading

from prophet_agent.api.service import MissionService
from prophet_agent.nim.errors import NotConfigured


def _no_planner(*args, **kwargs):
    raise NotConfigured("no planner for this test")


def _service(db_path, program_factory):
    svc = MissionService(db_path, _no_planner, {})
    svc._build_program = lambda mission: program_factory()
    return svc


def _hold_program(entered, gate):
    def program(ctx):
        def hold(c):
            entered.set()
            gate.wait(10)
            return "held"
        ctx.step("hold", hold)
        ctx.step("next", lambda c: "after")       # the cancel is checked before this step runs
        return {"status": "met"}
    return program


def _ask_after(entered, gate):
    def program(ctx):
        def ask(c):
            entered.set()
            gate.wait(10)
            return c.ask({"question": "Which?"})
        return {"status": "met", "answer": ctx.step("ask", ask)}
    return program


def _budget():
    return {"max_tool_calls": 5, "max_seconds": 60}


def test_a_cancel_after_another_process_took_over_the_lease_is_recorded_not_raised(tmp_path):
    db = tmp_path / "takeover.sqlite"
    e1, g1, e2, g2 = threading.Event(), threading.Event(), threading.Event(), threading.Event()
    stalled = _service(db, lambda: _hold_program(e1, g1))
    taker = _service(db, lambda: _hold_program(e2, g2))
    mid = stalled.store.create_mission("long run", _budget())
    stalled.start(mid)
    assert e1.wait(10)
    # The stalled process's lease lapses and the other process takes the mission over.
    with stalled.store.lock:
        stalled.store.conn.execute("UPDATE missions SET lease_until=0 WHERE id=?", (mid,))
    taker.start(mid)
    assert e2.wait(10)
    g1.set()
    stalled.wait(mid, 10)                         # its next write is refused and its run stops
    assert stalled.cancel(mid) == "cancelling"    # not LeaseLost
    m = taker.store.mission(mid)
    assert m["status"] == "running" and m["cancel_requested"] == 1
    g2.set()
    taker.wait(mid, 10)
    assert taker.store.mission(mid)["status"] == "cancelled"


def test_a_cancel_of_a_mission_another_process_is_running_leaves_it_to_that_run(tmp_path):
    db = tmp_path / "live.sqlite"
    entered, gate = threading.Event(), threading.Event()
    holder = _service(db, lambda: _hold_program(entered, gate))
    other = _service(db, lambda: _hold_program(threading.Event(), threading.Event()))
    mid = holder.store.create_mission("long run", _budget())
    holder.start(mid)
    assert entered.wait(10)
    assert other.cancel(mid) == "cancelling"
    assert other.store.mission(mid)["status"] == "running"
    gate.set()
    holder.wait(mid, 10)
    assert holder.store.mission(mid)["status"] == "cancelled"


def test_a_cancel_of_a_paused_mission_from_another_process_closes_it(tmp_path):
    db = tmp_path / "paused.sqlite"
    open_gate = threading.Event()
    open_gate.set()
    first = _service(db, lambda: _ask_after(threading.Event(), open_gate))
    mid = first.store.create_mission("ask me", _budget())
    first.start(mid)
    first.wait(mid, 10)
    assert first.store.mission(mid)["status"] == "waiting_user"
    other = _service(db, lambda: _ask_after(threading.Event(), open_gate))
    assert other.cancel(mid) == "cancelled"
    assert other.store.mission(mid)["status"] == "cancelled"
    assert other.store.open_questions(mid) == []


def test_a_cancel_that_arrives_while_the_run_is_pausing_is_not_left_waiting(tmp_path):
    entered, gate = threading.Event(), threading.Event()
    svc = _service(tmp_path / "pausing.sqlite", lambda: _ask_after(entered, gate))
    mid = svc.store.create_mission("ask me", _budget())
    svc.start(mid)
    assert entered.wait(10)
    assert svc.cancel(mid) == "cancelling"
    gate.set()                                    # the run now asks its question and pauses
    svc.wait(mid, 10)
    assert svc.store.mission(mid)["status"] == "cancelled"
    assert svc.store.open_questions(mid) == []
