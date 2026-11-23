"""Service writes on a mission must respect a lease another process holds.

Before the fix, start() and continue_mission() wrote the budget and then started a thread whose lease acquisition failed
silently, so the caller saw success for a mission that was not running. delete() removed a paused mission that another
process held. The shared store's fence raises LeaseLost for such writes, which is not a conflict the API can map.
"""
from __future__ import annotations

import pytest

from prophet_agent.api.service import MissionService
from prophet_agent.nim.errors import NotConfigured
from prophet_agent.runtime.store import LeaseHeld


def _no_planner(*args, **kwargs):
    raise NotConfigured("no planner for this test")


def _service(db_path):
    svc = MissionService(db_path, _no_planner, {})
    svc._build_program = lambda mission: (lambda ctx: {"status": "met"})
    return svc


def _budget():
    return {"max_tool_calls": 5, "max_seconds": 60}


def test_a_start_while_another_process_holds_the_lease_is_refused(tmp_path):
    db = tmp_path / "start.sqlite"
    holder, other = _service(db), _service(db)
    mid = holder.store.create_mission("long run", _budget())
    holder.store.set_status(mid, "running")
    holder.store.acquire_lease(mid, "holder-live", 300)
    with pytest.raises(LeaseHeld):
        other.start(mid)
    assert other.store.mission(mid)["status"] == "running"


def test_a_continue_while_another_process_holds_the_lease_changes_nothing(tmp_path):
    db = tmp_path / "continue.sqlite"
    holder, other = _service(db), _service(db)
    mid = holder.store.create_mission("long run", _budget())
    holder.store.set_status(mid, "checkpoint")
    holder.store.acquire_lease(mid, "holder-live", 300)
    before = other.store.mission(mid)["budget"]
    with pytest.raises(LeaseHeld):
        other.continue_mission(mid)
    assert other.store.mission(mid)["budget"] == before


def test_a_delete_of_a_mission_another_process_holds_is_refused(tmp_path):
    db = tmp_path / "delete.sqlite"
    holder, other = _service(db), _service(db)
    mid = holder.store.create_mission("long run", _budget())
    holder.store.set_status(mid, "checkpoint")
    holder.store.acquire_lease(mid, "holder-live", 300)
    with pytest.raises(ValueError):
        other.delete(mid)
    assert other.store.mission(mid)["id"] == mid
