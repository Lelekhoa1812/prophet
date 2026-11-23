"""Archiving a forecast must respect a live lease, as cancel does.

Defect (docs/prophet/GOAL.md): archive() refused only the statuses created, running and waiting_user. A forecast whose
status is final while a runtime still holds its lease (the runtime is finishing its writes) could be hidden under it.
cancel() refuses to close a mission that a live lease holds; archive now refuses the same way.
"""
from __future__ import annotations

import time

import pytest

from prophet_agent.api.service import MissionService
from prophet_agent.nim.errors import NotConfigured


def _no_planner(*args, **kwargs):
    raise NotConfigured("no planner for this test")


def _service(db_path):
    return MissionService(db_path, _no_planner, {})


def _budget():
    return {"max_tool_calls": 5, "max_seconds": 60}


def test_archive_is_refused_while_a_live_lease_is_held(tmp_path):
    svc = _service(tmp_path / "held.sqlite")
    mid = svc.store.create_mission("quick forecast", _budget())
    svc.store.set_status(mid, "met")
    svc.store.acquire_lease(mid, "runtime-1", 60)
    with pytest.raises(ValueError, match="still running"):
        svc.archive(mid)
    assert svc.store.mission(mid)["archived"] == 0


def test_archive_works_once_the_lease_is_released(tmp_path):
    svc = _service(tmp_path / "released.sqlite")
    mid = svc.store.create_mission("quick forecast", _budget())
    svc.store.set_status(mid, "met")
    svc.store.acquire_lease(mid, "runtime-1", 60)
    svc.store.release_lease(mid, "runtime-1")
    svc.archive(mid)
    assert svc.store.mission(mid)["archived"] == 1


def test_archive_works_when_the_lease_has_lapsed(tmp_path):
    svc = _service(tmp_path / "lapsed.sqlite")
    mid = svc.store.create_mission("quick forecast", _budget())
    svc.store.set_status(mid, "met")
    with svc.store.lock:
        svc.store.conn.execute("UPDATE missions SET lease_owner=?, lease_until=? WHERE id=?",
                               ("runtime-1", time.time() - 5, mid))
    svc.archive(mid)
    assert svc.store.mission(mid)["archived"] == 1
