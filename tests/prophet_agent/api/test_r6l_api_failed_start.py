"""A mission whose start fails is marked failed, not left in the "created" state.

Before the fix, the row was written as "created" and a failed start (no planner) returned 503 and left it there. A created
row counts as active: it blocked the notebook's next turn with 409, and it could not be deleted, until the orphan sweep
turned it into "interrupted" some seconds later. These tests check the row at once and after a sweep.
"""
from __future__ import annotations

from fastapi.testclient import TestClient

from prophet_agent.api.app import create_app
from prophet_agent.api.service import MissionService
from prophet_agent.nim.errors import NotConfigured
from prophet_agent.tools import default_tools


def _no_planner(*args, **kwargs):
    raise NotConfigured("no planner for this test")


def _client(tmp_path):
    service = MissionService(tmp_path / "api.sqlite", _no_planner, default_tools())
    return TestClient(create_app(service), base_url="http://127.0.0.1:8000"), service


def test_a_failed_start_marks_the_mission_failed_and_the_sweep_leaves_it_alone(tmp_path):
    client, service = _client(tmp_path)
    r = client.post("/api/missions", json={"goal": "check the pipeline"})
    assert r.status_code == 503
    rows = service.store.conn.execute("SELECT id, status FROM missions").fetchall()
    assert len(rows) == 1 and rows[0]["status"] == "failed"
    assert service.sweep() == []
    assert service.store.mission(rows[0]["id"])["status"] == "failed"


def test_a_failed_mission_can_be_deleted_at_once(tmp_path):
    client, service = _client(tmp_path)
    client.post("/api/missions", json={"goal": "check the pipeline"})
    mid = service.store.conn.execute("SELECT id FROM missions").fetchone()["id"]
    assert client.delete(f"/api/missions/{mid}").status_code == 200


def test_a_failed_notebook_turn_does_not_block_the_next_turn(tmp_path):
    client, service = _client(tmp_path)
    nid = client.post("/api/notebooks", json={"title": "start test"}).json()["id"]
    first = client.post(f"/api/notebooks/{nid}/turns", json={"goal": "first question"})
    assert first.status_code == 503                       # not 409: the failed turn is no longer active
    second = client.post(f"/api/notebooks/{nid}/turns", json={"goal": "second question"})
    assert second.status_code == 503
    statuses = [r["status"] for r in service.store.conn.execute("SELECT status FROM missions ORDER BY seq").fetchall()]
    assert statuses == ["failed", "failed"]
