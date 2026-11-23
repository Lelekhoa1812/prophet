"""Quick-forecast patches (round 6m): a patch is checked before any of its fields is written.

PATCH /api/missions/{id} can archive, rename and pin. Before this round an unknown id with an empty patch was 200, and a
patch whose archive was applied before its blank name was refused left the archive in place with a 409.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from prophet_agent.api import titles
from prophet_agent.api.app import create_app
from prophet_agent.api.service import MissionService
from prophet_agent.tools import default_tools

MISSING = "0" * 32


@pytest.fixture(autouse=True)
def no_title_model(monkeypatch):
    monkeypatch.setattr(titles, "name_in_background", lambda *a, **k: None)


def _client(tmp_path):
    service = MissionService(tmp_path / "api.sqlite", lambda: None, default_tools())
    return TestClient(create_app(service), base_url="http://127.0.0.1:8000"), service


def test_patching_an_unknown_mission_is_not_found_even_with_no_fields(tmp_path):
    client, _ = _client(tmp_path)
    assert client.patch(f"/api/missions/{MISSING}", json={}).status_code == 404


def test_a_patch_to_a_notebook_turn_is_not_found_because_it_is_not_a_quick_forecast(tmp_path):
    client, service = _client(tmp_path)
    nid = client.post("/api/notebooks", json={"title": "turns"}).json()["id"]
    mid = service.create("a turn that belongs to a notebook")
    with service.store._tx() as c:
        c.execute("UPDATE missions SET notebook_id=? WHERE id=?", (nid, mid))     # the row a notebook turn leaves behind
    assert client.patch(f"/api/missions/{mid}", json={}).status_code == 404
    assert client.patch(f"/api/missions/{mid}", json={"pinned": True}).status_code == 404


def test_a_patch_refused_for_a_blank_name_changes_none_of_its_fields(tmp_path):
    """archived and title in one patch: the blank name is refused, so the archive must not be applied either."""
    client, service = _client(tmp_path)
    mid = service.create("forecast the regional demand")
    service.store.set_status(mid, "failed")          # finished, so the archive itself would be allowed
    r = client.patch(f"/api/missions/{mid}", json={"archived": True, "title": "   "})
    assert r.status_code == 409
    assert service.store.mission(mid)["archived"] in (0, False)


def test_a_valid_patch_applies_every_field(tmp_path):
    client, service = _client(tmp_path)
    mid = service.create("forecast the regional demand")
    service.store.set_status(mid, "failed")
    r = client.patch(f"/api/missions/{mid}", json={"archived": True, "title": "Regional demand", "pinned": True})
    assert r.status_code == 200
    row = service.store.mission(mid)
    assert row["archived"] in (1, True) and row["title"] == "Regional demand" and row["pinned"] in (1, True)
