"""Table downloads by version (round 6m): version 0 must not fall through to the latest version.

Versions start at 1. A notebook table has one row per version; a request for a version that does not exist is 404.
"""
from __future__ import annotations

import time
import uuid

import pytest
from fastapi.testclient import TestClient

from prophet_agent.api import titles
from prophet_agent.api.app import create_app
from prophet_agent.api.service import MissionService
from prophet_agent.tools import default_tools


@pytest.fixture(autouse=True)
def no_title_model(monkeypatch):
    monkeypatch.setattr(titles, "name_in_background", lambda *a, **k: None)


def _client(tmp_path):
    service = MissionService(tmp_path / "api.sqlite", lambda: None, default_tools())
    return TestClient(create_app(service), base_url="http://127.0.0.1:8000"), service


def _table(service, nid, tmp_path):
    csv = tmp_path / "demand_v1.csv"
    csv.write_text("a\n1\n")
    tid, vid = uuid.uuid4().hex, uuid.uuid4().hex
    with service.store._tx() as c:
        c.execute("INSERT INTO derived_tables (id, notebook_id, name, description, created) VALUES (?,?,?,?,?)",
                  (tid, nid, "demand", "", time.time()))
        c.execute("INSERT INTO derived_versions (id, table_id, version, rows, columns_json, sha256, file_path, lineage_json, "
                  "turn_id, created) VALUES (?,?,?,?,?,?,?,?,?,?)",
                  (vid, tid, 1, 1, '["a"]', "0" * 64, str(csv), "[]", None, time.time()))


def test_a_table_download_serves_the_latest_and_the_named_version(tmp_path):
    client, service = _client(tmp_path)
    nid = client.post("/api/notebooks", json={"title": "tables"}).json()["id"]
    _table(service, nid, tmp_path)
    latest = client.get(f"/api/notebooks/{nid}/tables/demand/download")
    named = client.get(f"/api/notebooks/{nid}/tables/demand/download?version=1")
    assert latest.status_code == 200 and named.status_code == 200
    assert latest.content == named.content == b"a\n1\n"


@pytest.mark.parametrize("version", [0, -1, 2])
def test_a_table_version_that_is_not_stored_is_not_found(tmp_path, version):
    """Version 0 used to pass the store's truth test as "no version" and quietly serve the latest one."""
    client, service = _client(tmp_path)
    nid = client.post("/api/notebooks", json={"title": "tables"}).json()["id"]
    _table(service, nid, tmp_path)
    r = client.get(f"/api/notebooks/{nid}/tables/demand/download?version={version}")
    assert r.status_code == 404
