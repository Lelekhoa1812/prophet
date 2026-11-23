"""Upload size cap on the two raw-body routes (dataset upload and notebook source).

A declared Content-Length is refused before the body is read. A chunked request has no Content-Length, so the cap must
count the streamed bytes too; before this, a chunked body of any size was read whole into memory.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from prophet_agent.api import app as app_module
from prophet_agent.api.app import create_app
from prophet_agent.api.service import MissionService
from prophet_agent.tools import default_tools

CAP = 1024


def _client(tmp_path):
    service = MissionService(tmp_path / "api.sqlite", lambda: None, default_tools())
    return TestClient(create_app(service), base_url="http://127.0.0.1:8000"), service


def _chunks(data: bytes, size: int = 500):
    """A generator body: httpx sends it with Transfer-Encoding: chunked and no Content-Length."""
    for i in range(0, len(data), size):
        yield data[i:i + size]


@pytest.fixture()
def small_cap(monkeypatch):
    monkeypatch.setattr(app_module, "MAX_BYTES", CAP)


def test_a_declared_oversize_upload_is_refused_before_the_body_is_read(tmp_path, small_cap):
    client, service = _client(tmp_path)
    r = client.post("/api/datasets?name=big.csv", content=b"x" * (CAP + 1))
    assert r.status_code == 413
    assert service.datasets.list() == []


def test_a_chunked_oversize_upload_is_refused_too(tmp_path, small_cap):
    client, service = _client(tmp_path)
    body = b"x" * (CAP * 4)
    r = client.post("/api/datasets?name=big.csv", content=_chunks(body))
    assert r.status_code == 413
    assert service.datasets.list() == []


def test_a_chunked_upload_under_the_cap_is_still_accepted(tmp_path, small_cap):
    client, _ = _client(tmp_path)
    body = b"a,b\n1,2\n" * 20                      # 160 bytes, well under the cap
    r = client.post("/api/datasets?name=small.csv", content=_chunks(body))
    assert r.status_code == 201
    assert r.json()["bytes"] == len(body)


def test_a_chunked_oversize_notebook_source_is_refused(tmp_path, small_cap):
    client, service = _client(tmp_path)
    nid = client.post("/api/notebooks", json={"title": "cap test"}).json()["id"]
    r = client.post(f"/api/notebooks/{nid}/sources?name=big.csv", content=_chunks(b"x" * (CAP * 4)))
    assert r.status_code == 413
    assert service.notebooks.sources(nid) == []
