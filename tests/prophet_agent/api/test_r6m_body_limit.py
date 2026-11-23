"""Every route that reads a request body is capped, chunked or not (round 6m).

Before this round only the two raw upload routes checked their body, and only the upload size. A JSON route read its
body whole, so a chunked request with no Content-Length of any size reached the parser. One ASGI-level cap now covers
all of them; these tests send padded bodies larger than the JSON cap to every body-reading route.
"""
from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from prophet_agent.api import app as app_module
from prophet_agent.api.app import create_app
from prophet_agent.api.service import MissionService
from prophet_agent.tools import default_tools

BIG = 4 * 1024 * 1024          # well past any JSON cap


def _client(tmp_path):
    service = MissionService(tmp_path / "api.sqlite", lambda: None, default_tools())
    return TestClient(create_app(service), base_url="http://127.0.0.1:8000"), service


def _chunks(data: bytes, size: int = 64 * 1024):
    """A generator body: httpx sends it with Transfer-Encoding: chunked and no Content-Length."""
    for i in range(0, len(data), size):
        yield data[i:i + size]


def _padded(size: int = BIG, **fields) -> bytes:
    return json.dumps({**fields, "pad": "a" * size}).encode("utf-8")


def test_a_chunked_json_body_over_the_cap_is_refused(tmp_path):
    client, service = _client(tmp_path)
    nid = client.post("/api/notebooks", json={"title": "keep"}).json()["id"]
    r = client.patch(f"/api/notebooks/{nid}", content=_chunks(_padded(title="changed")),
                     headers={"content-type": "application/json"})
    assert r.status_code == 413
    assert service.notebooks.get(nid)["title"] == "keep"


def test_a_declared_json_body_over_the_cap_is_refused(tmp_path):
    client, service = _client(tmp_path)
    nid = client.post("/api/notebooks", json={"title": "keep"}).json()["id"]
    body = _padded(title="changed")
    r = client.patch(f"/api/notebooks/{nid}", content=body, headers={"content-type": "application/json"})
    assert r.status_code == 413
    assert service.notebooks.get(nid)["title"] == "keep"


def test_a_chunked_mission_create_over_the_cap_creates_nothing(tmp_path):
    client, service = _client(tmp_path)
    r = client.post("/api/missions", content=_chunks(_padded(goal="forecast the next quarter")),
                    headers={"content-type": "application/json"})
    assert r.status_code == 413
    assert service.list() == []


def test_a_chunked_json_body_under_the_cap_is_still_read(tmp_path):
    client, service = _client(tmp_path)
    nid = client.post("/api/notebooks", json={"title": "keep"}).json()["id"]
    body = json.dumps({"title": "renamed"}).encode("utf-8")
    r = client.patch(f"/api/notebooks/{nid}", content=_chunks(body, size=7),
                     headers={"content-type": "application/json"})
    assert r.status_code == 200
    assert service.notebooks.get(nid)["title"] == "renamed"


def _body_routes(client):
    """(method, path, nid-free setup) for every route that reads a body. Ids come from real rows, so the cap, not a 404,
    decides the answer."""
    nid = client.post("/api/notebooks", json={"title": "routes"}).json()["id"]
    return nid, [
        ("POST", "/api/missions"),
        ("POST", "/api/notebooks"),
        ("PATCH", f"/api/notebooks/{nid}"),
        ("POST", f"/api/notebooks/{nid}/turns"),
        ("PUT", f"/api/notebooks/{nid}/sharing"),
        ("PUT", "/api/settings"),
        ("POST", "/api/settings/clear-data"),
        ("POST", "/api/questions/nope/answer"),
        ("PATCH", "/api/missions/nope"),
        ("PATCH", f"/api/notebooks/{nid}/sources/nope"),
    ]


@pytest.mark.parametrize("index", range(10))
def test_every_body_route_refuses_a_chunked_body_over_the_cap(tmp_path, index):
    client, _ = _client(tmp_path)
    _, routes = _body_routes(client)
    method, path = routes[index]
    r = client.request(method, path, content=_chunks(_padded()), headers={"content-type": "application/json"})
    assert r.status_code == 413, (method, path, r.status_code)


def test_the_raw_upload_cap_still_applies_to_the_dataset_route(tmp_path, monkeypatch):
    """The dataset route keeps its own, larger cap: a body under MAX_BYTES but over the JSON cap is accepted."""
    client, _ = _client(tmp_path)
    monkeypatch.setattr(app_module, "MAX_BYTES", 8 * 1024 * 1024)
    body = b"a,b\n" + b"1,2\n" * (BIG // 4)            # ~4 MiB: over the JSON cap, under the upload cap
    r = client.post("/api/datasets?name=wide.csv", content=_chunks(body))
    assert r.status_code == 201
