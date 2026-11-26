"""HTTP routes for Fork and Rewind (docs/prophet/fork_rewind_copy.md): status codes, bodies, and that a refused call writes nothing."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from prophet_agent.api import titles
from prophet_agent.api.app import create_app
from prophet_agent.api.service import MissionService
from prophet_agent.runtime.runner import Runtime, Tool
from prophet_agent.tools import default_tools

MISSING = "0" * 32


@pytest.fixture(autouse=True)
def no_title_model(monkeypatch):
    monkeypatch.setattr(titles, "name_in_background", lambda *a, **k: None)


@pytest.fixture()
def api(tmp_path):
    service = MissionService(tmp_path / "api.sqlite", lambda: None, default_tools())
    client = TestClient(create_app(service), base_url="http://127.0.0.1:8000")
    yield client, service
    service.close()


def _turns(service, count=3):
    nid = service.notebooks.create("Sales")["id"]
    tool = Tool("work", lambda args, key: {"ok": args["n"]}, idempotent=True)
    ids = []
    for n in range(count):
        mid = service.create_turn(nid, f"question number {n + 1}")
        Runtime(service.store).run(mid, lambda ctx, n=n: ctx.step("act", lambda c: c.call(tool, {"n": n})) and {"status": "met", "answer": "ok"})
        ids.append(mid)
    return nid, ids


def test_rewind_then_undo_over_http(api):
    client, service = api
    nid, ids = _turns(service)
    r = client.post(f"/api/notebooks/{nid}/turns/{ids[1]}/rewind")
    assert r.status_code == 200
    body = r.json()
    assert body["prefill"] == "question number 2" and body["dropped"] == ids[1:]
    assert [t["id"] for t in client.get(f"/api/notebooks/{nid}").json()["turns"]] == [ids[:1][0]]
    assert [t["id"] for t in client.get(f"/api/notebooks/{nid}/turns").json()] == ids[:1]
    listing = client.get(f"/api/notebooks/{nid}/rewinds").json()
    assert listing[0]["id"] == body["rewind_id"] and len(listing[0]["dropped"]) == 2
    assert client.post(f"/api/rewinds/{body['rewind_id']}/undo").json()["restored"] == ids[1:]
    assert [t["id"] for t in client.get(f"/api/notebooks/{nid}").json()["turns"]] == ids
    assert client.post(f"/api/rewinds/{body['rewind_id']}/undo").status_code == 409


def test_fork_over_http_creates_a_notebook_and_leaves_the_original(api):
    client, service = api
    nid, ids = _turns(service)
    r = client.post(f"/api/notebooks/{nid}/turns/{ids[2]}/fork")
    assert r.status_code == 201
    body = r.json()
    assert body["copied_turns"] == 2 and body["prefill"] == "question number 3"
    new = client.get(f"/api/notebooks/{body['notebook_id']}").json()
    assert [t["goal"] for t in new["turns"]] == ["question number 1", "question number 2"]
    assert [t["id"] for t in client.get(f"/api/notebooks/{nid}").json()["turns"]] == ids
    assert any(n["id"] == body["notebook_id"] for n in client.get("/api/notebooks").json())
    named = client.post(f"/api/notebooks/{nid}/turns/{ids[1]}/fork", json={"title": "Try the other way"}).json()
    assert named["title"] == "Try the other way"


def test_unknown_and_refused_checkpoints(api):
    client, service = api
    nid, ids = _turns(service, 2)
    assert client.post(f"/api/notebooks/{nid}/turns/{MISSING}/rewind").status_code == 404
    assert client.post(f"/api/notebooks/{MISSING}/turns/{ids[0]}/fork").status_code == 404
    assert client.post(f"/api/rewinds/{MISSING}/undo").status_code == 404
    assert client.get(f"/api/notebooks/{MISSING}/rewinds").status_code == 404
    running = service.create_turn(nid, "still working")
    assert client.post(f"/api/notebooks/{nid}/turns/{ids[0]}/rewind").status_code == 409
    assert running and [t["id"] for t in client.get(f"/api/notebooks/{nid}").json()["turns"]] == [*ids, running]


def test_a_dropped_question_cannot_be_resumed_over_http(api):
    client, service = api
    nid, ids = _turns(service)
    client.post(f"/api/notebooks/{nid}/turns/{ids[1]}/rewind")
    service.store.set_status(ids[2], "interrupted")
    assert client.post(f"/api/missions/{ids[2]}/resume").status_code == 409


def test_quick_forecast_rewind_over_http(api):
    client, service = api
    mid = service.create("forecast the next quarter")
    Runtime(service.store).run(mid, lambda ctx: {"status": "met", "answer": "ok"})
    r = client.post(f"/api/missions/{mid}/rewind")
    assert r.status_code == 200 and r.json()["prefill"] == "forecast the next quarter"
    assert mid not in [m["id"] for m in client.get("/api/missions").json()]
    assert client.post(f"/api/missions/{mid}/rewind").status_code == 409
    assert client.post(f"/api/rewinds/{r.json()['rewind_id']}/undo").status_code == 200
    assert mid in [m["id"] for m in client.get("/api/missions").json()]
    assert client.post(f"/api/missions/{MISSING}/rewind").status_code == 404
