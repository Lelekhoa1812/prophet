"""Sidebar organiser: short names, rename, pin, delete for quick forecasts and notebooks. No real model is called."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from prophet_agent.api.app import create_app
from prophet_agent.api.service import MissionService
from prophet_agent.api.titles import clean_title, fallback_title, propose_title
from prophet_agent.tools import default_tools


@pytest.fixture(autouse=True)
def no_real_keys(monkeypatch):
    import prophet_agent.nim.transport as transport
    monkeypatch.setattr(transport, "load_keys", lambda *a, **k: [])


def _service(tmp_path):
    return MissionService(tmp_path / "org.sqlite", lambda: None, default_tools())


def _client(service):
    return TestClient(create_app(service), base_url="http://127.0.0.1:8000")


def test_fallback_title_is_at_most_four_words():
    title = fallback_title("Please forecast the next 30 days of regional demand for the north stores")
    assert len(title.split()) <= 4
    assert title and "please" not in title.lower().split()


def test_fallback_title_never_empty():
    assert fallback_title("   ") == "Untitled"


@pytest.mark.parametrize("raw, expected", [
    ("Regional demand forecast", "Regional demand forecast"),
    ('"Regional demand forecast."', "Regional demand forecast"),
    ("Title: Weekly sales trend", "Weekly sales trend"),
])
def test_clean_title_accepts_short_plain_names(raw, expected):
    assert clean_title(raw) == expected


@pytest.mark.parametrize("raw", ["", "one two three four five", "multi\nline title", "x" * 80])
def test_clean_title_rejects_long_or_multiline_replies(raw):
    assert clean_title(raw) is None


def test_propose_title_uses_model_when_valid_and_survives_failure():
    assert propose_title("forecast demand", lambda _p: "Demand outlook") == "Demand outlook"
    assert propose_title("forecast demand", lambda _p: "a very long reply with far too many words") is None

    def boom(_p):
        raise RuntimeError("no keys")

    assert propose_title("forecast demand", boom) is None


def test_new_forecast_gets_a_short_name_and_can_be_renamed(tmp_path):
    service = _service(tmp_path)
    mid = service.create("Forecast the next 30 days of regional demand for the north stores")
    client = _client(service)
    row = next(m for m in client.get("/api/missions").json() if m["id"] == mid)
    assert len(row["title"].split()) <= 4

    assert client.patch(f"/api/missions/{mid}", json={"title": "North stores"}).status_code == 200
    row = next(m for m in client.get("/api/missions").json() if m["id"] == mid)
    assert row["title"] == "North stores"


def test_rename_rejects_empty_and_unknown(tmp_path):
    service = _service(tmp_path)
    mid = service.create("Weekly demand review")
    client = _client(service)
    assert client.patch(f"/api/missions/{mid}", json={"title": "   "}).status_code == 409
    assert client.patch("/api/missions/nope", json={"title": "Fine name"}).status_code == 404


def test_pinned_forecasts_sort_first(tmp_path):
    service = _service(tmp_path)
    older = service.create("Older forecast goal")
    newer = service.create("Newer forecast goal")
    client = _client(service)
    client.patch(f"/api/missions/{older}", json={"pinned": True})
    ids = [m["id"] for m in client.get("/api/missions").json()]
    assert ids[0] == older and newer in ids
    assert client.get("/api/missions").json()[0]["pinned"] is True


def test_delete_removes_a_finished_forecast_and_its_history(tmp_path):
    service = _service(tmp_path)
    mid = service.create("Forecast to delete")
    with service.store.lock:
        service.store.conn.execute("UPDATE missions SET status='done' WHERE id=?", (mid,))
        service.store.append_event(mid, "test.event", {})
    client = _client(service)
    assert client.delete(f"/api/missions/{mid}").json() == {"id": mid, "deleted": True}
    assert all(m["id"] != mid for m in client.get("/api/missions").json())
    with service.store.lock:
        assert service.store.conn.execute("SELECT COUNT(*) FROM events WHERE mission_id=?", (mid,)).fetchone()[0] == 0


def test_delete_refuses_a_running_forecast(tmp_path):
    service = _service(tmp_path)
    mid = service.create("Running forecast")
    client = _client(service)
    assert client.delete(f"/api/missions/{mid}").status_code == 409


def test_notebook_pin_and_delete(tmp_path):
    service = _service(tmp_path)
    client = _client(service)
    a = client.post("/api/notebooks", json={"title": "Alpha"}).json()["id"]
    b = client.post("/api/notebooks", json={"title": "Beta"}).json()["id"]
    assert client.patch(f"/api/notebooks/{b}", json={"pinned": True}).status_code == 200
    names = [n["id"] for n in client.get("/api/notebooks").json()]
    assert names[0] == b

    assert client.delete(f"/api/notebooks/{a}").json() == {"id": a, "deleted": True}
    assert all(n["id"] != a for n in client.get("/api/notebooks").json())
    assert client.delete(f"/api/notebooks/{a}").status_code == 404
