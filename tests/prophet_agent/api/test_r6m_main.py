"""Behaviour of the entry point (api/__main__.py) (round 6m).

The module builds the app on the real database and data folder, so each test points those at tmp_path and replaces
uvicorn.run. Nothing here starts a server or touches the real data folder or .env.
"""
from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import uvicorn

from prophet_agent.api import __main__ as entry
from prophet_agent.api import titles


@pytest.fixture(autouse=True)
def isolated(monkeypatch, tmp_path):
    monkeypatch.setattr(entry, "DB_PATH", tmp_path / "api.sqlite")
    monkeypatch.setattr(entry, "DATA", tmp_path / "data")
    monkeypatch.setattr(entry, "ensure_dirs", lambda: None)
    monkeypatch.setattr(titles, "name_in_background", lambda *a, **k: None)
    monkeypatch.delenv("PROPHET_API_PORT", raising=False)


def _capture_run(monkeypatch):
    calls = []
    monkeypatch.setattr(uvicorn, "run", lambda app, **kwargs: calls.append((app, kwargs)))
    return calls


def test_build_app_serves_the_api_on_the_configured_database_and_data_folder(tmp_path):
    app = entry.build_app()
    assert isinstance(app, FastAPI)
    service = app.state.service
    assert service.settings.data_root == tmp_path / "data"
    assert (tmp_path / "api.sqlite").exists()
    r = TestClient(app, base_url="http://127.0.0.1:8000").get("/api/health")
    assert r.status_code == 200 and r.json()["ok"] is True


def test_build_app_creates_the_folders_it_needs(monkeypatch):
    made = []
    monkeypatch.setattr(entry, "ensure_dirs", lambda: made.append(True))
    entry.build_app()
    assert made == [True]


def test_main_listens_on_loopback_only_on_the_default_port(monkeypatch):
    calls = _capture_run(monkeypatch)
    sentinel = FastAPI()
    monkeypatch.setattr(entry, "build_app", lambda: sentinel)
    entry.main()
    assert len(calls) == 1
    app, kwargs = calls[0]
    assert app is sentinel
    assert kwargs["host"] == "127.0.0.1" and kwargs["port"] == 8010


def test_main_uses_the_port_from_the_environment(monkeypatch):
    calls = _capture_run(monkeypatch)
    monkeypatch.setattr(entry, "build_app", lambda: FastAPI())
    monkeypatch.setenv("PROPHET_API_PORT", "8123")
    entry.main()
    assert calls[0][1]["port"] == 8123 and calls[0][1]["host"] == "127.0.0.1"


def test_an_empty_port_setting_means_the_default_port(monkeypatch):
    calls = _capture_run(monkeypatch)
    monkeypatch.setattr(entry, "build_app", lambda: FastAPI())
    monkeypatch.setenv("PROPHET_API_PORT", "")
    entry.main()
    assert calls[0][1]["port"] == 8010


@pytest.mark.parametrize("value", ["http", "80a", "0", "65536", "-1", "8010.5"])
def test_a_port_that_is_not_a_port_number_stops_with_a_clear_message(monkeypatch, value):
    calls = _capture_run(monkeypatch)
    monkeypatch.setattr(entry, "build_app", lambda: FastAPI())
    monkeypatch.setenv("PROPHET_API_PORT", value)
    with pytest.raises(SystemExit, match="PROPHET_API_PORT"):
        entry.main()
    assert calls == []
