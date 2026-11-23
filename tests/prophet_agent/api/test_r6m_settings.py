"""PUT /api/settings (round 6m): a request with a bad shape changes nothing, not even its valid limits.

The route saves limits (Settings.update) before it saves keys. A shape error that is only found after the limits are saved
used to return 422 with the limits already changed. Shape checks now run before anything is written.
"""
from __future__ import annotations

import os

import pytest
from fastapi.testclient import TestClient

from prophet_agent.api import titles
from prophet_agent.api.app import create_app
from prophet_agent.api.service import MissionService
from prophet_agent.tools import default_tools


@pytest.fixture(autouse=True)
def no_title_model(monkeypatch):
    monkeypatch.setattr(titles, "name_in_background", lambda *a, **k: None)


@pytest.fixture()
def restore_env():
    """Settings.set_keys writes the process environment; put it back exactly as it was."""
    before = dict(os.environ)
    yield
    os.environ.clear()
    os.environ.update(before)


def _client(tmp_path):
    service = MissionService(tmp_path / "api.sqlite", lambda: None, default_tools())
    return TestClient(create_app(service), base_url="http://127.0.0.1:8000"), service


def _other_limit(service):
    current = service.settings.values()["agent"]["part_steps"]
    return 50 if current != 50 else 60, current


def test_an_unknown_provider_changes_no_limit(tmp_path, restore_env):
    client, service = _client(tmp_path)
    new, before = _other_limit(service)
    r = client.put("/api/settings", json={"agent": {"part_steps": new},
                                          "providers": {"no_such_provider": {"keys": []}}})
    assert r.status_code == 422
    assert service.settings.values()["agent"]["part_steps"] == before


def test_a_keys_value_that_is_not_a_list_changes_no_limit(tmp_path, restore_env):
    client, service = _client(tmp_path)
    new, before = _other_limit(service)
    r = client.put("/api/settings", json={"agent": {"part_steps": new}, "nim": {"keys": "nvapi-FAKETESTKEY0123456789"}})
    assert r.status_code == 422
    assert service.settings.values()["agent"]["part_steps"] == before


def test_a_provider_entry_with_a_keys_value_that_is_not_a_list_changes_no_limit(tmp_path, restore_env):
    client, service = _client(tmp_path)
    new, before = _other_limit(service)
    r = client.put("/api/settings", json={"agent": {"part_steps": new},
                                          "providers": {"nim": {"keys": "not-a-list"}}})
    assert r.status_code == 422
    assert service.settings.values()["agent"]["part_steps"] == before


def test_a_valid_limit_is_saved_and_returned(tmp_path, restore_env):
    client, service = _client(tmp_path)
    new, _ = _other_limit(service)
    r = client.put("/api/settings", json={"agent": {"part_steps": new}})
    assert r.status_code == 200
    assert service.settings.values()["agent"]["part_steps"] == new
