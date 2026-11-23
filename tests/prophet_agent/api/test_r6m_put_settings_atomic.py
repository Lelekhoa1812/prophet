"""PUT /api/settings is all-or-nothing. A valid limit sent with a key that is refused (wrong shape) must not leave the
limit saved. Before the fix the route called settings.update() first, so the limit was written and then the key was
refused. No test here stores a key: the refused write stops before any key list is written, so no .env is touched."""
from __future__ import annotations

from fastapi.testclient import TestClient

from prophet_agent.api.app import create_app
from prophet_agent.api.service import MissionService
from prophet_agent.settings import BOUNDS
from prophet_agent.tools import default_tools


def _client(tmp_path):
    service = MissionService(tmp_path / "api.sqlite", lambda: None, default_tools())
    return TestClient(create_app(service), base_url="http://127.0.0.1:8000"), service


def test_a_refused_key_after_a_valid_limit_leaves_the_limit_unchanged(tmp_path):
    client, service = _client(tmp_path)
    section, name = next(iter(BOUNDS))
    low, high = BOUNDS[(section, name)]
    current = service.settings.values()[section][name]
    new = low if current == high else high
    before = service.settings.values()
    # "notakey!" matches no key shape the NIM check accepts, so set_keys would refuse it after update() had saved.
    r = client.put("/api/settings", json={section: {name: new}, "nim": {"keys": ["notakey!"]}})
    assert r.status_code == 422
    assert service.settings.values() == before
    assert service.settings.values()[section][name] == current


def test_a_valid_save_still_applies_the_limit(tmp_path):
    client, service = _client(tmp_path)
    section, name = next(iter(BOUNDS))
    low, high = BOUNDS[(section, name)]
    current = service.settings.values()[section][name]
    new = low if current == high else high
    r = client.put("/api/settings", json={section: {name: new}})
    assert r.status_code == 200
    assert service.settings.values()[section][name] == new
