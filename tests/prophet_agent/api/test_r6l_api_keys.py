"""PUT /api/settings refuses a key string longer than KEY_CHARS_MAX before anything is saved. No test here stores a real key:
every refused write stops before the key list is written, so no .env or process environment is touched."""
from __future__ import annotations

from fastapi.testclient import TestClient

from prophet_agent.api.app import KEY_CHARS_MAX, create_app
from prophet_agent.api.service import MissionService
from prophet_agent.tools import default_tools


def _client(tmp_path):
    service = MissionService(tmp_path / "api.sqlite", lambda: None, default_tools())
    return TestClient(create_app(service), base_url="http://127.0.0.1:8000"), service


def test_an_overlong_nim_key_is_refused_with_a_clear_message_and_nothing_is_saved(tmp_path):
    client, service = _client(tmp_path)
    # Compare against the key list before the write: a checkout with a .env already holds keys.
    before = service.settings.masked_keys()
    long_key = "nvapi-" + "A" * KEY_CHARS_MAX
    r = client.put("/api/settings", json={"nim": {"keys": [long_key]}})
    assert r.status_code == 422
    assert f"longer than {KEY_CHARS_MAX} characters" in r.json()["detail"]
    assert service.settings.masked_keys() == before


def test_an_overlong_key_for_another_provider_is_refused(tmp_path):
    client, service = _client(tmp_path)
    before = service.settings.masked_keys("claude")
    r = client.put("/api/settings", json={"providers": {"claude": {"keys": ["k" * (KEY_CHARS_MAX + 1)]}}})
    assert r.status_code == 422
    assert "longer than" in r.json()["detail"]
    assert service.settings.masked_keys("claude") == before


def test_an_overlong_key_refuses_the_whole_patch_not_just_the_key(tmp_path):
    client, service = _client(tmp_path)
    before = client.get("/api/settings").json()
    r = client.put("/api/settings", json={"llm": {"models": {"gpt-6-luna": {"enabled": False}}},
                                          "nim": {"keys": ["nvapi-" + "B" * KEY_CHARS_MAX]}})
    assert r.status_code == 422
    assert client.get("/api/settings").json() == before
