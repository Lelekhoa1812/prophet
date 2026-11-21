"""HTTP tests for model settings: the provider and model flags in GET/PUT /api/settings, the key test route, the usage
route, and the optional "model" field on mission creation. A scripted brain stands in for the planner; the provider
probe is replaced, so no test calls a real provider or reads a real key. Every data folder is temporary.
"""
from __future__ import annotations

import json

from fastapi.testclient import TestClient

from prophet_agent.api import app as app_module
from prophet_agent.api.app import create_app
from prophet_agent.api.service import MissionService
from prophet_agent.nim.errors import NotConfigured
from prophet_agent.nim.providers import CATALOG, PROVIDERS
from prophet_agent.tools import default_tools


class ScriptedBrain:
    def __init__(self, plans, evals):
        self.plans, self.evals = list(plans), list(evals)

    def decide(self, messages, schema, name):
        return (self.plans if name == "plan" else self.evals).pop(0)


def _client(tmp_path, brain_factory):
    service = MissionService(tmp_path / "api.sqlite", brain_factory, default_tools())
    return TestClient(create_app(service), base_url="http://127.0.0.1:8000"), service


def _finish_brain():
    return ScriptedBrain(plans=[{"action": "finish", "rationale": "done"}],
                         evals=[{"verdict": "met", "reason": "ok", "answer": "all good"}])


def _settings(client) -> dict:
    return client.get("/api/settings").json()


def _provider(body: dict, name: str) -> dict:
    return next(p for p in body["llm"]["providers"] if p["name"] == name)


def _model(provider: dict, model_id: str) -> dict:
    return next(m for m in provider["models"] if m["id"] == model_id)


def test_usage_lists_every_catalog_model_with_nothing_used_yet(tmp_path):
    client, _ = _client(tmp_path, lambda: _finish_brain())
    rows = client.get("/api/usage").json()["models"]
    assert [row["id"] for row in rows] == [m.id for m in CATALOG]
    for row in rows:
        assert row["used"] == {"1d": 0, "7d": 0, "30d": 0}
        assert row["budget"] == {"1d": None, "7d": None, "30d": None}
        assert set(row["resets"]) == {"1d", "7d", "30d"}


def test_settings_show_every_provider_with_its_models_and_flags(tmp_path):
    client, _ = _client(tmp_path, lambda: _finish_brain())
    body = _settings(client)
    assert {p["name"] for p in body["llm"]["providers"]} == set(PROVIDERS)
    claude = _provider(body, "claude")
    assert [m["id"] for m in claude["models"]] == [m.id for m in CATALOG if m.provider == "claude"]
    for model in claude["models"]:
        assert {"id", "label", "enabled", "status", "checked_at"} <= set(model)
    assert {"enabled", "ready", "keys", "models"} <= set(claude)


def test_put_switches_a_model_off_and_the_view_shows_it(tmp_path):
    client, _ = _client(tmp_path, lambda: _finish_brain())
    r = client.put("/api/settings", json={"llm": {"models": {"gpt-6-luna": {"enabled": False}}}})
    assert r.status_code == 200
    assert _model(_provider(r.json(), "openai"), "gpt-6-luna")["enabled"] is False
    assert _model(_provider(_settings(client), "openai"), "gpt-6-luna")["enabled"] is False


def test_put_refuses_an_unknown_model_or_provider(tmp_path):
    client, _ = _client(tmp_path, lambda: _finish_brain())
    assert client.put("/api/settings", json={"llm": {"models": {"nope": {"enabled": False}}}}).status_code == 422
    assert client.put("/api/settings", json={"llm": {"providers": {"nope": {"enabled": True}}}}).status_code == 422
    assert client.put("/api/settings", json={"llm": {"budgets": {"claude-haiku-5-5": {"1w": 1}}}}).status_code == 422


def test_put_saves_a_budget_and_usage_reports_it(tmp_path):
    client, _ = _client(tmp_path, lambda: _finish_brain())
    r = client.put("/api/settings", json={"llm": {"budgets": {"claude-haiku-5-5": {"1d": 5000}}}})
    assert r.status_code == 200
    row = next(x for x in client.get("/api/usage").json()["models"] if x["id"] == "claude-haiku-5-5")
    assert row["budget"]["1d"] == 5000 and row["budget"]["7d"] is None


def test_a_mission_with_a_disabled_or_unknown_model_is_refused(tmp_path):
    client, service = _client(tmp_path, lambda: _finish_brain())
    client.put("/api/settings", json={"llm": {"models": {"gpt-6-luna": {"enabled": False}}}})
    before = len(client.get("/api/missions").json())
    assert client.post("/api/missions", json={"goal": "check the pipeline", "model": "gpt-6-luna"}).status_code == 422
    assert client.post("/api/missions", json={"goal": "check the pipeline", "model": "not-a-model"}).status_code == 422
    assert len(client.get("/api/missions").json()) == before, "a refused mission must not be written"


def test_the_chosen_model_reaches_the_brain_factory_and_the_default_is_unchanged(tmp_path):
    calls: list = []

    def factory(model=None):
        calls.append(model)
        return _finish_brain()

    client, service = _client(tmp_path, factory)
    client.put("/api/settings", json={"llm": {"providers": {"claude": {"enabled": True}}}})
    chosen = client.post("/api/missions", json={"goal": "check the pipeline", "model": "claude-haiku-5-5"})
    assert chosen.status_code == 201
    service.wait(chosen.json()["id"], timeout=10)
    default = client.post("/api/missions", json={"goal": "check the pipeline"})
    assert default.status_code == 201
    service.wait(default.json()["id"], timeout=10)
    assert calls == ["claude-haiku-5-5", None]


def test_test_keys_stores_the_result_and_an_invalid_key_is_saved_as_such(tmp_path, monkeypatch):
    client, service = _client(tmp_path, lambda: _finish_brain())
    monkeypatch.setattr(app_module, "probe_provider", lambda name: [{"slot": 1, "status": 401, "ms": 2}])
    r = client.post("/api/settings/test-keys", params={"provider": "matilda"})
    assert r.status_code == 200
    row = r.json()["keys"][0]
    assert row["status"] == "invalid" and row["http_status"] == 401 and row["ok"] is False
    stored = json.loads(service.settings.path.read_text())
    assert stored["key_checks"]["matilda"][0]["status"] == "invalid"


def test_test_keys_with_no_key_configured_is_503_and_an_unknown_provider_is_422(tmp_path, monkeypatch):
    client, _ = _client(tmp_path, lambda: _finish_brain())

    def no_key(name):
        raise NotConfigured("no key configured for this provider")

    monkeypatch.setattr(app_module, "probe_provider", no_key)
    assert client.post("/api/settings/test-keys", params={"provider": "matilda"}).status_code == 503
    assert client.post("/api/settings/test-keys", params={"provider": "nope"}).status_code == 422
