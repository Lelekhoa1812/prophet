"""A notebook question takes the same model choice as a quick forecast: the chosen model reaches the planner for that
turn, the default applies when none is sent, and a disabled or unknown model is refused before a turn row is written.
A scripted brain stands in for the planner, so no test calls a real provider or reads a real key."""
from __future__ import annotations

from fastapi.testclient import TestClient

from prophet_agent.api.app import create_app
from prophet_agent.api.service import MissionService
from prophet_agent.tools import default_tools


class ScriptedBrain:
    def __init__(self, plans, evals):
        self.plans, self.evals = list(plans), list(evals)

    def decide(self, messages, schema, name):
        return (self.plans if name == "plan" else self.evals).pop(0)


def _finish_brain():
    return ScriptedBrain(plans=[{"action": "finish", "rationale": "done"}],
                         evals=[{"verdict": "met", "reason": "ok", "answer": "all good"}])


def _client(tmp_path, brain_factory):
    service = MissionService(tmp_path / "nb_model.sqlite", brain_factory, default_tools())
    return TestClient(create_app(service), base_url="http://127.0.0.1:8000"), service


def test_a_notebook_turn_with_a_chosen_model_reaches_the_brain_factory(tmp_path):
    calls: list = []

    def factory(model=None):
        calls.append(model)
        return _finish_brain()

    client, service = _client(tmp_path, factory)
    client.put("/api/settings", json={"llm": {"providers": {"claude": {"enabled": True}}}})
    nid = client.post("/api/notebooks", json={"title": "Model check"}).json()["id"]

    chosen = client.post(f"/api/notebooks/{nid}/turns", json={"goal": "summarise the file", "model": "claude-haiku-5-5"})
    assert chosen.status_code == 201
    service.wait(chosen.json()["id"], timeout=10)

    default = client.post(f"/api/notebooks/{nid}/turns", json={"goal": "what stands out"})
    assert default.status_code == 201
    service.wait(default.json()["id"], timeout=10)

    assert calls == ["claude-haiku-5-5", None]


def test_a_notebook_turn_with_a_disabled_or_unknown_model_is_refused_and_not_written(tmp_path):
    client, _ = _client(tmp_path, lambda: _finish_brain())
    client.put("/api/settings", json={"llm": {"models": {"gpt-6-luna": {"enabled": False}}}})
    nid = client.post("/api/notebooks", json={"title": "Refusals"}).json()["id"]

    assert client.post(f"/api/notebooks/{nid}/turns", json={"goal": "check the file", "model": "gpt-6-luna"}).status_code == 422
    assert client.post(f"/api/notebooks/{nid}/turns", json={"goal": "check the file", "model": "not-a-model"}).status_code == 422
    assert client.get(f"/api/notebooks/{nid}").json()["turns"] == [], "a refused question must not be written"
