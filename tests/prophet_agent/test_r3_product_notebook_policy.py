"""Round 3, B6, goal 7 (A7-12): per-notebook data-sharing controls.

A notebook can switch web research off and limit which configured model providers may receive its data. Defaults keep
everything on. The rules are enforced by the API layer when a turn is created, when a mission starts, continues or is
answered now, and before automatic parts run. A scripted brain stands in for the planner, so no test calls a provider.
"""
from __future__ import annotations

from fastapi.testclient import TestClient

from prophet_agent.api import titles
from prophet_agent.api.app import create_app
from prophet_agent.api.service import MissionService
from prophet_agent.nim.providers import CATALOG
from prophet_agent.tools import default_tools
from prophet_agent.tools.research_tools import tools as research_tool_pairs

RESEARCH = {tool.name for tool, _ in research_tool_pairs()}


class ScriptedBrain:
    def __init__(self):
        self.calls: list = []

    def decide(self, messages, schema, name):
        if name == "plan":
            return {"action": "finish", "rationale": "done"}
        return {"verdict": "met", "reason": "ok", "answer": "all good"}


def _client(tmp_path):
    calls: list = []

    def factory(model=None):
        calls.append(model)
        return ScriptedBrain()

    service = MissionService(tmp_path / "policy.sqlite", factory, default_tools())
    client = TestClient(create_app(service), base_url="http://127.0.0.1:8000")
    client.put("/api/settings", json={"llm": {"providers": {"claude": {"enabled": True}, "nim": {"enabled": True}}}})
    return client, service, calls


def _model_of(provider: str) -> str:
    return next(m.id for m in CATALOG if m.provider == provider)


def _mission_turns(service, nid: str) -> int:
    return len(service.store.conn.execute("SELECT id FROM missions WHERE notebook_id=?", (nid,)).fetchall())


def test_defaults_keep_web_research_and_every_provider_on(tmp_path):
    client, service, _ = _client(tmp_path)
    nid = client.post("/api/notebooks", json={"title": "Defaults"}).json()["id"]
    sharing = client.get(f"/api/notebooks/{nid}/sharing").json()
    assert sharing["web_research"] is True
    assert sharing["providers"] is None
    turn = client.post(f"/api/notebooks/{nid}/turns", json={"goal": "summarise the file"}).json()
    service.wait(turn["id"], timeout=10)
    mission = service.store.mission(turn["id"])
    assert RESEARCH <= set(service.tools_for(mission))


def test_web_research_off_leaves_the_research_tools_out_of_the_mission(tmp_path):
    client, service, _ = _client(tmp_path)
    nid = client.post("/api/notebooks", json={"title": "No web"}).json()["id"]
    put = client.put(f"/api/notebooks/{nid}/sharing", json={"web_research": False})
    assert put.status_code == 200 and put.json()["web_research"] is False
    turn = client.post(f"/api/notebooks/{nid}/turns", json={"goal": "summarise the file"}).json()
    service.wait(turn["id"], timeout=10)
    names = set(service.tools_for(service.store.mission(turn["id"])))
    assert not (RESEARCH & names)
    assert "timesfm_forecast" in names      # only the research tools are removed


def test_web_research_off_is_enforced_for_a_continued_mission(tmp_path):
    client, service, _ = _client(tmp_path)
    nid = client.post("/api/notebooks", json={"title": "Continue"}).json()["id"]
    turn = client.post(f"/api/notebooks/{nid}/turns", json={"goal": "summarise the file"}).json()
    service.wait(turn["id"], timeout=10)
    client.put(f"/api/notebooks/{nid}/sharing", json={"web_research": False})
    assert not (RESEARCH & set(service.tools_for(service.store.mission(turn["id"]))))
    # The setting applies at the next part too: the tools are read from the live setting, not from the creation time.


def test_a_disallowed_provider_refuses_a_turn_before_any_row_is_written(tmp_path):
    client, service, calls = _client(tmp_path)
    nid = client.post("/api/notebooks", json={"title": "Providers"}).json()["id"]
    client.put(f"/api/notebooks/{nid}/sharing", json={"providers": ["claude"]})
    before = _mission_turns(service, nid)
    refused = client.post(f"/api/notebooks/{nid}/turns", json={"goal": "summarise", "model": _model_of("nim")})
    assert refused.status_code == 409
    assert "does not let" in refused.json()["detail"] or "allowed" in refused.json()["detail"]
    assert _mission_turns(service, nid) == before
    assert calls == []                      # the brain factory was never called for the refused turn


def test_an_allowed_provider_model_is_used_when_the_notebook_restricts_providers(tmp_path):
    client, service, calls = _client(tmp_path)
    nid = client.post("/api/notebooks", json={"title": "Allowed"}).json()["id"]
    client.put(f"/api/notebooks/{nid}/sharing", json={"providers": ["claude"]})
    model = _model_of("claude")
    turn = client.post(f"/api/notebooks/{nid}/turns", json={"goal": "summarise", "model": model})
    assert turn.status_code == 201
    service.wait(turn.json()["id"], timeout=10)
    assert calls == [model]


def test_with_no_allowed_provider_the_default_turn_is_refused(tmp_path):
    client, service, calls = _client(tmp_path)
    nid = client.post("/api/notebooks", json={"title": "Nothing allowed"}).json()["id"]
    client.put(f"/api/notebooks/{nid}/sharing", json={"providers": []})
    before = _mission_turns(service, nid)
    refused = client.post(f"/api/notebooks/{nid}/turns", json={"goal": "summarise"})
    assert refused.status_code == 409
    assert _mission_turns(service, nid) == before
    assert calls == []


def test_a_continue_after_the_provider_is_switched_off_is_refused_and_nothing_changes(tmp_path):
    client, service, _ = _client(tmp_path)
    nid = client.post("/api/notebooks", json={"title": "Continue refusal"}).json()["id"]
    turn = client.post(f"/api/notebooks/{nid}/turns", json={"goal": "summarise", "model": _model_of("nim")}).json()
    service.wait(turn["id"], timeout=10)
    service.store.set_status(turn["id"], "checkpoint")
    client.put(f"/api/notebooks/{nid}/sharing", json={"providers": ["claude"]})
    before = service.store.mission(turn["id"])["budget"]
    refused = client.post(f"/api/missions/{turn['id']}/continue")
    assert refused.status_code == 409
    assert service.store.mission(turn["id"])["budget"] == before


def test_a_restricted_provider_list_does_not_send_the_goal_to_the_naming_model(tmp_path, monkeypatch):
    client, service, _ = _client(tmp_path)
    sent: list = []
    monkeypatch.setattr(titles, "name_in_background", lambda text, apply, *a, **k: sent.append(text))
    nid = client.post("/api/notebooks", json={"title": "New notebook"}).json()["id"]
    client.put(f"/api/notebooks/{nid}/sharing", json={"providers": ["claude"]})
    turn = client.post(f"/api/notebooks/{nid}/turns", json={"goal": "forecast the quarterly sales", "model": _model_of("claude")})
    assert turn.status_code == 201
    service.wait(turn.json()["id"], timeout=10)
    assert sent == []


def test_the_sharing_settings_are_validated(tmp_path):
    client, _, _ = _client(tmp_path)
    nid = client.post("/api/notebooks", json={"title": "Validate"}).json()["id"]
    assert client.put(f"/api/notebooks/{nid}/sharing", json={"providers": ["not-a-provider"]}).status_code == 422
    assert client.put(f"/api/notebooks/{nid}/sharing", json={"web_research": "maybe"}).status_code == 422
    assert client.put("/api/notebooks/missing/sharing", json={"web_research": False}).status_code == 404
    assert client.get("/api/notebooks/missing/sharing").status_code == 404


def test_the_sharing_view_lists_every_provider_with_its_state(tmp_path):
    client, _, _ = _client(tmp_path)
    nid = client.post("/api/notebooks", json={"title": "View"}).json()["id"]
    view = client.get(f"/api/notebooks/{nid}/sharing").json()
    names = [p["name"] for p in view["available"]]
    assert "nim" in names and "claude" in names
    assert all({"name", "label", "ready"} <= set(p) for p in view["available"])
