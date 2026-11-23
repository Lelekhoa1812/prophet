"""Behaviour of per-notebook sharing (api/sharing.py) and its routes (round 6m).

Checks the PUT body rules, the tool filter for web research, and that a notebook which allows no provider cannot start a
turn. The routes are exercised against a real store in tmp_path.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from prophet_agent.api import titles
from prophet_agent.api.app import create_app
from prophet_agent.api.service import MissionService
from prophet_agent.api.sharing import DEFAULT_SHARING, RESEARCH_TOOL_NAMES, normalise_patch, tools_allowed
from prophet_agent.nim.providers import PROVIDERS
from prophet_agent.tools import default_tools

PROVIDER = next(iter(PROVIDERS))


@pytest.fixture(autouse=True)
def no_title_model(monkeypatch):
    monkeypatch.setattr(titles, "name_in_background", lambda *a, **k: None)


def test_an_empty_patch_changes_nothing():
    assert normalise_patch({}) == {}


def test_web_research_must_be_a_real_boolean():
    assert normalise_patch({"web_research": False}) == {"web_research": False}
    for bad in ("false", 0, 1, None):
        with pytest.raises(ValueError, match="web_research must be true or false"):
            normalise_patch({"web_research": bad})


def test_providers_are_deduplicated_and_sorted_and_null_means_all():
    assert normalise_patch({"providers": [PROVIDER, PROVIDER]}) == {"providers": [PROVIDER]}
    assert normalise_patch({"providers": None}) == {"providers": None}
    assert normalise_patch({"providers": []}) == {"providers": []}     # no provider may receive the data


def test_providers_must_be_known_names_in_a_list():
    with pytest.raises(ValueError, match="unknown provider: not_a_provider"):
        normalise_patch({"providers": [PROVIDER, "not_a_provider"]})
    for bad in (PROVIDER, [1], {"nim": True}):
        with pytest.raises(ValueError, match="providers must be a list"):
            normalise_patch({"providers": bad})


def test_an_unknown_field_is_refused_rather_than_ignored():
    with pytest.raises(ValueError, match="unknown field: model"):
        normalise_patch({"web_research": True, "model": "x"})


def test_tools_allowed_drops_every_research_tool_when_web_research_is_off():
    tools = {name: object() for name in (*RESEARCH_TOOL_NAMES, "forecast_report", "sandbox_python")}
    kept = tools_allowed(tools, {"web_research": False, "providers": None})
    assert set(kept) == {"forecast_report", "sandbox_python"}
    assert tools_allowed(tools, {"web_research": True, "providers": None}) is tools


def test_the_research_tool_set_is_taken_from_the_research_module():
    assert RESEARCH_TOOL_NAMES and all(isinstance(n, str) for n in RESEARCH_TOOL_NAMES)
    assert DEFAULT_SHARING == {"web_research": True, "providers": None}


# -- routes ---------------------------------------------------------------------------------------------------------

def _client(tmp_path):
    service = MissionService(tmp_path / "api.sqlite", lambda: None, default_tools())
    return TestClient(create_app(service), base_url="http://127.0.0.1:8000"), service


def _notebook(client):
    return client.post("/api/notebooks", json={"title": "sharing"}).json()["id"]


def test_sharing_defaults_keep_everything_on_and_list_every_provider(tmp_path):
    client, _ = _client(tmp_path)
    nid = _notebook(client)
    body = client.get(f"/api/notebooks/{nid}/sharing").json()
    assert body["web_research"] is True and body["providers"] is None
    assert {p["name"] for p in body["available"]} == set(PROVIDERS)


def test_a_partial_put_changes_only_the_fields_sent(tmp_path):
    client, _ = _client(tmp_path)
    nid = _notebook(client)
    client.put(f"/api/notebooks/{nid}/sharing", json={"providers": [PROVIDER]})
    r = client.put(f"/api/notebooks/{nid}/sharing", json={"web_research": False})
    assert r.status_code == 200
    body = client.get(f"/api/notebooks/{nid}/sharing").json()
    assert body["web_research"] is False and body["providers"] == [PROVIDER]


def test_a_refused_put_leaves_the_saved_sharing_as_it_was(tmp_path):
    client, _ = _client(tmp_path)
    nid = _notebook(client)
    client.put(f"/api/notebooks/{nid}/sharing", json={"web_research": False})
    r = client.put(f"/api/notebooks/{nid}/sharing", json={"web_research": True, "providers": ["not_a_provider"]})
    assert r.status_code == 422
    assert client.get(f"/api/notebooks/{nid}/sharing").json()["web_research"] is False


def test_sharing_routes_return_404_for_an_unknown_notebook(tmp_path):
    client, _ = _client(tmp_path)
    assert client.get("/api/notebooks/0123456789abcdef0123456789abcdef/sharing").status_code == 404
    r = client.put("/api/notebooks/0123456789abcdef0123456789abcdef/sharing", json={"web_research": False})
    assert r.status_code == 404


def test_a_notebook_that_allows_no_provider_cannot_start_a_turn(tmp_path):
    client, service = _client(tmp_path)
    nid = _notebook(client)
    client.put(f"/api/notebooks/{nid}/sharing", json={"providers": []})
    before = len(service.list(limit=500))
    r = client.post(f"/api/notebooks/{nid}/turns", json={"goal": "forecast the regional demand"})
    assert r.status_code == 409
    assert client.get(f"/api/notebooks/{nid}").json()["turns"] == []
    assert len(service.list(limit=500)) == before
