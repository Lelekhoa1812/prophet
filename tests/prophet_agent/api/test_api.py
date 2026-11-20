"""HTTP API tests (no real NIM or TimesFM). A scripted brain drives the goal loop; TimesFM is stubbed for the report tool.

These check the routes, the guards (host, origin, 503 without keys, artefact paths) and the question round trip. The
artefact test runs the real sandbox to build a forecast pack from the stubbed forecast.
"""
from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient

from prophet_agent.api.app import create_app
from prophet_agent.api.service import MissionService
from prophet_agent.forecast import timesfm as timesfm_mod
from prophet_agent.forecast.types import Forecast
from prophet_agent.nim.errors import NotConfigured
from prophet_agent.tools import default_tools


class ScriptedBrain:
    def __init__(self, plans, evals):
        self.plans, self.evals = list(plans), list(evals)

    def decide(self, messages, schema, name):
        return (self.plans if name == "plan" else self.evals).pop(0)


class _StubEngine:
    checkpoint = "stub"
    global_context = 64

    def forecast_batch(self, series_list, horizon):
        n = len(series_list)
        point = np.stack([np.full(horizon, float(np.asarray(x)[-1])) for x in series_list])
        q = np.tile(np.linspace(0, 8, 9), (n, horizon, 1))
        return Forecast(point, q)


@pytest.fixture(autouse=True)
def no_real_keys(monkeypatch):
    import prophet_agent.nim.transport as transport
    monkeypatch.setattr(transport, "load_keys", lambda *a, **k: [])


@pytest.fixture()
def stub_timesfm(monkeypatch):
    monkeypatch.setattr(timesfm_mod.TimesFMEngine, "get", classmethod(lambda cls: _StubEngine()))


def _client(tmp_path, brain_factory):
    service = MissionService(tmp_path / "api.sqlite", brain_factory, default_tools())
    return TestClient(create_app(service), base_url="http://127.0.0.1:8000"), service


def test_health_reports_no_keys(tmp_path):
    client, _ = _client(tmp_path, lambda: ScriptedBrain([], []))
    assert client.get("/api/health").json() == {"ok": True, "nim_keys_configured": False}


def test_mission_runs_to_completion_and_is_visible(tmp_path):
    brain = ScriptedBrain(plans=[{"action": "finish", "rationale": "done"}],
                          evals=[{"verdict": "met", "reason": "ok", "answer": "all good"}])
    client, service = _client(tmp_path, lambda: brain)
    r = client.post("/api/missions", json={"goal": "check the pipeline"})
    assert r.status_code == 201
    mid = r.json()["id"]
    service.wait(mid, timeout=10)
    snap = client.get(f"/api/missions/{mid}").json()
    assert snap["status"] == "completed"
    assert snap["answer"]["status"] == "met"
    assert any(e["kind"] == "mission.status" for e in snap["events"])
    assert client.get("/api/missions").json()[0]["id"] == mid


def test_question_round_trip_over_http(tmp_path):
    ask = {"action": "ask_user", "rationale": "need horizon", "questions": [
        {"question": "How far ahead?", "header": "Horizon", "multiSelect": False,
         "options": [{"label": "4 weeks", "description": ""}, {"label": "12 weeks", "description": ""}]}]}
    brain = ScriptedBrain(plans=[ask, {"action": "finish", "rationale": "x"}],
                          evals=[{"verdict": "met", "reason": "ok", "answer": "12-week plan"}])
    client, service = _client(tmp_path, lambda: brain)
    mid = client.post("/api/missions", json={"goal": "forecast demand"}).json()["id"]
    service.wait(mid, timeout=10)
    snap = client.get(f"/api/missions/{mid}").json()
    assert snap["status"] == "waiting_user"
    qid = snap["questions"][0]["id"]
    assert client.post(f"/api/questions/{qid}/answer", json={"answers": {"Horizon": "12 weeks"}}).status_code == 202
    service.wait(mid, timeout=10)
    assert client.get(f"/api/missions/{mid}").json()["status"] == "completed"


def test_no_keys_returns_503_and_does_not_run(tmp_path):
    def factory():
        raise NotConfigured("no keys")
    client, _ = _client(tmp_path, factory)
    r = client.post("/api/missions", json={"goal": "anything"})
    assert r.status_code == 503


def test_cross_origin_write_is_refused(tmp_path):
    client, _ = _client(tmp_path, lambda: ScriptedBrain([], []))
    r = client.post("/api/missions", json={"goal": "anything"}, headers={"Origin": "https://evil.example"})
    assert r.status_code == 403


def test_unknown_host_is_refused(tmp_path):
    client, _ = _client(tmp_path, lambda: ScriptedBrain([], []))
    r = client.get("/api/health", headers={"Host": "evil.example"})
    assert r.status_code == 400


def test_goal_validation(tmp_path):
    client, _ = _client(tmp_path, lambda: ScriptedBrain([], []))
    assert client.post("/api/missions", json={"goal": "x"}).status_code == 422
    assert client.post("/api/missions", json={"goal": "y" * 2001}).status_code == 422


def test_artifact_paths_are_confined(tmp_path):
    client, _ = _client(tmp_path, lambda: ScriptedBrain([], []))
    assert client.get("/api/artifacts/not-a-run-id/report.pdf").status_code == 404
    assert client.get("/api/artifacts/" + "a" * 32 + "/../../../etc/passwd").status_code == 404
    assert client.get("/api/artifacts/" + "a" * 32 + "/missing.png").status_code == 404


def test_forecast_report_produces_downloadable_files(tmp_path, stub_timesfm):
    values = [float(10 + (i % 5)) for i in range(40)]
    brain = ScriptedBrain(
        plans=[{"action": "call_tool", "tool": "forecast_report", "rationale": "pack",
                "args": {"values": values, "horizon": 4, "title": "Demand (synthetic)"}},
               {"action": "finish", "rationale": "x"}],
        evals=[{"verdict": "met", "reason": "files built", "answer": "pack ready"}])
    client, service = _client(tmp_path, lambda: brain)
    mid = client.post("/api/missions", json={"goal": "make a forecast pack"}).json()["id"]
    service.wait(mid, timeout=60)
    assert client.get(f"/api/missions/{mid}").json()["status"] == "completed"

    arts = client.get(f"/api/missions/{mid}/artifacts").json()
    assert len(arts) == 1
    run_id = arts[0]["run_id"]
    names = {f["path"] for f in arts[0]["files"]}
    assert "out/forecast.png" in names and "out/report.pdf" in names
    png = client.get(f"/api/artifacts/{run_id}/forecast.png")
    assert png.status_code == 200 and png.content[:8] == b"\x89PNG\r\n\x1a\n"
    assert arts[0]["forecast"]["config"]["checkpoint"] == "stub"
    assert len(arts[0]["forecast"]["point"]) == 4 and len(arts[0]["forecast"]["q10"]) == 4


def test_analysis_charts_are_listed_for_inline_display(tmp_path):
    code = (
        "import prophet_lib as pl, pandas as pd, numpy as np\n"
        "s = pd.Series(np.arange(20.0), index=pd.date_range('2024-01-01', periods=20, freq='W'))\n"
        "r = pl.chart('line', s, title='Demand rose every single week', subtitle='Units per week', unit='units')\n"
        "print(r['problems'])\n")
    scratch = "open('out/notes.txt', 'w').write('x')\nprint(1)\n"
    brain = ScriptedBrain(
        plans=[{"action": "call_tool", "tool": "sandbox_python", "rationale": "chart", "args": {"code": code}},
               {"action": "call_tool", "tool": "sandbox_python", "rationale": "scratch", "args": {"code": scratch}},
               {"action": "finish", "rationale": "x"}],
        evals=[{"verdict": "not_met", "reason": "more"}, {"verdict": "met", "reason": "done", "answer": "ok"}])
    client, service = _client(tmp_path, lambda: brain)
    mid = client.post("/api/missions", json={"goal": "chart it"}).json()["id"]
    service.wait(mid, timeout=60)
    arts = client.get(f"/api/missions/{mid}/artifacts").json()
    assert len(arts) == 1                                     # the run that only wrote a text note is not listed
    names = {f["path"] for f in arts[0]["files"]}
    assert {"out/chart.png", "out/chart.svg", "out/chart.csv"} <= names and arts[0]["forecast"] is None
    assert client.get(f"/api/artifacts/{arts[0]['run_id']}/chart.png").content[:4] == b"\x89PNG"


def test_finished_quick_forecasts_can_be_hidden_from_the_list(tmp_path):
    brain = ScriptedBrain(plans=[{"action": "finish", "rationale": "x"}],
                          evals=[{"verdict": "met", "reason": "ok", "answer": "done"}])
    client, service = _client(tmp_path, lambda: brain)
    mid = client.post("/api/missions", json={"goal": "say hello please"}).json()["id"]
    service.wait(mid, timeout=30)
    assert [m["id"] for m in client.get("/api/missions").json()] == [mid]
    assert client.patch(f"/api/missions/{mid}", json={"archived": True}).json() == {"id": mid, "archived": True}
    assert client.get("/api/missions").json() == []
    assert client.get(f"/api/missions/{mid}").status_code == 200          # hidden, not deleted
    assert client.patch(f"/api/missions/{mid}", json={"archived": False}).status_code == 200
    assert len(client.get("/api/missions").json()) == 1
    assert client.patch("/api/missions/nope", json={"archived": True}).status_code == 404
