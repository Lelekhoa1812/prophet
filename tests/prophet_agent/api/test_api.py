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
    ask = {"action": "ask_user", "rationale": "high-risk: the horizon sets every delivered number and the data cannot say it", "questions": [
        {"question": "How far ahead?", "header": "Horizon", "multiSelect": False,
         "options": [{"label": "4 weeks", "description": ""}, {"label": "12 weeks", "description": ""}]}]}
    # A5-04: a forecast goal's first two finishes are refused for checklist items still to do; the third reaches the evaluator.
    brain = ScriptedBrain(plans=[ask, {"action": "finish", "rationale": "x"}, {"action": "finish", "rationale": "x"}, {"action": "finish", "rationale": "x"}],
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


def test_forecast_report_produces_downloadable_files(tmp_path, stub_timesfm, monkeypatch):
    # This test is about the files, not the analysis: the finish gate has its own tests in test_goal_loop.py.
    monkeypatch.setattr("prophet_agent.runtime.goal.contracts.check_evidence", lambda *a, **k: [])
    monkeypatch.setattr("prophet_agent.runtime.prediction_quality.check_fundamentals", lambda *a, **k: None)
    values = [float(10 + (i % 5)) for i in range(40)]
    brain = ScriptedBrain(
        plans=[{"action": "call_tool", "tool": "forecast_report", "rationale": "pack",
                "args": {"values": values, "horizon": 4, "title": "Demand (synthetic)"}},
               {"action": "finish", "rationale": "x"}, {"action": "finish", "rationale": "x"}, {"action": "finish", "rationale": "x"}],  # A5-04: two refused finishes first
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


def test_an_interrupted_mission_is_marked_at_startup_and_can_be_resumed(tmp_path):
    # A previous process died mid-mission: its row says running, its lease has lapsed.
    seed = MissionService(tmp_path / "api.sqlite", lambda: ScriptedBrain([], []), default_tools())
    mid = seed.store.create_mission("forecast", {})
    seed.store.acquire_lease(mid, "dead", ttl_s=-1)
    seed.store.set_status(mid, "running")
    seed.store.close()

    # A5-04: the forecast goal's finish is refused twice for checklist items still to do before the evaluator is asked.
    brain = ScriptedBrain([{"action": "finish", "rationale": "nothing to do"}] * 3,
                          [{"verdict": "met", "reason": "ok", "answer": "Nothing was needed."}])
    client, service = _client(tmp_path, lambda: brain)
    assert service.interrupted_at_start == [mid]
    assert client.get(f"/api/missions/{mid}").json()["status"] == "interrupted"
    assert client.post(f"/api/missions/{mid}/resume").status_code == 202
    service.wait(mid, timeout=10)
    assert client.get(f"/api/missions/{mid}").json()["status"] == "completed"


def test_only_an_interrupted_mission_can_be_resumed(tmp_path):
    client, service = _client(tmp_path, lambda: ScriptedBrain([], []))
    mid = service.store.create_mission("forecast", {})
    service.store.set_status(mid, "completed")
    assert client.post(f"/api/missions/{mid}/resume").status_code == 409
    assert client.post("/api/missions/does-not-exist/resume").status_code == 404


def test_a_mission_orphaned_inside_the_lease_window_is_interrupted_once_the_lease_lapses(tmp_path):
    client, service = _client(tmp_path, lambda: ScriptedBrain([], []))
    mid = service.store.create_mission("forecast", {})
    service.store.acquire_lease(mid, "dead-process", ttl_s=300)          # the dead worker's lease is still valid at restart
    service.store.set_status(mid, "running")
    assert service.sweep() == [] and client.get(f"/api/missions/{mid}").json()["status"] == "running"
    service.store.acquire_lease(mid, "dead-process", ttl_s=-1)           # the window passes
    assert service.sweep() == [mid]
    assert client.get(f"/api/missions/{mid}").json()["status"] == "interrupted"
    assert client.post(f"/api/missions/{mid}/resume").status_code == 202


def test_the_sweep_never_touches_a_mission_this_process_is_running(tmp_path):
    import threading
    client, service = _client(tmp_path, lambda: ScriptedBrain([], []))
    mid = service.store.create_mission("forecast", {})
    service.store.acquire_lease(mid, "me", ttl_s=-1)                     # lapsed only because a long step has not renewed it
    service.store.set_status(mid, "running")
    hold = threading.Event()
    t = threading.Thread(target=hold.wait, daemon=True)
    t.start()
    service._threads[mid] = t
    try:
        assert service.sweep() == []
        assert client.get(f"/api/missions/{mid}").json()["status"] == "running"
    finally:
        hold.set()


def test_a_checkpointed_mission_continues_into_its_next_part_and_completes(tmp_path):
    """A part that ends in a checkpoint is continued from the ledger, not restarted, and reaches a verdict."""
    from prophet_agent.runtime.goal import GoalConfig
    brain = ScriptedBrain([{"action": "finish", "rationale": "a"}, {"action": "finish", "rationale": "b"}],
                          [{"verdict": "not_met", "reason": "need more"}, {"verdict": "met", "reason": "ok", "answer": "Done."}])
    service = MissionService(tmp_path / "api.sqlite", lambda: brain, default_tools(), GoalConfig(max_iterations=1))
    client = TestClient(create_app(service), base_url="http://127.0.0.1:8000")
    mid = service.store.create_mission("forecast", {"auto_continue": 0})
    service.start(mid)
    service.wait(mid, timeout=10)
    assert client.get(f"/api/missions/{mid}").json()["status"] == "checkpoint"

    assert client.post(f"/api/missions/{mid}/continue", ).status_code == 202
    service.wait(mid, timeout=10)
    body = client.get(f"/api/missions/{mid}").json()
    assert body["status"] == "completed" and body["answer"]["status"] == "met"
    assert brain.plans == [], "only the one new step was asked of the planner"
    assert client.post(f"/api/missions/{mid}/continue").status_code == 409, "a finished mission has nothing to continue"


def test_answer_now_writes_the_best_answer_from_a_checkpoint_as_partial(tmp_path):
    from prophet_agent.runtime.goal import GoalConfig
    brain = ScriptedBrain([{"action": "finish", "rationale": "a"}], [{"verdict": "not_met", "reason": "need more"}])
    service = MissionService(tmp_path / "api.sqlite", lambda: brain, default_tools(), GoalConfig(max_iterations=1))
    client = TestClient(create_app(service), base_url="http://127.0.0.1:8000")
    mid = service.store.create_mission("forecast", {"auto_continue": 0})
    service.start(mid)
    service.wait(mid, timeout=10)
    assert client.post(f"/api/missions/{mid}/answer-now").status_code == 202
    service.wait(mid, timeout=10)
    assert client.get(f"/api/missions/{mid}").json()["status"] == "partial"
    assert client.post("/api/missions/does-not-exist/continue").status_code == 404


def test_a_durable_mission_starts_its_next_part_by_itself_and_reaches_a_verdict(tmp_path):
    """The harness keeps a mission going: a spent part is followed by the next without the person pressing Continue."""
    from prophet_agent.runtime.goal import GoalConfig
    brain = ScriptedBrain([{"action": "finish", "rationale": "a"}, {"action": "finish", "rationale": "b"}],
                          [{"verdict": "not_met", "reason": "need more"}, {"verdict": "met", "reason": "ok", "answer": "Done."}])
    service = MissionService(tmp_path / "api.sqlite", lambda: brain, default_tools(), GoalConfig(max_iterations=1))
    mid = service.store.create_mission("forecast", {})
    service.start(mid)
    service.wait(mid, timeout=10)
    snap = service.snapshot(mid)
    assert snap["status"] == "completed" and snap["answer"]["status"] == "met"
    assert brain.plans == [], "the stored first part replayed; only the new step was planned"
    assert [e["payload"]["automatic_parts_used"] for e in snap["events"] if e["kind"] == "mission.auto_continue"] == [1]


def test_automatic_parts_are_capped_and_the_person_gets_the_checkpoint_after_them(tmp_path):
    from prophet_agent.runtime.goal import GoalConfig
    brain = ScriptedBrain([{"action": "finish", "rationale": "a"}] * 3, [{"verdict": "not_met", "reason": "need more"}] * 3)
    service = MissionService(tmp_path / "api.sqlite", lambda: brain, default_tools(), GoalConfig(max_iterations=1))
    counter = iter(range(1, 100))
    service._checklist_done = lambda mission_id: next(counter)      # the checklist advances in every part
    mid = service.store.create_mission("forecast", {"auto_continue": 2})
    service.start(mid)
    service.wait(mid, timeout=10)
    snap = service.snapshot(mid)
    assert snap["status"] == "checkpoint", "after the automatic parts the person decides: Continue or Answer now"
    assert sum(1 for e in snap["events"] if e["kind"] == "mission.auto_continue") == 2
    assert snap["budget"]["auto_continued"] == 2


def test_automatic_parts_keep_going_without_checklist_progress_up_to_a_set_cap(tmp_path):
    """No boundary (owner, 2026-11-27): a part that added nothing is recorded, not a stop. Only a cap the person set ends it."""
    from prophet_agent.runtime.goal import GoalConfig
    from prophet_agent.api import service as service_mod
    assert service_mod.AUTO_CONTINUE_PARTS is None, "no default limit on automatic parts"
    brain = ScriptedBrain([{"action": "finish", "rationale": "a"}] * 3, [{"verdict": "not_met", "reason": "need more"}] * 3)
    service = MissionService(tmp_path / "api.sqlite", lambda: brain, default_tools(), GoalConfig(max_iterations=1))
    service._checklist_done = lambda mission_id: 2                  # never advances
    mid = service.store.create_mission("forecast", {"auto_continue": 2})
    service.start(mid)
    service.wait(mid, timeout=10)
    snap = service.snapshot(mid)
    assert snap["status"] == "checkpoint"
    assert sum(1 for e in snap["events"] if e["kind"] == "mission.auto_continue") == 2
    assert any(e["kind"] == "mission.auto_continue_no_progress" for e in snap["events"])


def test_settings_show_masked_keys_and_bounded_limits_and_refuse_bad_input(tmp_path):
    client, service = _client(tmp_path, lambda: ScriptedBrain([], []))
    body = client.get("/api/settings").json()
    assert body["nim"]["configured"] is False and body["nim"]["keys"] == []
    assert body["agent"]["part_steps"] == 200 and body["timesfm"]["checkpoint"]
    assert "licence" not in body["about"] and "licence" not in body["timesfm"]  # the notice lives in NOTICE.md, not the console
    assert body["bounds"]["agent"]["part_steps"] == [5, 1000], "the page reads its limits from the server"
    assert body["sandbox"]["timeout_s"] == 20 and body["sandbox"]["mem_mb"] == 1024
    assert isinstance(body["timesfm"]["engine_installed"], bool)

    assert client.put("/api/settings", json={"agent": {"part_steps": 40}}).json()["agent"]["part_steps"] == 40
    assert client.put("/api/settings", json={"agent": {"part_steps": 99999}}).status_code == 422
    assert client.put("/api/settings", json={"nim": {"unknown": 1}}).status_code == 422
    assert client.post("/api/settings/test-keys").status_code == 503, "no keys, so nothing to test"


def test_clear_data_needs_the_phrase_and_export_carries_no_keys(tmp_path):
    client, service = _client(tmp_path, lambda: ScriptedBrain([], []))
    mid = service.store.create_mission("forecast demand", {})
    service.store.set_status(mid, "completed")
    assert client.post("/api/settings/clear-data", json={"confirm": "yes"}).status_code == 422
    export = client.get("/api/settings/export")
    assert export.status_code == 200 and export.headers["content-disposition"].startswith("attachment")
    assert "forecast demand" in export.text and "nvapi-" not in export.text
    running = service.store.create_mission("still going", {})
    service.store.set_status(running, "running")
    assert client.post("/api/settings/clear-data", json={"confirm": "delete all data"}).status_code == 409
    service.store.set_status(running, "completed")
    assert client.post("/api/settings/clear-data", json={"confirm": "delete all data"}).status_code == 200
    assert client.get(f"/api/missions/{mid}").status_code == 404
