"""The console's right-hand panel (owner, 2026-11-27): the main thread, sub-agents and lanes with their briefs, files without
duplicates grouped by question, the tools and skills this round used, and one sub-agent's whole session. A scripted brain
stands in for the planner, so nothing here calls a real provider."""
from __future__ import annotations

import threading

from fastapi.testclient import TestClient

from prophet_agent.api import activity
from prophet_agent.api.app import create_app
from prophet_agent.api.service import MissionService
from prophet_agent.tools import default_tools


class LaneBrain:
    """The lead fans out two lanes; each lane reads a playbook, then finishes. Lanes are told apart by their Task line."""

    def __init__(self):
        self.lock = threading.Lock()
        self.lead = [{"action": "lanes", "rationale": "two angles", "lanes": [
            {"task": "angle one: the RBA decisions", "agent_type": "researcher"},
            {"task": "angle two: market pricing", "agent_type": "researcher"}]},
            {"action": "finish", "rationale": "done"}]
        self.lane_calls: dict[str, int] = {}

    def decide(self, messages, schema, name):
        with self.lock:
            head = messages[1]["content"].split("Available tools")[0]
            task = head.split("Task: ", 1)[1].split("\n", 1)[0] if "Task: " in head else ""
            if name != "plan":
                return {"verdict": "met", "reason": "enough", "answer": "a lane report" if task else "final answer",
                        "findings": ["a finding"]}
            if task:
                k = self.lane_calls.get(task, 0)
                self.lane_calls[task] = k + 1
                if k == 0:
                    return {"action": "call_tool", "tool": "analyst_skill", "args": {"name": "choose-model-by-backtest"},
                            "rationale": "read the playbook"}
                return {"action": "finish", "rationale": "done"}
            return self.lead.pop(0) if self.lead else {"action": "finish", "rationale": "done"}


def _client(tmp_path):
    service = MissionService(tmp_path / "m.sqlite", lambda: LaneBrain(), default_tools())
    return TestClient(create_app(service), base_url="http://127.0.0.1:8000"), service


def test_the_panel_lists_lanes_with_briefs_tools_and_skills(tmp_path):
    client, service = _client(tmp_path)
    r = client.post("/api/missions", json={"goal": "What will the RBA cash rate be?"})
    assert r.status_code == 201, r.text
    mid = r.json()["id"]
    service.wait(mid, timeout=20)
    got = client.get(f"/api/missions/{mid}/activity").json()
    assert got["main"]["mission_id"] == mid and got["main"]["title"].startswith("What will the RBA")
    agents = {a["agent_id"]: a for a in got["agents"]}
    assert set(agents) == {"a1", "a2"}
    assert agents["a1"]["task"] == "angle one: the RBA decisions" and agents["a1"]["lane"] is True
    assert all(a["status"] == "met" and a["tool_calls"] == 1 for a in agents.values()), agents
    assert {"name": "analyst_skill", "title": next(t.title for t in service.tools.values() if t.name == "analyst_skill"),
            "calls": 2} in got["tools"]
    skills = {s["name"]: s["source"] for s in got["skills"]}
    assert skills.get("choose-model-by-backtest") == "read by the planner"
    assert "web-research-playbook" in skills, "the researcher type's playbook is loaded into each lane"


def test_one_sub_agent_session_reads_in_order(tmp_path):
    client, service = _client(tmp_path)
    mid = client.post("/api/missions", json={"goal": "What will the RBA cash rate be?"}).json()["id"]
    service.wait(mid, timeout=20)
    s = client.get(f"/api/missions/{mid}/agents/a2").json()
    assert s["task"] == "angle two: market pricing" and s["agent_type"] == "researcher" and s["lane"] is True
    roles = [e["role"] for e in s["entries"]]
    assert roles.index("plan") < roles.index("result") < roles.index("evaluation"), roles
    result = next(e for e in s["entries"] if e["role"] == "result")
    assert result["tool"] == "analyst_skill" and result["ok"] and "choose-model-by-backtest" in result["text"]
    assert client.get(f"/api/missions/{mid}/agents/a9").status_code == 404
    assert client.get(f"/api/missions/{mid}/agents/lead").status_code == 404
    assert client.get("/api/missions/nope/activity").status_code == 404


def test_the_start_event_is_written_once_per_sub_agent(tmp_path):
    client, service = _client(tmp_path)
    mid = client.post("/api/missions", json={"goal": "What will the RBA cash rate be?"}).json()["id"]
    service.wait(mid, timeout=20)
    started = [e for e in service.store.events(mid) if e["kind"] == "agent.started"]
    assert sorted(e["payload"]["agent_id"] for e in started) == ["a1", "a2"]


def test_files_are_listed_once_under_the_newest_question(tmp_path, monkeypatch):
    client, service = _client(tmp_path)
    nb = client.post("/api/notebooks", json={"title": "files"}).json()["id"]
    first = client.post(f"/api/notebooks/{nb}/turns", json={"goal": "first question"}).json()["id"]
    service.wait(first, timeout=20)
    second = client.post(f"/api/notebooks/{nb}/turns", json={"goal": "second question"}).json()["id"]
    service.wait(second, timeout=20)
    rows = {first: [{"path": "out/chart.png", "sha256": "aa"}, {"path": "out/table.csv", "sha256": "bb"}],
            second: [{"path": "out/chart.png", "sha256": "aa"}, {"path": "out/notes.md", "sha256": "cc"}]}

    def fake_calls(store, mission_id):
        return [{"step": 1, "tool": "sandbox_python", "args": {"code": "# Plot it\n"}, "status": "done",
                 "result": {"ok": True, "run_id": "f" * 32, "outputs": rows.get(mission_id, [])}, "error": None, "updated": 0}]

    monkeypatch.setattr(activity, "_calls", fake_calls)
    monkeypatch.setattr(activity, "_files", lambda calls, root, merge: [
        {"name": f["path"][4:], "path": f["path"], "run_id": "f" * 32, "sha256": f["sha256"], "bytes": 1,
         "kind": activity._file_kind(f["path"]), "url": "", "step": 1} for c in calls for f in c["result"]["outputs"]])
    got = client.get(f"/api/notebooks/{nb}/activity").json()
    groups = [(g["topic"], sorted(f["name"] for f in g["files"])) for g in got["files"]]
    assert groups == [("second question", ["chart.png", "notes.md"]), ("first question", ["table.csv"])]
    assert got["code"][0]["title"] == "Plot it"


def test_a_chart_saved_as_png_and_svg_is_listed_once(tmp_path, monkeypatch):
    client, service = _client(tmp_path)
    nb = client.post("/api/notebooks", json={"title": "twins"}).json()["id"]
    turn = client.post(f"/api/notebooks/{nb}/turns", json={"goal": "a chart"}).json()["id"]
    service.wait(turn, timeout=20)
    rows = [{"path": "out/fc.png", "sha256": "aa"}, {"path": "out/fc.svg", "sha256": "bb"},
            {"path": "out/other.svg", "sha256": "cc"}, {"path": "out/fc.csv", "sha256": "dd"}]
    monkeypatch.setattr(activity, "_calls", lambda store, mid: [{"step": 1, "tool": "sandbox_python", "args": {}, "status": "done",
                                                                  "result": {"ok": True, "run_id": "f" * 32}, "error": None,
                                                                  "updated": 0}])
    monkeypatch.setattr(activity, "_files", lambda calls, root, merge: [
        {"name": f["path"][4:], "path": f["path"], "run_id": "f" * 32, "sha256": f["sha256"], "bytes": 1,
         "kind": activity._file_kind(f["path"]), "url": "", "step": 1} for f in rows])
    got = client.get(f"/api/notebooks/{nb}/activity").json()
    names = sorted(f["name"] for g in got["files"] for f in g["files"])
    assert names == ["fc.csv", "fc.png", "other.svg"]       # the PNG stands for its SVG twin; a different chart still shows
