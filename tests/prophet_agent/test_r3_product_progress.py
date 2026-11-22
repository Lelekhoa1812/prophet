"""Round 3, B6: latency is shown honestly and progress is visible (A7-07).

The copy quotes the measured range from the 58 stored missions (median 557 s, longest 1,756 s), and a running mission
exposes its checklist progress and next item, which the console shows beside the elapsed time.
"""
from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from prophet_agent.api.app import create_app
from prophet_agent.api.service import MissionService
from prophet_agent.tools import default_tools

ROOT = Path(__file__).resolve().parents[2]


def _client(tmp_path):
    service = MissionService(tmp_path / "api.sqlite", lambda: None, default_tools())
    return TestClient(create_app(service), base_url="http://127.0.0.1:8000"), service


def test_a_running_mission_exposes_its_checklist_progress_and_next_item(tmp_path):
    client, service = _client(tmp_path)
    mid = service.store.create_mission("forecast the next 12 weeks", {"max_tool_calls": 20})
    service.store.append_event(mid, "mission.progress", {"step": 2, "tool_calls": 3, "elapsed_s": 40.0, "checklist_done": 1,
                                                         "checklist_total": 4, "next": "Backtest the forecast"})
    service.store.append_event(mid, "mission.progress", {"step": 5, "tool_calls": 9, "elapsed_s": 96.5, "checklist_done": 2,
                                                         "checklist_total": 4, "next": "Write the answer"})
    service.store.set_status(mid, "running")
    body = client.get(f"/api/missions/{mid}").json()
    assert body["progress"] == {"step": 5, "tool_calls": 9, "elapsed_s": 96.5, "checklist_done": 2, "checklist_total": 4,
                                "next": "Write the answer"}


def test_progress_is_null_before_the_first_status_line(tmp_path):
    client, service = _client(tmp_path)
    mid = service.store.create_mission("forecast", {"max_tool_calls": 20})
    assert client.get(f"/api/missions/{mid}").json()["progress"] is None


def test_the_latency_copy_quotes_the_measured_range_and_has_no_vague_estimate():
    api_md = (ROOT / "API.md").read_text(encoding="utf-8")
    console = "\n".join(p.read_text(encoding="utf-8") for p in (ROOT / "web" / "src").rglob("*.tsx"))
    for text in (api_md, console):
        assert "a few minutes" not in text
    assert "about 9 minutes" in api_md and "about 29 minutes" in api_md
    assert "median of about 9 minutes, and up to about 29 minutes" in console
