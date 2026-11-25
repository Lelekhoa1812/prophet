"""Continuing a mission does not cap the lead's tool calls. The delegated lane is still topped up the way the runtime
counts it (round 3, A5-01): read-only calls are not charged, and a stored lead cap is removed."""
from prophet_agent.api.service import MissionService
from prophet_agent.runtime.runner import DELEGATED_CALLS


def test_top_up_drops_the_lead_cap_and_tops_up_delegated_calls(tmp_path):
    svc = MissionService(tmp_path / "m.sqlite", brain_factory=lambda *a, **k: None, tools={})
    mid = svc.store.create_mission("g", {})
    calls = [("sandbox_python", "lead"), ("sandbox_python", "lead"), ("analyst_skill", "lead"),
             ("sandbox_python", "delegated"), ("web_research", "delegated"), ("notes", "delegated")]
    for n, (tool, lane) in enumerate(calls, start=1):
        svc.store.tool_start(mid, n, tool, {"n": n}, f"key{n}", lane=lane)
        svc.store.tool_set(f"key{n}", "done", result={})
    budget: dict = {"max_tool_calls": 20}
    svc._top_up(mid, budget)
    assert "max_tool_calls" not in budget
    assert budget["max_delegated_calls"] == 2 + DELEGATED_CALLS


def test_a_start_drops_a_stored_lead_cap(tmp_path):
    svc = MissionService(tmp_path / "m.sqlite", brain_factory=lambda *a, **k: None, tools={})
    mid = svc.store.create_mission("g", {"max_tool_calls": 20, "max_seconds": 60})
    svc._resume_program(svc.store.mission(mid))
    assert "max_tool_calls" not in svc.store.mission(mid)["budget"]
    assert svc.store.mission(mid)["budget"]["max_seconds"] == 60
