"""Round 3 follow-up (A5-01): continuing a mission tops up each lane the way the runtime counts it. Before, the lead's new
allowance was computed from every call in the mission (sub-agents' and read-only ones included), so it did not match the
lead-only check it is held to, and the delegated grant was never topped up."""
from prophet_agent.api.service import MissionService
from prophet_agent.runtime.runner import DELEGATED_CALLS


def test_top_up_counts_each_lane_like_the_runtime(tmp_path):
    svc = MissionService(tmp_path / "m.sqlite", brain_factory=lambda *a, **k: None, tools={})
    mid = svc.store.create_mission("g", {})
    calls = [("sandbox_python", "lead"), ("sandbox_python", "lead"), ("analyst_skill", "lead"),
             ("sandbox_python", "delegated"), ("web_research", "delegated"), ("notes", "delegated")]
    for n, (tool, lane) in enumerate(calls, start=1):
        svc.store.tool_start(mid, n, tool, {"n": n}, f"key{n}", lane=lane)
        svc.store.tool_set(f"key{n}", "done", result={})
    budget: dict = {}
    svc._top_up(mid, budget)
    assert budget["max_tool_calls"] == 2 + int(svc.default_budget()["max_tool_calls"])
    assert budget["max_delegated_calls"] == 2 + DELEGATED_CALLS
