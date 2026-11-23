"""GET /api/missions/{id}?after_step=N returns only the steps with index above N, over HTTP. The service-level cursor is tested
elsewhere; this checks that the route forwards it. No real keys, no network."""
from __future__ import annotations

from fastapi.testclient import TestClient

from prophet_agent.api.app import create_app
from prophet_agent.api.service import MissionService
from prophet_agent.tools import default_tools


def test_the_mission_route_returns_only_steps_after_the_cursor(tmp_path):
    service = MissionService(tmp_path / "api.sqlite", lambda: None, default_tools())
    try:
        mid = service.store.create_mission("forecast", {})
        for idx in range(4):
            service.store.finish_step(mid, idx, "decide", {"v": f"step {idx}"})
        with TestClient(create_app(service), base_url="http://127.0.0.1:8000") as client:
            full = client.get(f"/api/missions/{mid}").json()
            assert [s["idx"] for s in full["steps"]] == [0, 1, 2, 3]
            after = client.get(f"/api/missions/{mid}?after=0&after_step=1").json()
            assert [s["idx"] for s in after["steps"]] == [2, 3]
    finally:
        service.close()
