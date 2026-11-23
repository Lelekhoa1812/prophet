"""Round 6m: MissionService had no shutdown path. The orphan sweeper is a daemon thread that never stopped, so it kept
reading a store its owner had finished with. close() stops it, and the app's lifespan calls close() on shutdown.
No real keys, no repo-root .env: every service here uses its own tmp_path database."""
from __future__ import annotations

from fastapi.testclient import TestClient

from prophet_agent.api.app import create_app
from prophet_agent.api.service import MissionService
from prophet_agent.tools import default_tools


def _service(tmp_path):
    return MissionService(tmp_path / "api.sqlite", lambda: None, default_tools())


def test_close_stops_the_orphan_sweeper_and_can_be_called_twice(tmp_path):
    service = _service(tmp_path)
    assert service._sweeper.is_alive()
    service.close()
    assert not service._sweeper.is_alive()
    service.close()                                  # a second call is harmless


def test_leaving_the_app_stops_the_orphan_sweeper(tmp_path):
    service = _service(tmp_path)
    with TestClient(create_app(service), base_url="http://127.0.0.1:8000") as client:
        assert client.get("/api/health").status_code == 200
        assert service._sweeper.is_alive()
    assert not service._sweeper.is_alive()
