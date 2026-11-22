"""Round 3, B6: a panel's per-entity results are readable in the console (A7-09).

The ledger keeps every entity of a ``panel_backtest`` result (the planner only sees the first 30). The API pages through
them, sorted by name or by any model's MASE, with the best model for each entity worked out from the stored numbers.
"""
from __future__ import annotations

from fastapi.testclient import TestClient

from prophet_agent.api.app import create_app
from prophet_agent.api.service import MissionService
from prophet_agent.tools import default_tools


def _client(tmp_path):
    service = MissionService(tmp_path / "api.sqlite", lambda: None, default_tools())
    return TestClient(create_app(service), base_url="http://127.0.0.1:8000"), service


def _panel_result(n: int) -> dict:
    models = ["naive", "seasonal_naive", "global_gbm", "timesfm"]
    per_entity = {}
    for i in range(n):
        # Deterministic MASE values; the best model rotates so the best-model column is not constant.
        base = 1.0 + (i % 7) * 0.05
        per_entity[f"Station {i:02d}"] = {"naive": base + 0.4, "seasonal_naive": base + 0.2,
                                          "global_gbm": base + (0.1 if i % 3 else -0.3), "timesfm": base + 0.15}
    return {"horizon": 12, "origins": 6, "season": 52, "overall": {"naive": 1.4, "timesfm": 1.2},
            "wins": {"timesfm": 3}, "best_overall": "timesfm", "coverage_80": {"timesfm": 0.8}, "notes": [],
            "reading": "a stored reading", "basis": "measured", "per_entity": per_entity, "models": models}


def _store_panel(service, mission_id: str, result: dict, step: int = 3, key: str = "k1") -> None:
    service.store.tool_start(mission_id, step, "panel_backtest", {"horizon": 12}, key)
    service.store.tool_set(key, "done", result)


def test_the_panel_runs_of_a_mission_are_listed_with_their_size_and_models(tmp_path):
    client, service = _client(tmp_path)
    mid = service.store.create_mission("panel", {"max_tool_calls": 20})
    _store_panel(service, mid, _panel_result(25))
    runs = client.get(f"/api/missions/{mid}/panels").json()
    assert len(runs) == 1
    assert runs[0]["step"] == 3
    assert runs[0]["entities"] == 25
    assert runs[0]["models"] == ["naive", "seasonal_naive", "global_gbm", "timesfm"]
    assert runs[0]["best_overall"] == "timesfm"


def test_entities_are_paged_with_the_best_model_and_every_mase(tmp_path):
    client, service = _client(tmp_path)
    mid = service.store.create_mission("panel", {"max_tool_calls": 20})
    result = _panel_result(25)
    _store_panel(service, mid, result)
    page1 = client.get(f"/api/missions/{mid}/panels/3/entities", params={"page": 1, "page_size": 10}).json()
    assert page1["total"] == 25 and page1["page"] == 1 and page1["pages"] == 3
    assert len(page1["rows"]) == 10
    assert page1["models"] == ["naive", "seasonal_naive", "global_gbm", "timesfm"]
    row = page1["rows"][0]
    assert row["entity"] == "Station 00"
    assert row["mase"] == result["per_entity"]["Station 00"]
    best = min(row["mase"], key=row["mase"].get)
    assert row["best_model"] == best
    page3 = client.get(f"/api/missions/{mid}/panels/3/entities", params={"page": 3, "page_size": 10}).json()
    assert [r["entity"] for r in page3["rows"]] == [f"Station {i:02d}" for i in range(20, 25)]


def test_entities_can_be_sorted_by_a_model_mase(tmp_path):
    client, service = _client(tmp_path)
    mid = service.store.create_mission("panel", {"max_tool_calls": 20})
    result = _panel_result(25)
    _store_panel(service, mid, result)
    rows = client.get(f"/api/missions/{mid}/panels/3/entities",
                      params={"sort": "mase:timesfm", "order": "desc", "page_size": 25}).json()["rows"]
    values = [r["mase"]["timesfm"] for r in rows]
    assert values == sorted(values, reverse=True)


def test_bad_paging_and_unknown_runs_are_refused(tmp_path):
    client, service = _client(tmp_path)
    mid = service.store.create_mission("panel", {"max_tool_calls": 20})
    _store_panel(service, mid, _panel_result(5))
    url = f"/api/missions/{mid}/panels/3/entities"
    assert client.get(url, params={"page_size": 0}).status_code == 422
    assert client.get(url, params={"page_size": 500}).status_code == 422
    assert client.get(url, params={"sort": "mase:nope"}).status_code == 422
    assert client.get(f"/api/missions/{mid}/panels/99/entities").status_code == 404
    assert client.get("/api/missions/nope/panels").status_code == 404
