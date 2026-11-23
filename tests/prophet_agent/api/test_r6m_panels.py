"""Behaviour of the panel view (api/panels.py): summaries, paging, sorting and the routes that serve them (round 6m).

The result dicts are shaped like a stored panel_backtest result: per_entity maps an entity to MASE per model.
"""
from __future__ import annotations

import math

import pytest
from fastapi.testclient import TestClient

from prophet_agent.api import titles
from prophet_agent.api.app import create_app
from prophet_agent.api.panels import MAX_PAGE, _best, _models, entity_page, panel_summary
from prophet_agent.api.service import MissionService
from prophet_agent.tools import default_tools

NAN = float("nan")


@pytest.fixture(autouse=True)
def no_title_model(monkeypatch):
    """Creating a mission names it in the background; keep that off the network."""
    monkeypatch.setattr(titles, "name_in_background", lambda *a, **k: None)


def _result(per_entity, **extra):
    return {"per_entity": per_entity, "horizon": 4, "origins": 3, "season": 7, "overall": {"timesfm": 0.9},
            "wins": {"timesfm": 2}, "best_overall": "timesfm", "coverage_80": {"timesfm": 0.8}, **extra}


ROWS = {
    "stn-03": {"naive": 1.4, "timesfm": 0.9, "blend": 1.0},
    "stn-01": {"naive": 0.6, "timesfm": 0.8, "blend": NAN},
    "stn-02": {"naive": NAN, "timesfm": None, "blend": NAN},     # no finite value at all
    "stn-04": {"naive": 2.0, "timesfm": 0.5, "blend": 0.7},
}


def test_best_is_the_lowest_finite_mase_and_ignores_bools_and_nan():
    assert _best({"naive": NAN, "timesfm": True, "blend": 0.5, "seasonal_naive": math.inf}) == "blend"
    assert _best({"naive": NAN}) is None
    assert _best({}) is None


def test_models_keep_the_reading_order_and_put_unknown_models_last_alphabetically():
    result = _result({"a": {"zeta": 1.0, "naive": 1.0, "custom_b": 1.0}, "b": {"timesfm": 1.0, "custom_a": 1.0}})
    assert _models(result) == ["naive", "timesfm", "custom_a", "custom_b", "zeta"]


def test_summary_passes_the_run_through_and_defaults_missing_parts():
    s = panel_summary(7, _result(ROWS))
    assert s["step"] == 7 and s["entities"] == 4
    assert s["models"] == ["naive", "timesfm", "blend"]
    assert s["horizon"] == 4 and s["best_overall"] == "timesfm"
    bare = panel_summary(2, {})
    assert bare == {"step": 2, "entities": 0, "models": [], "horizon": None, "origins": None, "season": None,
                    "overall": {}, "wins": {}, "best_overall": None, "coverage_80": {}, "reading": None, "basis": None}


def test_entity_rows_carry_the_best_model_and_every_model_column():
    page = entity_page(_result(ROWS), sort="entity")
    by_name = {r["entity"]: r for r in page["rows"]}
    assert by_name["stn-04"]["best_model"] == "timesfm"
    assert by_name["stn-01"]["best_model"] == "naive"       # 0.6 beats timesfm's 0.8; blend is NaN and does not count
    assert by_name["stn-02"]["best_model"] is None
    assert set(by_name["stn-02"]["mase"]) == {"naive", "timesfm", "blend"}
    assert math.isnan(by_name["stn-01"]["mase"]["blend"])


def test_entity_sort_ascending_and_descending():
    asc = [r["entity"] for r in entity_page(_result(ROWS), sort="entity")["rows"]]
    desc = [r["entity"] for r in entity_page(_result(ROWS), sort="entity", order="desc")["rows"]]
    assert asc == ["stn-01", "stn-02", "stn-03", "stn-04"]
    assert desc == list(reversed(asc))


def test_mase_sort_puts_the_lowest_first_and_missing_values_last_in_both_orders():
    asc = [r["entity"] for r in entity_page(_result(ROWS), sort="mase:timesfm")["rows"]]
    desc = [r["entity"] for r in entity_page(_result(ROWS), sort="mase:timesfm", order="desc")["rows"]]
    assert asc == ["stn-04", "stn-01", "stn-03", "stn-02"]       # stn-02's timesfm is None: last
    assert desc == ["stn-03", "stn-01", "stn-04", "stn-02"]      # highest first, missing still last


def test_best_sort_groups_by_best_model_and_puts_entities_without_one_last():
    names = [r["entity"] for r in entity_page(_result(ROWS), sort="best")["rows"]]
    assert names == ["stn-01", "stn-03", "stn-04", "stn-02"]       # naive, then timesfm (two), then no best model


def test_paging_splits_the_rows_and_reports_the_page_count():
    result = _result({f"e{i:02d}": {"naive": float(i)} for i in range(5)})
    first = entity_page(result, page=1, page_size=2)
    last = entity_page(result, page=3, page_size=2)
    assert first["total"] == 5 and first["pages"] == 3 and len(first["rows"]) == 2
    assert [r["entity"] for r in last["rows"]] == ["e04"]
    beyond = entity_page(result, page=9, page_size=2)
    assert beyond["rows"] == [] and beyond["total"] == 5


def test_an_empty_panel_still_has_one_page():
    page = entity_page({}, page=1, page_size=10)
    assert page == {"total": 0, "page": 1, "page_size": 10, "pages": 1, "sort": "entity", "order": "asc",
                    "models": [], "rows": []}


@pytest.mark.parametrize("kwargs", [
    {"page": 0}, {"page": -1}, {"page_size": 0}, {"page_size": MAX_PAGE + 1},
    {"order": "sideways"}, {"sort": "random"}, {"sort": "mase:unknown_model"}, {"sort": "mase:"},
])
def test_bad_paging_or_sort_input_is_a_value_error(kwargs):
    with pytest.raises(ValueError):
        entity_page(_result(ROWS), **kwargs)


# -- routes ---------------------------------------------------------------------------------------------------------

def _client(tmp_path):
    service = MissionService(tmp_path / "api.sqlite", lambda: None, default_tools())
    return TestClient(create_app(service), base_url="http://127.0.0.1:8000"), service


def _store_panel(service, mission_id, step, result):
    import json
    import time
    with service.store._tx() as c:
        c.execute("INSERT INTO tool_calls (idem_key, mission_id, step_idx, tool, args_json, status, result_json, error, updated) "
                  "VALUES (?,?,?,?,?,?,?,?,?)",
                  (f"{mission_id}-{step}", mission_id, step, "panel_backtest", "{}", "done", json.dumps(result), None, time.time()))


def test_the_panel_routes_list_and_page_a_stored_run(tmp_path):
    client, service = _client(tmp_path)
    mid = service.create("compare the station forecasts")
    _store_panel(service, mid, 3, _result(ROWS))
    listed = client.get(f"/api/missions/{mid}/panels").json()
    assert [p["step"] for p in listed] == [3] and listed[0]["entities"] == 4
    page = client.get(f"/api/missions/{mid}/panels/3/entities?page=1&page_size=2&sort=mase:timesfm").json()
    assert page["total"] == 4 and [r["entity"] for r in page["rows"]] == ["stn-04", "stn-01"]


def test_the_panel_entities_route_refuses_bad_input_and_unknown_runs(tmp_path):
    client, service = _client(tmp_path)
    mid = service.create("compare the station forecasts")
    _store_panel(service, mid, 3, _result(ROWS))
    assert client.get(f"/api/missions/{mid}/panels/3/entities?page_size=500").status_code == 422
    assert client.get(f"/api/missions/{mid}/panels/3/entities?sort=bogus").status_code == 422
    assert client.get(f"/api/missions/{mid}/panels/8/entities").status_code == 404
    assert client.get("/api/missions/no-such-mission/panels").status_code == 404
    assert client.get("/api/missions/no-such-mission/panels/3/entities").status_code == 404
