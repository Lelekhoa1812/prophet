"""Round 3 gate B4, goal 5 (A3-12): a panel goal has a checklist item that every entity has a delivered forecast.

Live evidence (measured): Beijing, 12 stations in the panel, the forecast held 10, and the answer said "for all 12".
"""
from prophet_agent.runtime import contracts


def _backtest_panel(names, n=None):
    return {"kind": "tool", "tool": "backtest_series", "args": {"horizon": 14},
            "result": {"ok": True, "n_series": n or len(names), "horizon": 14,
                       "series": {k: {"models": {}} for k in names}}}


def _panel_pack(names, n, not_shown=0):
    return {"kind": "tool", "tool": "forecast_report", "args": {"horizon": 14},
            "result": {"ok": True, "n_series": n, "run_id": "c" * 32, "files": [],
                       "series": {k: {"health": "ok", "centre": "timesfm"} for k in names},
                       **({"not_shown": not_shown} if not_shown else {})}}


def _entity_item(history):
    items = contracts.progress(history, run_root=None)
    return next((i for i in items if i["item"].startswith("every panel entity")), None)


def test_a_panel_with_two_of_three_forecasts_names_the_missing_entity():
    # Before the fix there was no such item at all, so the only signal was the free-text answer.
    hist = [_backtest_panel(["Tiantan", "Wanliu", "Wanshouxigong"]), _panel_pack(["Tiantan", "Wanliu"], n=3)]
    item = _entity_item(hist)
    assert item is not None and item["status"] == "todo"
    assert "Wanshouxigong" in item["detail"] and "2 of 3" in item["detail"]


def test_a_panel_with_every_entity_forecast_ticks_the_item():
    hist = [_backtest_panel(["A", "B", "C"]), _panel_pack(["A", "B", "C"], n=3)]
    assert _entity_item(hist)["status"] == "done"


def test_a_panel_pack_that_shows_only_some_lines_counts_the_rest_as_delivered():
    # A pack of more than the shown lines reports not_shown; its names are unknown, so the count decides.
    hist = [{"kind": "tool", "tool": "backtest_series", "args": {}, "result": {"ok": True, "n_series": 40, "horizon": 7}},
            _panel_pack([f"s{i}" for i in range(30)], n=40, not_shown=10)]
    assert _entity_item(hist)["status"] == "done"


def test_a_single_series_mission_has_no_entity_item():
    single = {"kind": "tool", "tool": "backtest_series", "args": {}, "result": {"ok": True, "horizon": 7, "models": {}}}
    assert _entity_item([single]) is None
    assert _entity_item([]) is None


def test_a_panel_with_no_forecast_yet_is_todo_with_the_count():
    item = _entity_item([_backtest_panel(["A", "B", "C", "D"])])
    assert item["status"] == "todo" and "0 of 4" in item["detail"]
