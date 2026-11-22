"""Round 5: four defects from live electricity rep5 (measured, 2026-11-23, 60 clients, 100 tool calls, stalled).

1. source.name as a list of ten clients failed with "unhashable type: 'list'", three times.
2. The pack was built for MT_001 alone and its total was answered as the 60-client total; no coverage note fired.
3. The stall rescue gave "compare with the simple baselines (backtest_series)" to an analyst, which has no backtest tool.
4. With only panel_backtest run, the baseline checklist item stayed todo; the same panel_backtest was sent twice more.
"""
from __future__ import annotations

import json

import pytest

from prophet_agent.runtime import agents, contracts
from prophet_agent.runtime.goal import _agent_for, _coverage_note
from prophet_agent.tools import series_io

RID = "c" * 32


@pytest.fixture()
def panel_file(tmp_path, monkeypatch):
    monkeypatch.setattr(series_io, "SANDBOX_ROOT", tmp_path)
    (tmp_path / RID / "out").mkdir(parents=True)
    (tmp_path / RID / "out" / "series.json").write_text(json.dumps({"series": {f"MT_{i:03d}": [1.0, 2.0, 3.0] for i in range(1, 6)}}))
    return tmp_path


def test_a_list_of_names_selects_that_subset(panel_file):
    got = series_io.resolve_series({"source": {"run_id": RID, "name": ["MT_002", "MT_004", "MT_002"]}})
    assert [n for n, _ in got] == ["MT_002", "MT_004"]
    with pytest.raises(ValueError, match="MT_009"):
        series_io.resolve_series({"source": {"run_id": RID, "name": ["MT_001", "MT_009"]}})


def _inspect(n=60):
    return {"kind": "tool", "tool": "inspect_data", "result": {"profile": {
        "columns": [{"name": "client", "unique": n}],
        "layout_hints": ["long: a time column, grouping column(s) ['client'] and numeric column(s)"]}}}


def test_a_pack_for_one_named_entity_of_a_panel_gets_a_coverage_note():
    obs = {"kind": "tool", "tool": "forecast_report", "args": {"source": {"run_id": RID, "name": "MT_001"}},
           "result": {"forecast": {"point": [1.0]}, "centre_line_total": 1389.72}}
    note = _coverage_note(obs, [_inspect()])
    assert note and "1 of the 60" in note
    whole = {**obs, "args": {"source": {"run_id": RID}}}         # one aggregated series: not named, so no note
    assert _coverage_note(whole, [_inspect()]) is None


def test_model_work_goes_to_a_modeler_that_can_backtest_a_panel():
    assert _agent_for("compare the forecast method with the simple baselines (backtest_series)") == "modeler"
    assert _agent_for("analysis ledger: at least 4 stages") == "analyst"
    assert "panel_backtest" in agents.TYPES["modeler"].tools


def test_a_panel_backtest_ticks_the_baseline_item_and_the_gate_for_a_panel_pack():
    pb = {"kind": "tool", "tool": "panel_backtest", "args": {}, "result": {"best_overall": "blend", "entities": 60}}
    insp = {"kind": "tool", "tool": "inspect_data", "args": {}, "result": {"profile": {}}}
    pack = {"kind": "tool", "tool": "forecast_report", "args": {},
            "result": {"n_series": 60, "series": {f"MT_{i}": {"health": "ok"} for i in range(60)}}}
    items = contracts.progress([insp, pb, pack])
    assert any("simple baselines" in it["item"] and it["status"] == "done" for it in items)
    gate = [c for c in contracts.check_evidence([insp, pb, pack]) if c["name"] == "baseline_compared"]
    assert gate and gate[0]["ok"]
