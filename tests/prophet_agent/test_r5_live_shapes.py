"""Round 5, rep6: two fixes that tests had passed on the wrong shape, or that a refusal blocked.

1. data_entity_count read the full inspect_data profile, but the history holds the planner's compact view (columns as lines
   such as "entity: text, unique=60", layout hints at the top), so the coverage note never fired live: Beijing backtested 3
   of 12 stations and electricity 15 of 60, and both stalled. This test uses the compact view as stored (electricity rep6).
2. Carrying several analysts' out/scan.json failed with "two carried files are named 'scan.json'" five times (Beijing rep6).
"""
from __future__ import annotations

import pytest

from prophet_agent.runtime import contracts
from prophet_agent.runtime.goal import _coverage_note
from prophet_agent.tools import sandbox_tool

COMPACT = {"kind": "tool", "tool": "inspect_data", "args": {"file": "electricity_daily_60_clients.csv"}, "result": {
    "columns": ["entity: text, unique=60; e.g. ['MT_001', 'MT_001']",
                "timestamp: time, first=2011-01-01 00:00:00, last=2015-01-01 00:00:00, typical_step=1 days 00:00:00, unique=1462",
                "kwh_daily_sum: number, max=163704.56, unique=56535"],
    "file": "electricity_daily_60_clients.csv", "issues": [],
    "layout_hints": ["long: a time column, grouping column(s) ['entity'] and numeric column(s); each group may be its own series"],
    "rows": 87720}}


def test_the_compact_inspect_view_gives_the_entity_count():
    assert contracts.data_entity_count([COMPACT]) == (60, "entity")


def test_a_fifteen_client_panel_backtest_gets_the_note_from_the_compact_view():
    pb = {"kind": "tool", "tool": "panel_backtest", "args": {}, "result": {"entities": 15, "best_overall": "timesfm"}}
    note = _coverage_note(pb, [COMPACT])
    assert note and "15 of the 60" in note


@pytest.fixture()
def runs(tmp_path, monkeypatch):
    monkeypatch.setattr(sandbox_tool, "SANDBOX_ROOT", tmp_path)
    ids = ["1" * 32, "2" * 32]
    for rid in ids:
        (tmp_path / rid / "out").mkdir(parents=True)
        (tmp_path / rid / "out" / "scan.json").write_text("{}")
    return ids


def test_two_files_with_the_same_name_are_both_mounted(runs):
    m = sandbox_tool.carried_inputs([{"run_id": r, "file": "out/scan.json"} for r in runs])
    assert set(m) == {"carried/scan.json", "carried/22222222_scan.json"}
