"""Round 5: a panel result that covers fewer entities than the uploaded data holds is named as such.

Live Beijing rep4 (measured, 2026-11-23): the data held 12 stations (inspect_data: grouping column 'entity', 12 unique). The
lead fed one analyst's 3-station file to every backtest and to the pack; the panel check read its expected count from those
same results (3), so the answer covered 3 of 12 and the checklist looked complete.
"""
from __future__ import annotations

from prophet_agent.runtime import contracts
from prophet_agent.runtime.goal import _coverage_note


def _inspect(unique=12, hints=("long: a time column, grouping column(s) ['entity'] and numeric column(s); each group may be its own series",)):
    return {"kind": "tool", "tool": "inspect_data", "args": {"file": "b.csv"},
            "result": {"file": "b.csv", "profile": {"columns": [{"name": "entity", "kind": "text", "unique": unique},
                                                                {"name": "pm25", "kind": "number", "unique": 900}],
                                                    "layout_hints": list(hints)}}}


def _pack(names):
    return {"kind": "tool", "tool": "forecast_report", "args": {},
            "result": {"n_series": len(names), "series": {n: {"health": "ok"} for n in names}}}


def test_the_data_entity_count_comes_from_the_one_grouping_column():
    assert contracts.data_entity_count([_inspect()]) == (12, "entity")
    assert contracts.data_entity_count([_inspect(hints=("wide: one column per series",))]) is None
    two = "long: a time column, grouping column(s) ['store', 'sku'] and numeric column(s)"
    assert contracts.data_entity_count([_inspect(hints=(two,))]) is None


def test_panel_entities_expects_what_the_data_holds():
    p = contracts.panel_entities([_inspect(), _pack(["A", "B", "C"])])
    assert p["expected"] == 12 and p["delivered"] == 3


def test_a_full_panel_is_complete():
    p = contracts.panel_entities([_inspect(), _pack([f"S{i}" for i in range(12)])])
    assert p["expected"] == 12 and p["delivered"] == 12


def test_a_short_result_gets_a_coverage_note_and_a_full_one_does_not():
    hist = [_inspect()]
    note = _coverage_note(_pack(["A", "B", "C"]), hist)
    assert note and "3 of the 12" in note and "pl.save_panel" in note
    assert _coverage_note(_pack([f"S{i}" for i in range(12)]), hist) is None
    assert _coverage_note({"kind": "tool", "tool": "panel_backtest", "result": {"entities": 4}}, hist)
    assert _coverage_note({"kind": "tool", "tool": "sandbox_python", "result": {"n_series": 1}}, hist) is None
    assert _coverage_note(_pack(["A"]), []) is None            # no inspection: nothing to compare with


def test_a_repeat_of_a_short_call_carries_its_coverage_note():
    from prophet_agent.runtime.goal import _repeat_obs
    earlier = {**_pack(["A", "B", "C"]), "step": 2, "coverage_note": "this result covers 3 of the 12 values"}
    obs = _repeat_obs("forecast_report", {}, [_inspect(), earlier], 2)
    assert obs["coverage_note"] == earlier["coverage_note"] and "3 of the 12" in obs["note"]


def test_the_coverage_note_gives_the_cell_that_saves_every_entity():
    # live electricity rep12 (measured, 2026-11-23): the note said "pl.panel(df, entity='entity', ...)" and the lead re-ran a
    # cell listing the entities three times, then stalled with no forecast
    insp = {"kind": "tool", "tool": "inspect_data", "args": {"file": "electricity_daily_60_clients.csv"}, "result": {
        "columns": ["entity: text, unique=60; e.g. ['MT_001']", "timestamp: time, first=2011-01-01, unique=1462",
                    "kwh_daily_sum: number, max=163704.56, unique=56535"],
        "layout_hints": ["long: a time column, grouping column(s) ['entity'] and numeric column(s); each group may be its own series"]}}
    short = {"kind": "tool", "tool": "panel_backtest", "args": {}, "result": {"entities": 15, "overall": {}}}
    note = _coverage_note(short, [insp, short])
    assert ("pl.panel(df, entity='entity', time='timestamp', value='kwh_daily_sum')" in note
            and "in/electricity_daily_60_clients.csv" in note and "pl.save_panel(series)" in note)
    two_values = dict(insp, result=dict(insp["result"], columns=insp["result"]["columns"] + ["price: number, unique=9"]))
    assert "value='<value column>'" in contracts.panel_build_cell([two_values])
