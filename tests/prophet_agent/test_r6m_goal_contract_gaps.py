"""Round 6m, goal audit item 3: contract gaps where a wrong figure or an unverified claim passed without a check.

Each test reproduces one gap against the stored contract behaviour. Every series and backtest here is synthetic.
"""
from __future__ import annotations

from prophet_agent.runtime import contracts


def _history():
    series = {
        "Aotizhongxin": {"point": [40.0, 75.0], "quantiles": {"0.1": [4.5, 24.3], "0.9": [114.2, 157.0]}},
        "Dingling": {"point": [10.0, 12.0], "quantiles": {"0.1": [1.0, 2.0], "0.9": [20.0, 25.0]}},
    }
    full = {"n_series": 2, "forecast": {"series": series}}
    return [{"kind": "tool", "tool": "forecast_report", "args": {}, "result": {"n_series": 2, "series": {}}, "full": full}]


def _table_check(text):
    return contracts._table_matches_panel(text, _history())


# (b) A table whose rows have no outer pipes is not read at all, so a wrong figure in it passed.
def test_a_wrong_figure_in_a_table_without_outer_pipes_is_blocked():
    text = "Station | Day 1\n--- | ---\nAotizhongxin | 52.0 (12.0 to 132.0)\n"
    check = _table_check(text)
    assert check is not None and check["ok"] is False


def test_a_prose_line_with_a_pipe_and_no_separator_is_not_read_as_a_table():
    text = "Choose A | B for the day.\nAotizhongxin | 52.0 is the figure we used."
    assert _table_check(text) is None


def test_snap_corrects_a_wrong_cell_in_a_table_without_outer_pipes_and_keeps_the_shape():
    text = "Station | Day 1\n--- | ---\nAotizhongxin | 52.0 (12.0 to 132.0)\n"
    new, fixed = contracts.snap_tables(text, _history())
    assert "Aotizhongxin | 40.0 (4.5 to 114.2)" in new and "|" not in new.splitlines()[2][:1]
    assert fixed == [("52.0 (12.0 to 132.0)", "40.0 (4.5 to 114.2)")]
