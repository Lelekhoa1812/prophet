"""Round 6m, goal audit item 3(c): a "Total" column in a station table is checked against the forecast's horizon total.

Synthetic series only.
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


# (c) A "Total" column is a numeric column with no day and no average name, so a wrong total passed unchecked.
def test_a_wrong_total_column_is_blocked_against_the_sum_of_the_forecast():
    # Dingling's point over the horizon is 10 + 12 = 22; the day cell is right, the total is not.
    text = "| Station | Day 1 | Total |\n|---|---|---|\n| Dingling | 10.0 (1.0 to 20.0) | 99.0 |\n"
    check = _table_check(text)
    assert check is not None and check["ok"] is False


def test_a_correct_total_column_passes():
    text = "| Station | Day 1 | Total |\n|---|---|---|\n| Dingling | 10.0 (1.0 to 20.0) | 22.0 |\n"
    check = _table_check(text)
    assert check is not None and check["ok"] is True


def test_snap_sets_a_wrong_total_cell_to_the_forecast_sum_and_then_passes():
    text = "| Station | Day 1 | Total |\n|---|---|---|\n| Dingling | 10.0 (1.0 to 20.0) | 99.0 |\n"
    new, fixed = contracts.snap_tables(text, _history())
    assert "| Dingling | 10.0 (1.0 to 20.0) | 22.0 |" in new
    assert fixed == [("99.0", "22.0")]
    assert _table_check(new)["ok"] is True
