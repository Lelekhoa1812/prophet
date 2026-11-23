"""Round 6h (2026-11-23): a station table's day figures match that station's own forecast.

Live Beijing 0137871e (build cf1f0cc): the table gave Aotizhongxin day 1 as 52 (12 to 132); the stored forecast gives 40.7
(4.5 to 114.2), and the figure check passed it because 52 appears elsewhere in the evidence. Every series here is synthetic.
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


def _check(text, history=None):
    return contracts._table_matches_panel(text, _history() if history is None else history)


def test_a_table_whose_day_cells_match_each_stations_own_forecast_passes():
    text = ("| Station | Day 1 | Day 2 |\n|---|---|---|\n"
            "| **Aotizhongxin** | 40 (5 to 114) | 75 (24 to 157) |\n"
            "| Dingling | 10 (1 to 20) | 12 (2 to 25) |\n")
    c = _check(text)
    assert c["ok"] is True and c["name"] == "answer_table_matches_forecast"


def test_a_day_cell_copied_from_another_station_is_blocked_and_named():
    text = ("| Station | Day 1 |\n|---|---|\n"
            "| Aotizhongxin | 52 (12 to 132) |\n| Dingling | 10 (1 to 20) |\n")
    c = _check(text)
    assert c["ok"] is False and c["severity"] == "block"
    assert "Aotizhongxin day 1 is 52" in c["detail"] and "40.0" in c["detail"]


def test_the_average_column_is_checked_against_the_station_mean():
    good = "| Station | 14-day average |\n|---|---|\n| Aotizhongxin | 57.5 |\n"
    bad = "| Station | 14-day average |\n|---|---|\n| Aotizhongxin | 77.1 |\n"
    assert _check(good)["ok"] is True
    assert _check(bad)["ok"] is False


def test_a_range_header_with_an_arrow_is_not_read_as_a_day_column():
    text = "| Station | 10-90% range (Day 1 → Day 14) |\n|---|---|\n| Aotizhongxin | 8.8-131.2 → 17.1-186.3 |\n"
    assert _check(text) is None


def test_a_table_with_no_station_rows_is_not_checked():
    assert _check("| Day | Value |\n|---|---|\n| 1 | 40 |\n") is None


def test_no_panel_forecast_means_no_check():
    assert _check("| Station | Day 1 |\n|---|---|\n| Aotizhongxin | 52 |\n", history=[]) is None


def test_the_panel_view_gives_every_day_of_each_station_in_printed_form():
    from prophet_agent.tools import report_tool
    fcs = {"Aotizhongxin": {"point": [40.0, 75.0], "quantiles": {"0.1": [4.5, 24.3], "0.9": [114.2, 157.0]}, "health": {"verdict": "ok"}}}
    result = {"n_series": 1, "run_id": "r", "files": [], "manifest": "m", "forecast": {"series": fcs}}
    line = report_tool._panel_observe(result)["series"]["Aotizhongxin"]
    assert line["by_day"] == ["40.0 (4.5 to 114.2)", "75.0 (24.3 to 157.0)"]


def test_a_table_copied_from_the_day_by_day_view_passes_the_check():
    from prophet_agent.tools import report_tool
    fcs = {"Aotizhongxin": {"point": [40.0, 75.0], "quantiles": {"0.1": [4.5, 24.3], "0.9": [114.2, 157.0]}, "health": {"verdict": "ok"}}}
    view = report_tool._panel_observe({"n_series": 1, "run_id": "r", "files": [], "manifest": "m", "forecast": {"series": fcs}})
    by_day = view["series"]["Aotizhongxin"]["by_day"]
    text = "| Station | Day 1 | Day 2 |\n|---|---|---|\n| Aotizhongxin | " + " | ".join(by_day) + " |\n"
    history = [{"kind": "tool", "tool": "forecast_report", "args": {}, "result": {"n_series": 1, "series": {}},
                "full": {"n_series": 1, "forecast": {"series": fcs}}}]
    assert contracts._table_matches_panel(text, history)["ok"] is True
