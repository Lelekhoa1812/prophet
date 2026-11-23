"""Round 6i (2026-11-23): a station table's wrong day cell is set to that station's own forecast, not left to a rewrite.

Live Beijing 80bbc853 (build a1c81fd): eight stations' day cells were not the stored values (Guanyuan day 2 was 124.0 (57.5 to
225.2); the view had 129.1 (59.2 to 226.5)). The one rewrite copied them wrong again and the answer left partial. The
correction is deterministic: the row names the station and the column names the day. Every series here is synthetic.
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


def _snap(text, history=None):
    return contracts.snap_tables(text, _history() if history is None else history)


def test_a_wrong_day_cell_is_set_to_the_stations_own_forecast_and_then_passes_the_check():
    text = ("| Station | Day 1 |\n|---|---|\n"
            "| Aotizhongxin | 52.0 (12.0 to 132.0) |\n| Dingling | 10.0 (1.0 to 20.0) |\n")
    new, fixed = _snap(text)
    assert "| Aotizhongxin | 40.0 (4.5 to 114.2) |" in new
    assert "| Dingling | 10.0 (1.0 to 20.0) |" in new
    assert fixed == [("52.0 (12.0 to 132.0)", "40.0 (4.5 to 114.2)")]
    assert contracts._table_matches_panel(new, _history())["ok"] is True


def test_a_table_that_already_matches_is_returned_unchanged():
    text = ("| Station | Day 1 | Day 2 |\n|---|---|---|\n"
            "| Aotizhongxin | 40.0 (4.5 to 114.2) | 75.0 (24.3 to 157.0) |\n")
    new, fixed = _snap(text)
    assert new == text and fixed == []


def test_bold_station_name_and_the_cell_punctuation_are_kept():
    text = "| Station | Day 1 |\n|---|---|\n| **Aotizhongxin** | 52.0 (12.0 to 132.0) |\n"
    new, _ = _snap(text)
    assert "| **Aotizhongxin** | 40.0 (4.5 to 114.2) |" in new


def test_a_point_alone_is_set_to_the_point_and_keeps_its_decimals():
    text = "| Station | Day 2 |\n|---|---|\n| Aotizhongxin | 80.0 |\n"
    new, fixed = _snap(text)
    assert "| Aotizhongxin | 75.0 |" in new and fixed == [("80.0", "75.0")]


def test_the_average_column_is_set_to_the_station_mean():
    text = "| Station | 14-day average |\n|---|---|\n| Aotizhongxin | 77.1 |\n"
    new, fixed = _snap(text)
    assert "| Aotizhongxin | 57.5 |" in new and fixed == [("77.1", "57.5")]


def test_a_band_header_column_is_never_rewritten():
    text = ("| Station | Day 1 | Day 1 q10 |\n|---|---|---|\n"
            "| Aotizhongxin | 40.0 (4.5 to 114.2) | 99.0 |\n")
    new, fixed = _snap(text)
    assert "| 99.0 |" in new and fixed == []


def test_a_range_that_is_not_a_single_station_cell_is_left_alone():
    text = "| Station | Day 1 |\n|---|---|\n| Aotizhongxin | 8.8-131.2 |\n"
    new, fixed = _snap(text)
    assert new == text and fixed == []


def test_a_thousands_comma_cell_is_left_alone():
    text = "| Station | Day 1 |\n|---|---|\n| Aotizhongxin | 1,052.0 (12.0 to 132.0) |\n"
    new, fixed = _snap(text)
    assert new == text and fixed == []


def test_prose_outside_the_table_is_untouched():
    text = ("Aotizhongxin day 1 is 52.0 (12.0 to 132.0).\n\n"
            "| Station | Day 1 |\n|---|---|\n| Aotizhongxin | 52.0 (12.0 to 132.0) |\n")
    new, _ = _snap(text)
    assert new.startswith("Aotizhongxin day 1 is 52.0 (12.0 to 132.0).\n\n")


def test_a_row_that_names_no_station_is_untouched():
    text = "| Station | Day 1 |\n|---|---|\n| Total | 52.0 (12.0 to 132.0) |\n"
    new, fixed = _snap(text)
    assert new == text and fixed == []


def test_no_panel_forecast_means_no_change():
    text = "| Station | Day 1 |\n|---|---|\n| Aotizhongxin | 52.0 (12.0 to 132.0) |\n"
    assert _snap(text, history=[]) == (text, [])
