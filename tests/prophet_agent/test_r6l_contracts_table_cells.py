"""Round 6l (2026-11-23), audit item: a wrong figure in a table cell under a numeric header must not escape the check.

Before the fix, a day cell with two or four figures ("52 (12)", "52 (12 to 132, 90%)") or a spread ("52 ± 5") was skipped
outright, so a wrong point passed. Each such cell now has its leading figure, the point, checked against the station's own
forecast. Every series here is synthetic.
"""
from prophet_agent.runtime import contracts


def _history():
    series = {
        "Aotizhongxin": {"point": [40.0, 75.0], "quantiles": {"0.1": [4.5, 24.3], "0.9": [114.2, 157.0]}},
        "Dingling": {"point": [10.0, 12.0], "quantiles": {"0.1": [1.0, 2.0], "0.9": [20.0, 25.0]}},
    }
    full = {"n_series": 2, "forecast": {"series": series}}
    return [{"kind": "tool", "tool": "forecast_report", "args": {}, "result": {"n_series": 2, "series": {}}, "full": full}]


def _table(day1_cell, header="Day 1"):
    return f"| Station | {header} |\n|---|---|\n| Aotizhongxin | {day1_cell} |\n| Dingling | 10 |\n"


def _check(text):
    return contracts._table_matches_panel(text, _history())


def test_a_wrong_point_with_a_two_figure_cell_is_blocked():
    c = _check(_table("52 (12)"))
    assert c["ok"] is False and c["severity"] == "block"
    assert "Aotizhongxin day 1 is 52 (12)" in c["detail"]


def test_a_wrong_point_with_a_four_figure_cell_is_blocked():
    c = _check(_table("52 (12 to 132, 90%)"))
    assert c["ok"] is False and "Aotizhongxin day 1" in c["detail"]


def test_a_wrong_point_with_a_plus_minus_spread_is_blocked():
    c = _check(_table("52 ± 5"))
    assert c["ok"] is False and "Aotizhongxin day 1" in c["detail"]


def test_a_right_point_with_extra_figures_in_the_cell_still_passes():
    assert _check(_table("40 (rank 2)"))["ok"] is True
    assert _check(_table("40 kWh (+5% on yesterday)"))["ok"] is True


def test_a_correct_point_with_its_band_still_passes_and_a_wrong_band_is_still_blocked():
    assert _check(_table("40 (4.5 to 114.2)"))["ok"] is True
    assert _check(_table("40 (9 to 114.2)"))["ok"] is False


def test_a_wrong_average_with_a_spread_is_blocked():
    text = "| Station | 14-day average |\n|---|---|\n| Aotizhongxin | 77.1 ± 3 |\n"
    c = _check(text)
    assert c["ok"] is False and "Aotizhongxin average is 77.1 ± 3" in c["detail"]


def test_a_right_average_with_a_spread_passes():
    text = "| Station | 14-day average |\n|---|---|\n| Aotizhongxin | 57.5 ± 3 |\n"
    assert _check(text)["ok"] is True
