"""Round 6m, goal audit item 3(a): a station row whose name carries a suffix is checked against its own series.

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


# (a) A station row whose name carries a suffix is not matched to its series, so its wrong figure passed unchecked.
def test_a_wrong_figure_in_a_station_row_with_a_suffix_is_blocked():
    text = "| Station | Day 1 |\n|---|---|\n| Aotizhongxin (north) | 52.0 (12.0 to 132.0) |\n"
    check = _table_check(text)
    assert check is not None and check["ok"] is False


def test_a_station_row_with_a_suffix_after_a_dash_is_matched_to_its_series():
    text = "| Station | Day 1 |\n|---|---|\n| Aotizhongxin - north | 52.0 (12.0 to 132.0) |\n"
    check = _table_check(text)
    assert check is not None and check["ok"] is False


def test_a_suffix_does_not_match_a_different_station_that_starts_with_the_same_name():
    # "Dingling" must not stand in for a longer, unrelated name that begins with it.
    text = "| Station | Day 1 |\n|---|---|\n| Dingling Annex | 52.0 (12.0 to 132.0) |\n"
    assert _table_check(text) is None
