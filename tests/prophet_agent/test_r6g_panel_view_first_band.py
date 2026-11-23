"""Round 6g (2026-11-23): the panel view gives each station's first-step band, not only its last-step band.

Live Beijing d1a1d077 (build 785a42f): the view had point_step_1 but no day-1 band, so the answer wrote day-1 bands that were
not the stored ones (Aotizhongxin 8.8-131.2 against the stored 4.5-114.2) and the figure check blocked it. Every series here
is synthetic.
"""
from __future__ import annotations

from prophet_agent.tools import report_tool


def _fc(point, q10, q90):
    return {"point": list(point), "quantiles": {"0.1": list(q10), "0.9": list(q90)}, "health": {"verdict": "ok"}}


def test_the_panel_view_carries_the_first_and_last_step_bands_of_each_station():
    fcs = {"Aotizhongxin": _fc([40.7, 50.0, 75.2], [4.5, 20.0, 24.3], [114.2, 120.0, 157.0])}
    result = {"n_series": 1, "run_id": "r", "files": [], "manifest": "m", "forecast": {"series": fcs}}
    line = report_tool._panel_observe(result)["series"]["Aotizhongxin"]
    assert line["point_step_1"] == 40.7
    assert line["q10_step_1"] == 4.5 and line["q90_step_1"] == 114.2
    assert line["q10_last_step"] == 24.3 and line["q90_last_step"] == 157.0
