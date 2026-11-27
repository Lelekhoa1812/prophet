"""A prediction is charted, and a flat centre line is held until several ways of finding its shape were tried (owner, 2026-11-27)."""
from __future__ import annotations

from prophet_agent.runtime import prediction_quality as pq


def pack(point, files=("out/forecast.png (1 KB)",), centre="timesfm"):
    q10 = [v - 5 for v in point]
    q90 = [v + 5 for v in point]
    return {"kind": "tool", "tool": "forecast_report", "args": {"centre": centre},
            "result": {"ok": True, "files": list(files), "forecast": {"point": point}},
            "full": {"forecast": {"point": point, "quantiles": {"0.1": q10, "0.9": q90}}}}


def call(tool, args=None, **result):
    return {"kind": "tool", "tool": tool, "args": args or {}, "result": {"ok": True, **result}}


FLAT = [10.0] * 12
MOVING = [10.0 + 2 * i for i in range(12)]


def test_forecast_with_a_chart_passes():
    c = pq.check_prediction_chart([pack(MOVING)])
    assert c["ok"]


def test_numbers_only_forecast_is_blocked_with_how_to_chart():
    c = pq.check_prediction_chart([call("timesfm_forecast", point=MOVING)])
    assert not c["ok"] and c["severity"] == "block" and "forecast_report" in c["detail"]


def test_league_simulation_without_a_picture_asks_for_a_club_compared_chart():
    c = pq.check_prediction_chart([call("simulate_season", teams=[{"team": "A"}])])
    assert not c["ok"] and "club-compared" in c["detail"] and "finishing place" in c["detail"]


def test_league_chart_written_after_the_simulation_passes():
    hist = [call("simulate_season"), call("sandbox_python", outputs=["out/clubs.png (20 KB)"])]
    assert pq.check_prediction_chart(hist)["ok"]


def test_a_chart_made_before_the_prediction_does_not_count():
    hist = [call("sandbox_python", outputs=["out/eda.png"]), call("simulate_season")]
    assert not pq.check_prediction_chart(hist)["ok"]


def test_no_prediction_no_chart_check():
    assert pq.check_prediction_chart([call("sandbox_python", outputs=[])]) is None


def test_chart_made_by_a_sub_agent_counts():
    hist = [call("simulate_season"),
            {"kind": "subagent", "evidence": [call("sandbox_python", outputs=[{"path": "out/clubs.svg"}])]}]
    assert pq.check_prediction_chart(hist)["ok"]


def test_a_line_that_moves_is_not_held():
    assert pq.check_flat_centre([pack(MOVING)]) is None


def test_flat_line_is_blocked_and_names_the_untried_strategies():
    c = pq.check_flat_centre([pack(FLAT)])
    assert not c["ok"] and c["severity"] == "block"
    assert "0 of 4" in c["detail"] and "steps" in c["detail"] and "drivers" in c["detail"]


def test_flat_line_passes_after_four_strategies():
    hist = [call("sandbox_python", {"code": "level shift regime step path via hmm"}),
            call("sandbox_python", {"code": "stl seasonal decomposition"}),
            call("control_lab", {"with_covariates": True, "past_covariates": [[1, 2]]}),
            call("sandbox_python", {"code": "hampel outlier filter and kalman level"}),
            pack(FLAT)]
    tried = pq.strategies_tried(hist)
    assert {"steps", "seasonal", "drivers", "cleaning"} <= set(tried)
    c = pq.check_flat_centre(hist)
    assert c["ok"]


def test_three_strategies_is_not_enough():
    hist = [call("sandbox_python", {"code": "regime step path"}), call("sandbox_python", {"code": "stl"}),
            call("control_lab", {"with_covariates": True}), pack(FLAT)]
    assert not pq.check_flat_centre(hist)["ok"]


def test_prediction_checks_collects_both_and_never_raises():
    out = pq.prediction_checks([pack(FLAT, files=())])
    assert {c["name"] for c in out} == {"prediction_charted", "centre_not_flat"}
    assert pq.prediction_checks([None, "x", {"kind": "tool"}]) == []
