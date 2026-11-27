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
    hist = [call("sandbox_python", {"code": "path = step_path(meeting_dates, moves)"}),
            call("sandbox_python", {"code": "res = STL(y, period=12).fit()"}),
            call("control_lab", {"with_covariates": True, "past_covariates": [[1, 2]]}),
            call("sandbox_python", {"code": "y = hampel(y); level = UnobservedComponents(y, 'local level').fit()"}),
            pack(FLAT)]
    tried = pq.strategies_tried(hist)
    assert {"steps", "seasonal", "drivers", "cleaning"} <= set(tried)
    c = pq.check_flat_centre(hist)
    assert c["ok"]


def test_three_strategies_is_not_enough():
    hist = [call("sandbox_python", {"code": "p = step_path(d)"}), call("sandbox_python", {"code": "STL(y).fit()"}),
            call("control_lab", {"with_covariates": True, "past_covariates": [[1.0]]}), pack(FLAT)]
    assert not pq.check_flat_centre(hist)["ok"]


def test_prediction_checks_collects_both_and_never_raises():
    out = pq.prediction_checks([pack(FLAT, files=())])
    assert {c["name"] for c in out} == {"prediction_charted", "centre_not_flat", "signals_gathered"}
    assert pq.prediction_checks([None, "x", {"kind": "tool"}]) == []


def test_signals_gathered_warns_without_blocking_and_clears_with_a_driver_test():
    warn = pq.check_signals_gathered([pack(MOVING)])
    assert not warn["ok"] and warn["severity"] == "warn"
    with_driver = [call("control_lab", {"with_covariates": True, "past_covariates": [[1.0]]}), pack(MOVING)]
    assert pq.check_signals_gathered(with_driver) is None
    assert pq.check_signals_gathered([call("sandbox_python")]) is None


def test_words_in_a_scan_a_default_argument_or_a_data_search_are_not_strategies():
    # Round 22 (AURATE, AUUNEMP): "trend" and "Level shift" printed by a scan cell, backtest_series' default combine and
    # open-data searches passed every flat line.
    hist = [call("sandbox_python", {"code": "print('trend', 'Level shift', 'Outlier', 'Seasonal', 'covariate')"}),
            call("backtest_series", {"combine": True}), call("search_open_data", {"query": "open_data rates"}),
            {"kind": "tool", "tool": "web_research", "args": {}, "error": "timeout"}, pack(FLAT)]
    assert pq.strategies_tried(hist) == []
    assert not pq.check_flat_centre(hist)["ok"]


def test_a_daily_forecast_for_a_month_by_month_request_is_blocked():
    # Round 22 (SPX): "month by month over the next 2 years" delivered as 504 business days.
    daily = call("timesfm_forecast", point=[1.0] * 504, config={"horizon": 504, "series_step": 1.0})
    c = pq.check_grain("Forecast it month by month over the next 2 years", [daily])
    assert not c["ok"] and c["severity"] == "block" and "Aggregate" in c["detail"]
    monthly = call("timesfm_forecast", point=[1.0] * 24, config={"horizon": 24, "series_step": "monthly"})
    assert pq.check_grain("monthly unemployment, next 2 years", [monthly])["ok"]
    quarterly = call("timesfm_forecast", point=[1.0] * 8, config={"horizon": 8, "series_step": "quarterly"})
    assert pq.check_grain("monthly unemployment", [quarterly])["severity"] == "warn"
    assert pq.check_grain("forecast the next 2 years", [daily]) is None


def test_an_answer_range_that_is_not_the_delivered_one_is_blocked():
    # Round 22 (AUUNEMP): the pack ended at 2.63 to 8.23 and the answer said 4.3% to 5.5%.
    p = pack([4.86] * 24)
    full = p["full"]["forecast"]["quantiles"]
    full["0.1"][-1], full["0.9"][-1] = 2.6318, 8.2283
    wrong = "The likely range is 4.3% to 5.5% by October 2028."
    c = pq.check_answer_band(wrong, [p])
    assert not c["ok"] and "2.632 to 8.228" in c["detail"]
    assert pq.check_answer_band("The 10-90 range by October 2028 is 2.6% to 8.2%.", [p])["ok"]
    assert pq.check_answer_band("The centre stays near 4.9%.", [p]) is None


def test_a_goal_that_asks_for_drivers_blocks_until_one_is_tested():
    # Round 22 (GOLD, AURATE): "Search for the signals that move it" and no driver was ever tested.
    goal = "Forecast gold monthly over the next year. Search for the signals that move it."
    researched = [call("web_research", {"query": "gold drivers"}), pack(MOVING)]
    c = pq.check_signals_gathered(researched, goal)
    assert not c["ok"] and c["severity"] == "block" and "control_lab" in c["detail"]
    tested = [call("control_lab", {"with_covariates": True, "past_covariates": ["dxy"]}), pack(MOVING)]
    assert pq.check_signals_gathered(tested, goal)["ok"]
