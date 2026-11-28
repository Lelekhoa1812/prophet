"""Mission 10: the answer states the combination's leading weight and the season model's measured error."""
from prophet_agent.runtime import prediction_quality as pq

COMB = [{"kind": "tool", "tool": "forecast_report", "result": {"combination": {"weights": {
    "timesfm": 0.4634, "vacancy_model": 0.2988, "naive": 0.058}}}}]
LEAGUE = [{"kind": "tool", "tool": "league_backtest", "result": {"summary": ["x"]},
           "full": {"summary": [{"cut": 12, "model": "history_market", "mae": 6.53},
                                {"cut": 12, "model": "pace", "mae": 8.98}]}}]


def test_combination_weights_must_be_stated():
    bad = pq.check_combination_stated("The forecast rises to 5.1 per cent by 2028.", COMB)
    assert bad["ok"] is False and "timesfm at weight 0.46" in bad["detail"]
    good = pq.check_combination_stated("The line combines five models; TimesFM 3 carries 46% of the weight, the "
                                       "vacancy model 30%.", COMB)
    assert good["ok"] is True
    assert pq.check_combination_stated("", COMB) is None
    assert pq.check_combination_stated("text", [{"kind": "tool", "tool": "forecast_report", "result": {}}]) is None


def test_league_backtest_error_must_be_quoted():
    bad = pq.check_league_backtest_stated("Arsenal are favourites.", LEAGUE)
    assert bad["ok"] is False and "6.53" in bad["detail"]
    assert pq.check_league_backtest_stated("On past seasons the final table was out by 6.5 points per club.", LEAGUE)["ok"]
    assert pq.check_league_backtest_stated("text", []) is None


def test_both_run_inside_prediction_checks():
    names = {c["name"] for c in pq.prediction_checks(COMB + LEAGUE, "goal", "an answer")}
    assert {"combination_stated", "league_backtest_stated"} <= names


CAL = [{"kind": "tool", "tool": "forecast_report", "result": {"combination": {
    "weights": {"timesfm": 0.5, "naive": 0.5},
    "band": {"coverage_scored": 0.74, "widened": True, "calibration": {
        "points": 240, "below_p10": 0.14, "above_p90": 0.12, "above_median": 0.55, "reading": "about right in width"}}}}}]


def test_band_calibration_must_be_stated_with_any_widening():
    """M10-7: the answer says how the combination's band did on the backtest, and that it was widened."""
    bad = pq.check_calibration_stated("The range is 4.2 to 5.6 per cent.", CAL)
    assert bad["ok"] is False and "74%" in bad["detail"] and "widened" in bad["detail"]
    half = pq.check_calibration_stated("On the backtest the 10-90 band held 74% of outcomes.", CAL)
    assert half["ok"] is False                                           # the widening is not mentioned
    good = pq.check_calibration_stated("On the backtest the band held 74% of outcomes (14% fell below the 10th "
                                       "percentile, 12% above the 90th), so each tail was widened.", CAL)
    assert good["ok"] is True
    assert pq.check_calibration_stated("text", COMB) is None
    assert "calibration_stated" in {c["name"] for c in pq.prediction_checks(CAL, "goal", "an answer")}


def test_scenario_shifts_trace_to_calls_and_judgement_is_named():
    """M10-6: every scenario shift cites a call of the mission; judgement probabilities are called judgement."""
    page = {"kind": "tool", "tool": "web_research", "args": {"queries": ["rba"]},
            "result": {"sources": [{"url": "https://www.rba.gov.au/media-releases/2026/mr-26-30.html"}]}}
    sc = {"kind": "tool", "tool": "scenario_forecast", "result": {"end": {}}, "args": {"pack": "r1", "scenarios": [
        {"name": "RBA hikes", "probability": 0.3, "shift": 0.25, "evidence": "https://www.rba.gov.au/media-releases/2026/mr-26-30.html"},
        {"name": "market path", "probability": 0.2, "path_of": "market_path"}]}}
    assert pq.check_scenarios_traced("A 30% judgement that the RBA hikes.", [page, sc])["ok"] is True
    assert pq.check_scenarios_traced("A 30% chance that the RBA hikes.", [page, sc])["ok"] is False
    loose = {**sc, "args": {"pack": "r1", "scenarios": [{"name": "war", "probability": 0.1, "shift": 1.0,
                                                          "evidence": "https://example.org/never-fetched"}]}}
    bad = pq.check_scenarios_traced("judgement", [page, loose])
    assert bad["ok"] is False and "war" in bad["detail"]
    assert pq.check_scenarios_traced("x", [page]) is None
