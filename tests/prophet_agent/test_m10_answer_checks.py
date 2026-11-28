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
