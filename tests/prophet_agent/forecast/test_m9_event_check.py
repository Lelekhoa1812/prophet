"""M9-4 backtest check: an event feature is kept only when a rolling backtest (the existing run_backtest) shows it beats the
simple rules, with a paired sign-flip test across origins. Synthetic series only: every figure here is synthetic, from a
generator whose event effect is known (+25 on event days, AR(1) coefficient 0.6, noise sd 3)."""
import numpy as np
import pandas as pd
import pytest

from prophet_agent.forecast.event_check import check_feature, check_features, guess_season
from prophet_agent.sandbox import prophet_events as pe

HISTORY, HORIZON = 300, 10
EVENT_DAYS = [40, 103, 166, 229, 292, 305]   # 305 falls inside the horizon: a scheduled event, known ahead


def synthetic(seed: int):
    """Synthetic: a business-day series with a known earnings effect, and a null feature (random days, no effect)."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2024-01-01", periods=HISTORY + HORIZON)
    event = np.zeros(HISTORY + HORIZON)
    event[EVENT_DAYS] = 1.0
    null_days = rng.choice(HISTORY + HORIZON, 20, replace=False)
    null = np.zeros(HISTORY + HORIZON)
    null[null_days] = 1.0
    y = np.zeros(HISTORY)
    y[0] = 100.0
    eps = rng.normal(0, 3, HISTORY)
    for t in range(1, HISTORY):
        y[t] = 100 + 0.6 * (y[t - 1] - 100) + 25 * event[t] + eps[t]
    return dates, y, event, null


def as_features(dates, event, null):
    table = pd.DataFrame({"date": [str(dates[i].date()) for i in EVENT_DAYS], "end_date": "", "type": "earnings"})
    other = pd.DataFrame({"date": [str(dates[i].date()) for i in np.where(null)[0]], "end_date": "", "type": "other"})
    feats = pe.event_features(dates[:HISTORY], table, HORIZON)
    feats.update(pe.event_features(dates[:HISTORY], other, HORIZON))
    return feats


def test_a_feature_with_a_known_effect_is_kept_with_a_measured_gain():
    dates, y, event, null = synthetic(100)
    feats = as_features(dates, event, null)
    verdict = check_features(y, feats, HORIZON, season=guess_season(dates[:HISTORY]))
    v = verdict["verdicts"]["ev_earnings"]
    assert v["kept"] is True
    assert v["improvement_vs_best_rule"] > 0.05
    assert v["p_feature_better"] <= 0.05 / 2        # Bonferroni over the two candidates
    assert "ev_earnings" in verdict["kept"]
    assert v["origins"] == 26 and v["horizon"] == HORIZON


def test_a_null_feature_is_not_kept_under_the_multiplicity_correction():
    dates, y, event, null = synthetic(100)
    feats = as_features(dates, event, null)
    verdict = check_features(y, feats, HORIZON, season=5)
    assert verdict["verdicts"]["ev_other"]["kept"] is False
    assert "ev_other" not in verdict["kept"]


def test_a_single_feature_check_reports_the_rule_it_beat_and_the_figures():
    dates, y, event, null = synthetic(101)
    feats = as_features(dates, event, null)
    v = check_feature(y, "ev_earnings", feats["ev_earnings"], HORIZON, season=5)
    assert v["best_rule"] in {"ar1", "seasonal_naive"}
    assert v["challenger_head_wape"] < v["best_rule_head_wape"]
    assert set(v) >= {"feature", "challenger_head_wape", "best_rule_head_wape", "improvement_vs_best_rule",
                      "p_feature_better", "origins", "horizon", "kept"}


def test_the_feature_must_span_history_plus_horizon():
    dates, y, event, null = synthetic(102)
    with pytest.raises(ValueError, match="must span the history plus the horizon"):
        check_feature(y, "ev_earnings", np.zeros(HISTORY), HORIZON)


def test_a_short_series_refuses_the_backtest_with_the_reason():
    dates, y, event, null = synthetic(103)
    with pytest.raises(ValueError):
        check_feature(y[:12], "ev_earnings", np.zeros(12 + HORIZON), HORIZON)


def test_the_season_is_read_from_the_calendar():
    bdays = pd.bdate_range("2024-01-01", periods=40)
    assert guess_season(bdays) == 5
    assert guess_season(pd.date_range("2024-01-01", periods=40, freq="D")) == 7
    assert guess_season(pd.date_range("2024-01-05", periods=40, freq="W-FRI")) == 1
