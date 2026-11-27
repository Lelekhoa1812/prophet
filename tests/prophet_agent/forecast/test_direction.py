"""A flat centre line gets the best-supported direction (owner review of round 14 charts, 2026-11-27)."""
import numpy as np

from prophet_agent.forecast import direction as d


def _flat_fc(last, h, spread):
    w = [spread * (i + 1) ** 0.5 for i in range(h)]
    return {"point": [last] * h, "quantiles": {"0.1": [last - x for x in w], "0.5": [last] * h, "0.9": [last + x for x in w]}}


def test_a_flat_line_is_detected_and_a_sloped_one_is_not():
    assert d.is_flat([5.0] * 10, [4.0] * 10, [6.0] * 10)
    assert not d.is_flat(np.linspace(5, 7, 10), [4.0] * 10, [6.0] * 10)


def test_a_rising_series_gets_a_rising_centre_with_a_positive_band():
    rng = np.random.default_rng(1)
    x = 100 + np.arange(300) * 0.4 + rng.normal(0, 1.0, 300)
    fc = _flat_fc(float(x[-1]), 20, 3.0)
    out = d.tilt(fc, x)
    assert out["tilt_check"]["adopt"] is True
    assert out["centre"]["rule"] in ("damped_trend", "recent_drift")
    assert out["point"][-1] > out["point"][0] > float(x[-1]) - 1
    assert min(out["quantiles"]["0.1"]) > 0
    assert out["timesfm_point"] == fc["point"]


def test_a_trendless_series_stays_flat_and_says_why():
    rng = np.random.default_rng(2)
    x = 50 + rng.normal(0, 1.0, 300)
    fc = _flat_fc(float(x[-1]), 20, 10.0)
    out = d.tilt(fc, x)
    assert out["tilt_check"]["adopt"] is False
    assert out["point"] == fc["point"]
    assert "range is the forecast" in out["tilt_check"]["reason"]


def test_a_centre_that_already_moves_is_left_alone():
    fc = {"point": list(np.linspace(10, 14, 8)), "quantiles": {"0.1": [8.0] * 8, "0.9": [16.0] * 8}}
    assert d.tilt(fc, np.arange(100.0)) is fc


def test_a_positive_series_never_gets_a_band_through_zero():
    x = np.exp(np.linspace(np.log(200), np.log(80), 300))          # a falling price
    fc = _flat_fc(float(x[-1]), 20, 6.0)
    out = d.tilt(fc, x)
    if out["tilt_check"]["adopt"]:
        assert min(out["quantiles"]["0.1"]) > 0 and out["point"][-1] < out["point"][0]


def test_a_positive_series_band_that_reaches_zero_is_rebuilt_in_ratio():
    from prophet_agent.tools.report_tool import _positive_band
    h = 10
    fc = {"point": [150.0] * h, "quantiles": {"0.1": [120.0 - 14 * i for i in range(h)], "0.2": [130.0] * h, "0.3": [140.0] * h,
                                             "0.4": [145.0] * h, "0.5": [150.0] * h, "0.6": [155.0] * h, "0.7": [165.0] * h,
                                             "0.8": [180.0] * h, "0.9": [200.0 + 10 * i for i in range(h)]}}
    out = _positive_band(fc, [120.0, 130.0, 150.0])
    assert min(out["quantiles"]["0.1"]) > 0 and out["band_shape"]["log_symmetric"]
    assert out["quantiles"]["0.9"] == fc["quantiles"]["0.9"]
    assert _positive_band(fc, [-1.0, 5.0]) is fc                   # a series that goes negative keeps its band


def _tilted_obs(rule="damped_trend"):
    return [{"kind": "tool", "tool": "forecast_report", "args": {"centre": "naive"}, "result": {"forecast": {}},
             "full": {"forecast": {"centre": {"rule": rule}}}}]


def test_an_answer_cannot_call_a_tilted_line_flat():
    from prophet_agent.runtime import contracts
    obs = _tilted_obs()
    bad = contracts._flat_claim_after_tilt("The centre line is a level, so read the range.", obs)
    assert bad and bad["name"] == "answer_flat_claim_after_tilt" and not bad["ok"]
    assert contracts._flat_claim_after_tilt("The centre line follows a damped trend.", obs) is None
    assert contracts._flat_claim_after_tilt("The centre line is a level.", _tilted_obs("naive")) is None


def test_a_tilted_centre_is_not_a_level_line():
    from prophet_agent.runtime import contracts
    assert contracts._delivered_centre(_tilted_obs()) == "damped_trend"


def test_a_trend_rule_that_contradicts_the_models_own_direction_is_not_drawn():
    rng = np.random.default_rng(3)
    x = 100 + np.arange(300) * 0.4 + rng.normal(0, 1.0, 300)          # rising history
    fc = _flat_fc(float(x[-1]), 20, 3.0)
    fc["timesfm_point"] = [float(x[-1]) - 0.4 * (i + 1) for i in range(20)]       # the model's own median falls
    out = d.tilt(fc, x)
    assert out["tilt_check"]["adopt"] is False
    tried = out["tilt_check"]["tried"]
    assert all("no agreement" in tried[r]["result"] for r in ("damped_trend", "recent_drift", "momentum"))
    assert tried["mean_reversion"]["result"] != "passed"     # it agrees with the falling median, and loses on error instead


def test_a_recent_climb_is_drawn_where_a_long_drift_averages_it_away():
    rng = np.random.default_rng(4)
    flat = 4.0 + rng.normal(0, 0.05, 500)
    climb = 4.0 + np.linspace(0, 0.9, 40) + rng.normal(0, 0.03, 40)       # a year of climbing after a long level stretch
    x = np.concatenate([flat, climb])
    out = d.tilt(_flat_fc(float(x[-1]), 6, 0.3), x)
    assert out["tilt_check"]["adopt"] is True and out["centre"]["rule"] in d.TILT_RULES
    assert out["point"][-1] > out["point"][0] > float(x[-1])
    assert out["tilt_check"]["tried"]["recent_drift"]["result"].startswith("moves under")


def _pack(tilt_adopt, **extra):
    return {"kind": "tool", "tool": "forecast_report", "args": {}, "result": {"forecast": {}, "tilt_check": {"adopt": tilt_adopt}}, **extra}


def test_a_flat_pack_with_no_driver_tested_is_sent_back():
    from prophet_agent.runtime import contracts
    bad = contracts._signal_search_missing([_pack(False)])
    assert bad and bad["name"] == "answer_signal_search" and not bad["ok"]


def test_the_signal_search_is_satisfied_by_a_covariate_test_or_a_drawn_trend():
    from prophet_agent.runtime import contracts
    lab = {"kind": "tool", "tool": "control_lab", "args": {"source": {}}, "result": {"candidates": [{"name": "with_covariates"}]}}
    assert contracts._signal_search_missing([lab, _pack(False)]) is None
    passed = {"kind": "tool", "tool": "timesfm_forecast", "args": {"past_covariates": ["peer"]}, "result": {}}
    assert contracts._signal_search_missing([passed, _pack(False)]) is None
    assert contracts._signal_search_missing([_pack(True)]) is None
    null_only = {"kind": "tool", "tool": "calibrate_intervals", "args": {"past_covariates": None, "future_covariates": None}, "result": {}}
    assert contracts._signal_search_missing([null_only, _pack(False)])


def _moving_obs(point, verdict="flat_expected"):
    return [{"kind": "tool", "tool": "forecast_report", "args": {"centre": "timesfm"}, "result": {"forecast": {}},
             "full": {"forecast": {"point": point, "quantiles": {"0.1": [x - 0.3 for x in point], "0.9": [x + 0.3 for x in point]},
                                   "health": {"verdict": verdict}}}}]


def test_a_line_that_moves_is_not_a_level_whatever_the_health_verdict_says():
    from prophet_agent.runtime import contracts
    rising = _moving_obs([4.9, 4.92, 4.96, 4.99, 5.01, 5.04])
    assert contracts._drawn_line_moves(rising)[0] is True
    assert contracts._level_line_applies(rising) is False
    bad = contracts._flat_claim_after_tilt("The centre line is a flat level, so read the range.", rising)
    assert bad and bad["name"] == "answer_flat_claim_after_tilt"
    level = _moving_obs([5.0] * 6)
    assert contracts._drawn_line_moves(level)[0] is False
    assert contracts._flat_claim_after_tilt("The centre line is a flat level.", level) is None


def test_a_four_point_mean_is_not_called_a_season_mean():
    from prophet_agent.runtime import contracts
    bt = {"kind": "tool", "tool": "backtest_series", "args": {}, "result": {
        "models": {"timesfm": {"mase": 5.42}, "mean_last_season": {"mase": 4.8}}, "best_baseline": "mean_last_season",
        "origins": 11, "horizon": 42, "verdict": "baseline_better", "timesfm_vs_best_baseline_pct": -12.9,
        "mean_last_season_window": {"points": 4, "season": 1, "note": "mean of the last 4 points"}}}
    text = contracts.trust_statement([bt])
    assert "average of the last 4 points" in text and "average of the last season" not in text


def test_an_adopted_tilt_tells_the_health_note_the_line_is_not_the_models_flat_median():
    rng = np.random.default_rng(1)
    x = 100 + np.arange(300) * 0.4 + rng.normal(0, 1.0, 300)
    fc = _flat_fc(float(x[-1]), 20, 3.0)
    fc["health"] = {"verdict": "flat_expected", "causes": ["no repeating pattern stands out in the history, so a flat median is the honest forecast"]}
    out = d.tilt(fc, x)
    assert out["tilt_check"]["adopt"] is True
    assert "tilt" in out["health"]["drawn_line"] and "honest forecast" not in " ".join(out["health"]["causes"])
    assert out["health"]["verdict"] == "flat_expected"


def test_the_signal_search_is_an_evidence_gate_the_planner_can_repair():
    from prophet_agent.runtime import contracts
    pack = {"kind": "tool", "tool": "forecast_report", "args": {}, "result": {"forecast": {"point": [1.0, 1.0]}, "tilt_check": {"adopt": False}},
            "full": {"forecast": {"point": [1.0, 1.0], "quantiles": {"0.1": [0.0, 0.0], "0.9": [2.0, 2.0]}}}}
    names = [c["name"] for c in contracts.check_evidence([pack])]
    assert "answer_signal_search" in names


def test_a_sentence_about_the_models_own_median_is_not_a_claim_about_the_drawn_line():
    from prophet_agent.runtime import contracts
    obs = _tilted_obs()
    assert contracts._flat_claim_after_tilt("TimesFM produced a flat median line; a damped trend was drawn instead.", obs) is None
    assert contracts._flat_claim_after_tilt("The median line is flat.", obs)


def test_a_baseline_score_line_is_not_a_claim_about_the_cycle():
    from prophet_agent.runtime import grounding
    line = "Seasonal-naive (lag 365) MASE 1.18, far worse, confirming the 365-day season is wrong for this series."
    assert grounding.claimed_periods(line) == []
    assert grounding.claimed_periods("The data has a 7-day cycle.")


# ILOSTAT, Viet Nam, unemployment rate, 15+, total, quarterly, 2011-Q1 to 2024-Q4 (DF_UNE_DEAP_SEX_AGE_RT, read 2026-11-27).
VN_QUARTERLY = [1.352, 0.992, 0.874, 0.787, 1.086, 0.966, 1.074, 0.961, 1.389, 1.287, 1.445, 1.144, 1.377, 1.125, 1.341, 1.182,
                1.923, 1.923, 1.771, 1.404, 2.167, 1.987, 1.768, 1.464, 2.352, 1.724, 1.85, 1.562, 1.246, 1.211, 1.099, 1.037,
                1.665, 1.697, 1.713, 1.649, 1.882, 2.243, 2.187, 2.105, 1.997, 1.755, 2.541, 2.651, 1.631, 1.619, 1.569, 1.461,
                1.547, 1.579, 1.674, 1.627, 1.52, 1.559, 1.586, 1.452]


def test_a_seasonal_rate_that_swings_around_a_level_is_not_drawn_flat():
    """Owner, 2026-11-27 ("look too flat"): the VN quarterly chart drew the last value for six quarters over a history that
    repeats every four quarters. Mean reversion with the seasonal profile scores under the flat line's error on both
    vintages, and carries the cycle."""
    for x in (np.array(VN_QUARTERLY), np.array(VN_QUARTERLY[:51])):       # ILOSTAT to 2024-Q4; the mirror's copy to 2023-Q3
        fc = _flat_fc(float(x[-1]), 6, 0.2)
        out = d.tilt(fc, x, period=4)
        assert out["tilt_check"]["adopt"] is True, out["tilt_check"]
        assert out["centre"]["rule"] == "mean_reversion"
        assert out["centre"]["evidence"]["error_ratio"] < 1.0
        path = np.array(out["point"])
        assert path.max() - path.min() > 0.1          # the season shows: Q1 above Q4 by about a fifth of a point


def test_mean_reversion_is_refused_on_a_trending_series():
    rng = np.random.default_rng(3)
    x = 100 + np.arange(200) * 0.5 + rng.normal(0, 1.0, 200)
    ev = d.tilt_evidence(x, 6, "mean_reversion", 12)
    assert ev["scored"] and ev["error_ratio"] > d.TOLERANCE


def test_the_seasonal_profile_reads_each_phase_and_sums_to_zero():
    x = np.tile([1.0, 0.0, -1.0, 0.0], 8) + 5.0
    prof = d.seasonal_profile(x, 4)
    assert np.allclose(prof, [1.0, 0.0, -1.0, 0.0]) and abs(prof.sum()) < 1e-9
    assert d.seasonal_profile(x, None) is None and d.seasonal_profile(x[:8], 4) is None
