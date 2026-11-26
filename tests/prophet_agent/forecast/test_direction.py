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
    assert all("no agreement" in t["result"] for t in out["tilt_check"]["tried"].values())
