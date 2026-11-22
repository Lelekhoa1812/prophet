"""Round 4 follow-up (A7-13): the chart caption calls a band calibrated only when the forecast config says so
(web Results.tsx reads config.band_source and config.band_coverage). The report tool now writes both when calibration
is applied, and leaves the config alone when it is not."""
import sys

import prophet_agent.tools.report_tool  # noqa: F401

rt = sys.modules["prophet_agent.tools.report_tool"]


def _fake_calibrate(recommend):
    def calibrate(values, horizon, engine=None, control=None):
        return {"recommend": recommend, "heldout_coverage_raw": 0.62, "heldout_coverage_calibrated": 0.79,
                "scales_by_horizon_third": [1.2, 1.3, 1.4], "reason": "test",
                "calibrated": {"quantiles": {"0.1": [1.0], "0.9": [3.0]}}}
    return calibrate


def _run(monkeypatch, recommend):
    import prophet_agent.forecast.calibrate as cal
    import prophet_agent.forecast.timesfm as tf
    monkeypatch.setattr(cal, "calibrate", _fake_calibrate(recommend))
    monkeypatch.setattr(tf.TimesFMEngine, "get", classmethod(lambda cls: object()))
    fc = {"point": [2.0], "quantiles": {"0.1": [1.5], "0.9": [2.5]}, "config": {"horizon": 1}}
    return rt._calibrated(fc, [1.0] * 60, {"horizon": 1})


def test_a_calibrated_band_says_so_in_its_config(monkeypatch):
    out = _run(monkeypatch, "use_calibrated")
    assert out["config"]["band_source"] == "calibrated" and out["config"]["band_coverage"] == 0.79
    assert out["config"]["horizon"] == 1


def test_a_raw_band_keeps_its_config(monkeypatch):
    out = _run(monkeypatch, "keep_raw")
    assert "band_source" not in out["config"]
