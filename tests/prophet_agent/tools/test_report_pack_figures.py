"""The report pack's figures (audit round 3, A7-02 and A7-03): every table row and total the answer may quote comes from the same
arrays the pack writes to out/forecast.json, a non-negative series never shows a negative day or total, and the headline total
carries the health verdict.

No real TimesFM: the engine is a fake. The pack itself is built in the real sandbox, as the other artefact tests do.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from prophet_agent.forecast import timesfm as timesfm_mod
from prophet_agent.forecast.types import Forecast
from prophet_agent.tools import report_tool as rt

H = 14


class _Engine:
    """Forecasts a flat centre at ``level`` with a band that widens with the step, so day 1 and day 14 differ."""
    checkpoint = "stub"
    global_context = 64

    def __init__(self, level=100.0, dip=None, follow=False):
        self.level, self.dip, self.follow = level, dip, follow   # dip = (step index, value) to force one day below zero
                                                                 # follow = repeat the last weekly cycle (a healthy shape)

    def forecast_batch(self, series_list, horizon):
        if self.follow:
            point = np.array([[float(ctx[-7 + (j % 7)]) for j in range(horizon)] for ctx in series_list])
        else:
            point = np.full((len(series_list), horizon), float(self.level))
        if self.dip is not None:
            point[:, self.dip[0]] = self.dip[1]
        widen = (1.0 + np.arange(horizon))[None, :, None]
        quant = point[:, :, None] + np.linspace(-1.0, 1.0, 9)[None, None, :] * widen * 5.0
        return Forecast(point, quant)


@pytest.fixture
def engine(monkeypatch):
    def use(**kw):
        eng = _Engine(**kw)
        monkeypatch.setattr(timesfm_mod.TimesFMEngine, "get", classmethod(lambda cls: eng))
        return eng
    return use


def _values(n=60):
    return [round(50 + 0.1 * i + 3 * np.sin(2 * np.pi * i / 7), 4) for i in range(n)]


def _pack_json(out: dict) -> dict:
    """The forecast.json the pack wrote for this run: the numbers the files hold."""
    return json.loads((Path(out["manifest"]).parent / "out" / "forecast.json").read_text())


def _fc(point, verdict="ok"):
    """A forecast as _one returns it: a centre, nine quantile lines, and a health verdict."""
    point = [float(v) for v in point]
    levels = ["0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9"]
    return {"point": point, "quantiles": {lv: [p + (i - 4) for p in point] for i, lv in enumerate(levels)},
            "config": {"history_points": 60}, "health": {"verdict": verdict}}


# ---------------------------------------------------------------------------------------------- A7-02: tables match files

def test_the_observation_table_has_every_step_and_equals_the_arrays_in_the_file(engine):
    engine()
    out = rt._run({"values": _values(), "horizon": H, "title": "Table"}, "k")
    view = rt.observe(out)
    rows = view["table"]["rows"]
    file = _pack_json(out)
    assert len(rows) == H
    assert [r["point"] for r in rows] == file["point"]
    assert [r["q10"] for r in rows] == file["q10"]
    assert [r["q90"] for r in rows] == file["q90"]


def test_the_table_band_differs_between_the_first_and_last_step(engine):
    # A7-02: the answer printed day 14's band in the day 1 column. The rows must carry each step's own band.
    engine()
    out = rt._run({"values": _values(), "horizon": H, "title": "Band"}, "k")
    rows = rt.observe(out)["table"]["rows"]
    assert (rows[0]["q90"] - rows[0]["q10"]) < (rows[-1]["q90"] - rows[-1]["q10"])


def test_the_total_quoted_by_the_observation_is_the_sum_of_the_table_rows(engine):
    engine(follow=True)              # a forecast that follows the weekly cycle, so its health verdict is ok and a total is quoted
    out = rt._run({"values": _values(), "horizon": H, "title": "Total"}, "k")
    view = rt.observe(out)
    assert view["centre_line_total"] == pytest.approx(round(sum(r["point"] for r in view["table"]["rows"]), 2))


def test_a_long_horizon_is_not_thinned_in_the_table(engine):
    # timesfm_tool._compact thins a forecast above 120 periods to 30 points; the table must not inherit that.
    engine()
    out = rt._run({"values": _values(), "horizon": 150, "title": "Long"}, "k")
    table = rt.observe(out)["table"]
    assert len(table["rows"]) == 10 and table["rows_omitted"] == 140   # first five and last five, the rest named
    assert table["rows"][-1]["step"] == 150 and table["file"] == "out/forecast.json"
# ---------------------------------------------------------------------------- A7-03: negative days and the headline total

def test_a_non_negative_series_never_shows_a_negative_day_or_total_in_the_file(engine):
    # Live retail rep3: day 9 at minus 3,606 was inside the headline total of a series that cannot be negative.
    engine(level=100.0, dip=(8, -3606.0))
    out = rt._run({"values": _values(), "horizon": H, "title": "Floor"}, "k")
    file = _pack_json(out)
    assert min(file["point"]) >= 0 and min(file["q10"]) >= 0
    view = rt.observe(out)
    assert min(r["point"] for r in view["table"]["rows"]) >= 0
    assert view["headline"]["total"] is None and view["headline"]["withheld"]


def test_a_flat_unexpected_health_verdict_withholds_the_headline_total_and_names_it():
    fc = _fc([80.0, 81.0, 82.0], verdict="flat_unexpected")
    view = rt.observe({"forecast": fc, "run_id": "r", "files": [], "manifest": "m"})
    assert view["headline"]["verdict"] == "flat_unexpected"
    assert view["headline"]["total"] is None and view["centre_line_total"] is None
    assert "flat_unexpected" in view["headline"]["withheld"]


def test_a_healthy_forecast_keeps_its_headline_total_and_the_verdict_beside_it():
    fc = _fc([80.0, 81.0, 82.0], verdict="ok")
    view = rt.observe({"forecast": fc, "run_id": "r", "files": [], "manifest": "m"})
    assert view["headline"] == {"verdict": "ok", "total": 243.0, "withheld": None}
    assert view["centre_line_total"] == 243.0


def test_a_flat_expected_forecast_keeps_its_headline_total():
    """flat_expected is a valid verdict (a random walk has no pattern to repeat; the range is the forecast), so its total is
    not withheld. Only a verdict that says the forecast is wrong or unchecked withholds it."""
    import sys
    import prophet_agent.tools.report_tool  # noqa: F401
    rt = sys.modules["prophet_agent.tools.report_tool"]
    fc = {"point": [10.0, 10.0, 10.0], "health": {"verdict": "flat_expected"}}
    out = rt._headline(fc)
    assert out["withheld"] is None and out["total"] is not None
    assert rt._headline({**fc, "health": {"verdict": "flat_unexpected"}})["total"] is None
