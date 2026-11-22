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

    def __init__(self, level=100.0, dip=None):
        self.level, self.dip = level, dip        # dip = (step index, value) to force one day below zero

    def forecast_batch(self, series_list, horizon):
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
    engine()
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
