"""timesfm_forecast records two things the planner could not see before: the quantile crossings the model returned (A6-12,
tool side) and the arguments it accepted but ignored (A6-11). The engine is stubbed; no model is loaded.

Motivation vs Logic
Motivation: a refused "season" cost a live call (WTI rep2), so the tool keeps accepting season and centre. Accepting them
silently let the planner believe a season had been applied (A6-11). The config now says which arguments were ignored and why.
Logic: the crossing count is read from the engine right after its call; an engine that does not report counts gives none
rather than an invented zero.
"""
from __future__ import annotations

import numpy as np

from prophet_agent.forecast import timesfm as timesfm_mod
from prophet_agent.forecast.types import Forecast
from prophet_agent.tools.timesfm_tool import _run


class _CountingStub:
    checkpoint = "stub-checkpoint"
    global_context = 16

    def __init__(self, counts):
        self._counts = counts

    def forecast_batch(self, series_list, horizon):
        n = len(series_list)
        point = np.stack([np.full(horizon, float(np.asarray(x)[-1])) for x in series_list])
        q = np.tile(np.linspace(0, 8, 9), (n, horizon, 1))
        self._last = list(self._counts)
        return Forecast(point, q)

    def last_quantile_crossings(self):
        return getattr(self, "_last", None)


class _NoCountStub(_CountingStub):
    last_quantile_crossings = None  # an engine that does not report counts


def test_config_records_the_quantile_crossings_of_the_call(monkeypatch):
    monkeypatch.setattr(timesfm_mod.TimesFMEngine, "get", classmethod(lambda cls: _CountingStub([3])))
    out = _run({"values": list(range(40)), "horizon": 3}, "k")
    assert out["config"]["quantile_crossings"] == 3


def test_an_engine_without_counts_records_nothing_rather_than_zero(monkeypatch):
    monkeypatch.setattr(timesfm_mod.TimesFMEngine, "get", classmethod(lambda cls: _NoCountStub([])))
    out = _run({"values": list(range(40)), "horizon": 3}, "k")
    assert "quantile_crossings" not in out["config"]


def test_season_and_centre_are_accepted_but_reported_as_ignored(monkeypatch):
    monkeypatch.setattr(timesfm_mod.TimesFMEngine, "get", classmethod(lambda cls: _CountingStub([0])))
    out = _run({"values": list(range(40)), "horizon": 3, "season": 7, "centre": "naive"}, "k")
    ignored = out["config"]["ignored_arguments"]
    assert set(ignored) == {"season", "centre"}
    assert "ignored" in ignored["season"] and "season" in ignored["season"]
    assert len(out["point"]) == 3                         # the call still goes through


def test_no_ignored_arguments_no_key(monkeypatch):
    monkeypatch.setattr(timesfm_mod.TimesFMEngine, "get", classmethod(lambda cls: _CountingStub([0])))
    out = _run({"values": list(range(40)), "horizon": 3}, "k")
    assert "ignored_arguments" not in out["config"]


def test_ignored_arguments_names_season_and_centre_only_when_given():
    from prophet_agent.tools.timesfm_tool import ignored_arguments
    assert set(ignored_arguments({"season": 5, "centre": "naive", "horizon": 3})) == {"season", "centre"}
    assert ignored_arguments({"horizon": 3, "values": [1.0]}) == {}
