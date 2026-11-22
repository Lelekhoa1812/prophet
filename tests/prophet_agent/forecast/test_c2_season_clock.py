"""A6-04: a given season is checked against the measured cycle, and a business-day series says so.

Motivation vs Logic
Motivation: live retail rep4 was business-day indexed (504 points) and was backtested and forecast at season=7 (a calendar
week). Health measured period 5 on the same series, and the answer called 7 a confirmed weekly cycle. Nothing compared the two.
Logic: season_check compares a given season with dominant_period on the history (in steps). business_day_calendar reads the
dates: weekend days absent and a Friday-to-Monday gap present means one week is 5 steps. The series here is synthetic (a
5-step cycle on weekdays), so the numbers are a construction test, not a measured result.
"""
from __future__ import annotations

import threading
from types import SimpleNamespace

import numpy as np

from prophet_agent.forecast.health import business_day_calendar, season_check
from prophet_agent.forecast.types import Forecast


def _business_days(n: int) -> list[str]:
    return [str(d) for d in np.busday_offset("2024-01-01", np.arange(n), roll="forward")]


def _weekday_cycle(n: int, period: int = 5, seed: int = 0) -> np.ndarray:
    t = np.arange(n)
    rng = np.random.default_rng(seed)
    return 100 + 12 * np.sin(2 * np.pi * t / period) + rng.normal(0, 1.0, n)


def test_a_business_day_calendar_is_recognised_with_a_five_step_week():
    cal = business_day_calendar(_business_days(120))
    assert cal == {"calendar": "business_day", "weekly_cycle_steps": 5,
                   "note": "weekend days are absent: one week is 5 steps of this series"}


def test_calendar_days_are_not_called_business_days():
    days = [str(d) for d in np.arange(np.datetime64("2024-01-01"), np.datetime64("2024-05-01"))]
    assert business_day_calendar(days) is None


def test_a_season_that_does_not_match_the_measured_cycle_is_flagged():
    x = _weekday_cycle(504)
    out = season_check(7, x, _business_days(504))
    assert out is not None
    assert out["season_given"] == 7 and out["period_measured"] == 5
    assert "5 steps" in out["warning"] and "one week" in out["warning"]


def test_a_matching_season_is_not_flagged():
    assert season_check(5, _weekday_cycle(504)) is None


def test_no_season_no_check_and_no_cycle_no_check():
    assert season_check(None, _weekday_cycle(504)) is None
    rng = np.random.default_rng(4)
    assert season_check(7, 50 + np.cumsum(rng.normal(0, 1, 500))) is None   # no repeating cycle to compare with


class _ForecastStub:
    checkpoint = "stub"
    global_context = 1024

    def forecast_batch(self, series_list, horizon):
        point = np.stack([np.full(horizon, float(np.asarray(x)[-1])) for x in series_list])
        return Forecast(point, np.tile(np.linspace(0, 8, 9), (len(series_list), horizon, 1)))


def test_the_backtest_result_carries_the_season_check_on_its_prefix():
    from prophet_agent.forecast.rolling import rolling_backtest
    x = _weekday_cycle(504)
    out = rolling_backtest(x, 5, season=7, engine=_ForecastStub(), dates=_business_days(504))
    assert out["season_used"] == 7
    assert out["season_check"]["period_measured"] == 5
    assert "one week" in out["season_check"]["warning"]
    assert out["season_check"]["measured_on_points"] == 504 - 5 * 12   # the prefix before the first test window


def test_the_backtest_without_a_given_season_has_no_season_check():
    from prophet_agent.forecast.rolling import rolling_backtest
    out = rolling_backtest(_weekday_cycle(504), 5, engine=_ForecastStub())
    assert "season_check" not in out


def test_the_forecast_tool_records_the_check_and_the_calendar(monkeypatch):
    from prophet_agent.forecast import timesfm as timesfm_mod
    from prophet_agent.tools.timesfm_tool import _run
    monkeypatch.setattr(timesfm_mod.TimesFMEngine, "get", classmethod(lambda cls: _ForecastStub()))
    out = _run({"values": list(_weekday_cycle(120)), "horizon": 3, "season": 7}, "k")
    assert out["config"]["season_check"]["period_measured"] == 5
    assert "series_step_unit" not in out["config"]        # values only: no dates, so no calendar claim


def test_the_forecast_tool_names_business_days_when_the_dates_are_business_days(monkeypatch):
    import importlib
    from prophet_agent.forecast import timesfm as timesfm_mod
    from prophet_agent.tools.timesfm_tool import _run
    timesfm_tool = importlib.import_module("prophet_agent.tools.timesfm_tool")   # the package also exports a function of this name
    monkeypatch.setattr(timesfm_mod.TimesFMEngine, "get", classmethod(lambda cls: _ForecastStub()))
    monkeypatch.setattr(timesfm_tool, "run_dates", lambda args, name: _business_days(120))
    out = _run({"values": list(_weekday_cycle(120)), "horizon": 3}, "k")
    assert out["config"]["series_step_unit"] == "business day"
    assert out["config"]["weekly_cycle_steps"] == 5
    assert out["config"]["series_step"] == 1.0            # the step median is still one day; the unit is what changed
