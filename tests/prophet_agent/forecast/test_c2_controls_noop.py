"""A6-08 (backtest side): a context that covers every origin is a no-op and is not counted as a treatment, and a configured
backtest says it was selected on the windows it scores.

Motivation vs Logic
Motivation: live retail rep4 ran backtest_series with context_points=504 on a 504-point history. The configured run equalled
the default (configured_vs_default_pct 0.0), but it was still reported as a tested configuration (audit A6-08).
Logic: a context of at least the last origin (n - horizon) gives every backtest origin its whole history, so it changes
nothing. It is dropped from the configuration and reported as a no-op. A configured backtest is labelled as selected on the
windows it scores. The engine is a stub that returns a different forecast when a control is given.
"""
from __future__ import annotations

import numpy as np

from prophet_agent.forecast.control import Control
from prophet_agent.forecast.rolling import rolling_backtest
from prophet_agent.forecast.types import Forecast


class _Engine:
    checkpoint = "stub"
    global_context = 4096

    def forecast_batch(self, series_list, horizon):
        point = np.stack([np.full(horizon, float(np.asarray(x)[-1])) for x in series_list])
        return Forecast(point, np.tile(np.linspace(-1, 9, 9), (len(series_list), horizon, 1)))

    def forecast_controlled(self, series, horizon, control):
        # A configured run that differs from the default: it drifts by the context length, so the difference is visible.
        base = float(np.asarray(series)[-1])
        point = np.full(horizon, base + (0.5 if control.context_points else 0.0))
        return Forecast(point, np.tile(np.linspace(-1, 9, 9), (horizon, 1)))


def _series(n: int = 504) -> np.ndarray:
    t = np.arange(n)
    rng = np.random.default_rng(7)
    return 50 + 0.05 * t + 4 * np.sin(2 * np.pi * t / 7) + rng.normal(0, 1, n)


def test_a_context_covering_every_origin_is_a_noop_and_not_a_treatment():
    x = _series(504)
    out = rolling_backtest(x, 5, engine=_Engine(), control=Control(context_points=504))
    assert "timesfm_default" not in out["models"]
    assert "configured_vs_default_pct" not in out
    assert "control" not in out
    assert out["control_noop"]["context_points"] == 504
    assert "already reads all" in out["control_noop"]["reason"]


def test_a_noop_context_mixed_with_a_real_option_keeps_the_real_option_only():
    x = _series(504)
    out = rolling_backtest(x, 5, engine=_Engine(), control=Control(context_points=504, znorm=True))
    assert out["control"] == {"znorm": True}
    assert "timesfm_default" in out["models"]
    assert out["control_noop"]["context_points"] == 504


def test_a_context_that_truncates_some_origin_is_a_real_treatment():
    x = _series(504)
    out = rolling_backtest(x, 5, engine=_Engine(), control=Control(context_points=100))
    assert "control_noop" not in out
    assert out["control"] == {"context_points": 100}
    assert "timesfm_default" in out["models"]


def test_a_configured_backtest_says_it_was_selected_on_the_windows_it_scores():
    x = _series(504)
    out = rolling_backtest(x, 5, engine=_Engine(), control=Control(context_points=100))
    assert out["selection"]["selected_on_these_windows"] is True
    assert out["selection"]["optimistic"] is True
    assert "control_lab" in out["selection"]["note"]


def test_the_lab_does_not_treat_a_window_covering_the_history_as_a_treatment():
    # Lab side (A6-08): at an origin whose history fits inside the window, the window's control is the default, so the lab
    # reuses the default's forecast and does not count a treatment there.
    from prophet_agent.forecast.lab import FoldContext, _grid
    grid = {c.name: c for c in _grid(None, False)}
    short = FoldContext(hist=np.arange(30, dtype=float), origin=30, horizon=5, period=None, past={}, future={})
    long_ = FoldContext(hist=np.arange(100, dtype=float), origin=100, horizon=5, period=None, past={}, future={})
    assert grid["ctx_8h"].prepare(short).control.is_default
    assert not grid["ctx_8h"].prepare(long_).control.is_default


def test_a_default_backtest_has_no_selection_block():
    out = rolling_backtest(_series(504), 5, engine=_Engine())
    assert "selection" not in out and "control_noop" not in out
