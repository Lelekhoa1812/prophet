"""The forecast health check: a flat line is judged against the history it came from. No model is loaded here; the forecasts
are constructed so the check's reasoning is what is under test."""
from __future__ import annotations

import numpy as np
import pytest

from prophet_agent.forecast.health import assess, dominant_period


def _cycle(n: int, period: int, amp: float = 10.0, level: float = 100.0, seed: int = 0) -> np.ndarray:
    t = np.arange(n)
    rng = np.random.default_rng(seed)
    return level + 0.02 * t + amp * np.sin(2 * np.pi * t / period) + rng.normal(0, 0.5, n)


def test_a_repeating_cycle_is_found_at_its_period():
    period, _ = dominant_period(_cycle(730, 7))
    assert period == 7


def test_a_flat_forecast_on_a_strong_cycle_is_flagged_as_unexpected():
    hist = _cycle(600, 7)
    flat = np.full(28, hist[-1])
    out = assess(hist, flat, context_used=len(hist))
    assert out["verdict"] == "flat_unexpected"
    assert any("repeats every 7" in c for c in out["causes"])


def test_a_forecast_that_keeps_the_cycle_is_ok():
    hist = _cycle(600, 7)
    t = np.arange(600, 628)
    point = 100 + 0.02 * t + 10 * np.sin(2 * np.pi * t / 7)
    out = assess(hist, point, context_used=len(hist))
    assert out["verdict"] == "ok"
    assert 0.8 < out["shape_ratio"] < 1.2


def test_a_random_walk_with_a_flat_median_is_an_honest_flat_forecast():
    rng = np.random.default_rng(3)
    hist = 50 + np.cumsum(rng.normal(0, 1, 800))
    flat = np.full(30, hist[-1])
    out = assess(hist, flat, context_used=len(hist))
    assert out["verdict"] == "flat_expected"
    assert "flat median is the honest forecast" in out["causes"][0]


def test_too_few_cycles_in_the_context_is_named_as_the_cause():
    hist = _cycle(900, 30)
    flat = np.full(20, hist[-1])
    out = assess(hist, flat, context_used=40, context_points=40)
    assert out["verdict"] == "flat_unexpected"
    assert any("context_points=40 holds" in c for c in out["causes"])
    assert any("at least 90" in t for t in out["try"])


def test_a_history_that_ends_in_a_filled_stretch_is_named():
    hist = _cycle(400, 7)
    hist[-30:] = hist[-31]
    out = assess(hist, np.full(14, hist[-1]), context_used=len(hist))
    assert any("identical values" in c for c in out["causes"])


def test_a_long_run_of_zeros_at_the_floor_of_a_count_is_not_a_filled_gap():
    hist = np.zeros(200)
    hist[::5] = 3
    out = assess(hist, np.zeros(14), context_used=len(hist))
    assert not any("identical values" in c for c in out.get("causes", []))


def test_short_history_is_a_check_not_a_verdict():
    out = assess(np.arange(10.0), np.arange(5.0))
    assert out["verdict"] == "check"


def test_a_window_far_shorter_than_a_cycle_cannot_be_called_flat_on_shape_alone():
    # Measured on the daily CO2 record: 60 days of a 364-day cycle looked "flat" in 7 of 13 holdouts, and TimesFM beat seasonal
    # naive in all 13. Shape over a sixth of a cycle is noise, so the verdict is "check" and points to the backtest.
    hist = _cycle(3000, 200, amp=10.0)
    flat = np.full(20, hist[-1])
    out = assess(hist, flat, context_used=len(hist))
    assert out["verdict"] == "check" and out["shape_reliable"] is False
    assert any("too short a window" in c for c in out["causes"])
    assert any("backtest_series" in t for t in out["try"])


def test_a_window_of_a_quarter_cycle_or_more_is_still_judged_on_shape():
    hist = _cycle(3000, 200, amp=10.0)
    out = assess(hist, np.full(120, hist[-1]), context_used=len(hist))
    assert out["verdict"] == "flat_unexpected"


def test_a_flat_median_at_the_last_value_is_said_to_carry_no_directional_view():
    """Added with the naive comparison: a flat median on a random walk sits at the last value, so the health check now
    says how far it is in typical one-step moves, and that it carries no direction."""
    rng = np.random.default_rng(3)
    hist = 50 + np.cumsum(rng.normal(0, 1, 800))
    out = assess(hist, np.full(30, hist[-1]), context_used=len(hist))
    assert out["vs_last_value"] == 0.0
    assert out["vs_last_value_note"].startswith("the median is within 0.0 typical one-step moves of the last value")
    assert "carries no directional view" in out["vs_last_value_note"]


def test_a_flat_median_well_away_from_the_last_value_is_not_called_directionless():
    rng = np.random.default_rng(3)
    hist = 50 + np.cumsum(rng.normal(0, 1, 800))
    step = float(np.mean(np.abs(np.diff(hist))))
    out = assess(hist, np.full(30, hist[-1] + 5 * step), context_used=len(hist))
    assert out["vs_last_value"] == pytest.approx(5.0, abs=0.01)
    assert "carries no directional view" not in out["vs_last_value_note"]
    assert "does carry a level view" in out["vs_last_value_note"]


def test_the_naive_comparison_is_absent_when_the_history_never_moves():
    out = assess(np.full(60, 7.0), np.full(10, 7.0), context_used=60)
    assert "vs_last_value" not in out
