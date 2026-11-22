"""A6-07 (health side): a flat_unexpected verdict names the detected period and the lab call that tests it.

Motivation vs Logic
Motivation: live retail rep4 detected period 5 and returned flat_unexpected, but the fix text did not say which period to
pass, and no control_lab ran with it. The planner had the number in the record and no instruction to use it.
Logic: the same synthetic 5-step cycle as the live case; the flat forecast is built, not produced by a model.
"""
from __future__ import annotations

import numpy as np

from prophet_agent.forecast.health import assess


def _cycle(n: int, period: int, amp: float = 10.0, level: float = 100.0, seed: int = 0) -> np.ndarray:
    t = np.arange(n)
    rng = np.random.default_rng(seed)
    return level + 0.02 * t + amp * np.sin(2 * np.pi * t / period) + rng.normal(0, 0.5, n)


def test_flat_unexpected_try_names_the_period_and_the_lab_call():
    hist = _cycle(600, 5)
    out = assess(hist, np.full(28, hist[-1]), context_used=len(hist))
    assert out["period"] == 5
    assert out["verdict"] == "flat_unexpected"
    text = " ".join(out["try"])
    assert "control_lab with period=5" in text
    assert "backtest_series with season=5" in text


def test_the_try_text_follows_the_detected_period_not_a_fixed_one():
    hist = _cycle(600, 7)
    out = assess(hist, np.full(28, hist[-1]), context_used=len(hist))
    assert out["verdict"] == "flat_unexpected"
    assert "period=7" in " ".join(out["try"])
