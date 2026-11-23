"""Locked invariant: the improvement bar in scripts/backtest_twin.py is 0.15.

This is a locked invariant from CLAUDE.md: "The 15 % improvement bar over the best baseline is not met on the synthetic
twin. Do not move it to claim a win." Moving the bar would turn a missed result into a claimed win without any change to
the model. If this test fails, do not edit the constant or the test; report the result against the bar as it stands.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "backtest_twin.py"


def _load_twin():
    spec = importlib.util.spec_from_file_location("backtest_twin_for_bar_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)   # main() sits behind the __main__ guard, so this only defines constants
    return module


def test_improvement_bar_is_locked_at_fifteen_percent():
    twin = _load_twin()
    assert twin.BAR == 0.15
    assert isinstance(twin.BAR, float)
