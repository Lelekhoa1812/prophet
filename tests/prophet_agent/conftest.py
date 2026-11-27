"""Shared fixtures for the agent tests."""
from __future__ import annotations

import pytest


@pytest.fixture()
def old_stop_gate(monkeypatch):
    """The stop limits from before the progress-aware gate (2026-11-27): no pivots, three rejected finishes. Tests written to
    check when stall and rejection detection fire use it, so they keep checking the detection; the pivot and the later
    write-up are tested in test_stop_gate_progress.py."""
    from prophet_agent.runtime import goal

    monkeypatch.setattr(goal, "MAX_PIVOTS", 0)
    monkeypatch.setattr(goal, "PIVOT_AFTER_REJECTIONS", 99)
    monkeypatch.setattr(goal, "MAX_REJECTED_FINISHES", 3)
