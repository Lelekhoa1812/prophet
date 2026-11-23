"""Round 6f (2026-11-23): a level's total is withheld per series, not only as the panel total.

Live Beijing (3581d07d, d861b74e): the panel total was withheld for PM2.5, but the per-station "14-day total" was still
emitted, so the planner summed a level again. Every series here is synthetic.
"""
from __future__ import annotations

from prophet_agent.tools import report_tool

HEALTHY = {"verdict": "ok"}


def _fc(level: bool, point=(10.0, 12.0, 11.0)):
    band = [p + 2.0 for p in point], [p - 2.0 for p in point]
    fc = {"point": list(point), "quantiles": {"0.1": band[1], "0.9": band[0]}, "health": HEALTHY}
    if level:
        fc["level"] = True
    return fc


def test_a_level_series_has_no_headline_total_and_says_why():
    head = report_tool._headline(_fc(level=True))
    assert head["total"] is None
    assert "level" in head["withheld"] and "average over the horizon" in head["withheld"]


def test_a_flow_series_keeps_its_headline_total():
    head = report_tool._headline(_fc(level=False))
    assert head["total"] is not None and abs(head["total"] - 33.0) < 1e-6
    assert head["withheld"] is None


def test_a_level_has_no_range_for_its_total():
    assert report_tool.total_range(_fc(level=True)) is None
    assert report_tool.total_range(_fc(level=False)) is not None
