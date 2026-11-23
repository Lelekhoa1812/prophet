"""Round 6l, goal audit item 4: the snap tolerance is stated and bounded in absolute units as well as relatively.

Synthetic figures only. The relative bound (SNAP_TOLERANCE, 0.05%) alone allowed a written 10,000,000 to be rewritten as the
printed 10,004,900. The snap now uses the smaller of that relative bound and SNAP_MAX_ABS (1,000 units).
"""
from __future__ import annotations

from prophet_agent.runtime.grounding import SNAP_MAX_ABS, SNAP_TOLERANCE, snap, unverified


def _printed(value):
    return [{"kind": "tool", "tool": "t", "args": {}, "result": {"v": value}}]


def test_the_tolerance_is_stated_as_a_relative_and_an_absolute_bound():
    assert SNAP_TOLERANCE == 0.0005
    assert SNAP_MAX_ABS == 1000.0


def test_a_large_written_figure_is_not_rewritten_to_a_printed_value_thousands_of_units_away():
    # 10,000,000 is 4,900.5 units from the printed 10,004,900.5: beyond the 1,000-unit bound, so it is left to the repair pass.
    text = "The target was 10,000,000 units."
    assert snap(text, _printed(10004900.5), "") == (text, [])
    assert unverified(text, _printed(10004900.5), "") == ["10,000,000"]


def test_a_copying_slip_of_a_hundred_units_is_still_corrected():
    out, fixed = snap("The total was 352,602 units.", _printed(352702.4), "")
    assert out == "The total was 352,702 units."
    assert fixed == [("352,602", "352,702")]


def test_a_copying_slip_in_the_centre_of_a_small_figure_is_still_corrected():
    out, fixed = snap("The total was 557,050 units.", _printed(557051.21), "")
    assert out == "The total was 557,051 units."
    assert fixed == [("557,050", "557,051")]


def test_the_absolute_bound_applies_just_inside_and_just_outside_1000_units():
    # 3,000,000 allows 1,500 by the relative bound, so 1,000 is the limit: 999.5 units is corrected, 1,000.5 is not.
    inside = snap("It was 3,000,000 units.", _printed(3000999.5), "")
    assert inside[0] == "It was 3,001,000 units." and len(inside[1]) == 1
    outside = snap("It was 3,000,000 units.", _printed(3001000.5), "")
    assert outside == ("It was 3,000,000 units.", [])
