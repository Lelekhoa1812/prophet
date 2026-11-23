"""Round 6m, goal audit item 4: labels that are real must stay blanked, and real measured p-values must stay checked.

Synthetic wording only.
"""
from __future__ import annotations

from prophet_agent.runtime import grounding


def test_an_interval_level_written_as_a_band_is_still_a_label():
    assert grounding.figures("10–90% bands run wide") == []
    assert grounding.figures("the (10–90%) range") == []
    assert grounding.figures("summing each client's 10-90% interval, is 3,871,341") == ["3,871,341"]


def test_an_iso_year_month_and_day_is_still_a_date_label():
    assert grounding.figures("Recorded on 2026-01-19 and again 2026-11.") == []


def test_a_month_and_a_real_day_range_is_still_a_date_label():
    assert grounding.figures("Sales from 16-19 January were 45.") == ["45"]
    assert grounding.figures("Peak in March 3-5 was 1200.") == ["1200"]


def test_a_significance_threshold_is_still_a_label_and_a_measured_p_is_kept():
    assert grounding.figures("Result: p < 0.05, so the effect is significant.") == []
    assert grounding.figures("The difference was p = 0.04 between arms.") == ["0.04"]


def test_a_month_with_a_percentage_is_still_kept():
    assert grounding.figures("May 5% growth") == ["5%"]
