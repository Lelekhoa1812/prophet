"""A day or range of days beside a month name is a date label, not a figure; nothing else is relaxed."""
from prophet_agent.runtime.grounding import figures, unverified

HIST = [{"kind": "tool", "result": {"promo": "2026-01-16 to 2026-01-19", "mean": 528, "label": "Promo 9"}}]


def test_day_and_day_range_beside_a_month_are_labels():
    for text in ("runs 16-19 January", "runs 16\u201319 January", "Jan 16-19", "the 5 May sale", "on 3rd of June", "March 3, 2027"):
        assert figures(text) == [], text


def test_look_alike_words_and_real_figures_are_still_checked():
    assert figures("Mark 15 and Decision 12") == ["15", "12"]
    assert figures("May 5% growth") == ["5%"]
    assert figures("on 3rd of June sales were 40") == ["40"]


def test_invented_interval_widths_stay_unverified_while_promo_dates_pass():
    ans = "Promo 9 runs 16\u201319 January at 528 units, interval \u00b115\u201320%"
    assert unverified(ans, HIST, "forecast") == ["15", "20%"]
