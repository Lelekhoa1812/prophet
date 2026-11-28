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


def test_a_finding_date_no_tool_printed_is_unverified():
    # Live SPX (round 20261128-063203): two of three level-shift dates appear in no tool output.
    from prophet_agent.runtime.grounding import unverified
    h = [{"kind": "tool", "tool": "sandbox_python", "args": {}, "result": {"stdout": "break at 2020-11-05 00:00:00, 1.88x"}}]
    out = unverified("Scan found three level shifts (2020-11-05, 2024-01-24, 2025-08-08).", h, "S&P 500")
    assert "2024-01-24" in out and "2025-08-08" in out and "2020-11-05" not in out
    assert unverified("Figures as of 2026-11-28 are the newest.", h, "S&P 500") == []
    assert unverified("The forecast for 2027-03-31 is in the table.", h, "S&P 500") == []


def test_a_flagged_date_does_not_break_the_slip_correction():
    # Live EPL 28f79257 (round 20261128-110827): snap parsed the flagged '2026-11-28' as a number and the mission failed.
    from prophet_agent.runtime.grounding import snap
    h = [{"kind": "tool", "tool": "sandbox_python", "args": {}, "result": {"stdout": "total 557,051.21"}}]
    text, fixed = snap("A level shift on 2026-11-28 lifted the total to 557,050.", h, "sales")
    assert "2026-11-28" in text and fixed == [("557,050", "557,051")], "no crash, and the slip is still corrected"
