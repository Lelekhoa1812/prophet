"""Audit of the live ledger (partial missions, 2026-11-21): these were false positives of the grounding check, not
invented figures. Each case is a label or a date that the tokenizer used to read as a number."""
from prophet_agent.runtime.grounding import figures, unverified

HIST = [{"kind": "tool", "result": {"orders": [360, 365, 370, 375], "customers": 125}}]


def test_an_iso_date_inside_a_week_label_is_one_date_not_fragments():
    # Found in mission fd3d9cb9: "Week 2026-01-05" blanked the year, and the day and month became "-01" and "05".
    assert figures("| Week 2026-01-05: 360 orders |") == ["360"]
    assert figures("Week starting 2026-01-26 forecast 375") == ["375"]
    assert unverified("Week 2026-01-26 is 375", HIST, "forecast") == []


def test_plural_row_and_id_labels_and_their_ranges_are_labels():
    # Found in mission 19f1db8f ("rows 60 through 66") and 69b07e76 ("customer IDs 41-48").
    assert figures("rows 60 through 66 repeat rows 0 through 6") == []
    assert figures("The unmatched orders have customer IDs 41\u201348 in the list") == []
    assert figures("promo peaks (days 3, 10, 17, 24)") == []


def test_standard_quantile_level_pairs_are_labels():
    # Found in missions f6700fe0 and d9738d05: "10\u201390% bands" names the interval level, it is not a measured value.
    for text in ("10\u201390% bands run wide", "the (10\u201390%) range", "5-95% interval", "2.5\u201397.5% band"):
        assert figures(text) == [], text


def test_invented_widths_and_derived_numbers_are_still_checked():
    # Only canonical quantile levels are labels. A width such as +/-15\u201320% and a derived count stay checked.
    assert figures("roughly \u00b115\u201320% around the point") == ["15", "20%"]
    assert figures("There are 125 different customers") == ["125"]
    assert figures("the peak is 2.6\u00d7 the next month") == ["2.6"]
    assert figures("it rises 10 to 90 units a week") == ["10", "90"]


def test_a_number_inside_a_compact_json_list_is_a_figure_of_its_own():
    # Compact JSON read [360,365,370,375] as one thousands-separated number. Each list item must be matchable.
    hist = [{"kind": "tool", "result": {"orders": [360, 365, 370, 375]}}]
    assert unverified("The last forecast week is 375", hist, "forecast") == []
    assert unverified("The total is 1,205", [{"kind": "tool", "result": {"total": 1205}}], "g") == []
    assert unverified("The total is 1,204", [{"kind": "tool", "result": {"total": 1205}}], "g") == ["1,204"]


def test_a_percentage_written_with_a_space_matches_a_printed_fraction():
    from prophet_agent.runtime.grounding import unverified
    history = [{"kind": "tool", "tool": "backtest_series", "result": {"coverage": 0.82}}]
    assert unverified("The interval held in 82 % of windows and 82% of folds.", history, "") == []
    assert unverified("The interval held in 71 % of windows.", history, "") == ["71 %"]


def test_a_loss_written_with_a_currency_sign_matches_the_printed_negative_number():
    from prophet_agent.runtime.grounding import unverified
    history = [{"kind": "tool", "result": {"stdout": "net change -1683.5"}}]
    assert unverified("The run lost £1,683.5 in total.", history, "") == []
    assert unverified("The run lost −£1,683.5 in total.", history, "") == []
    assert unverified("The run lost £1,700.5 in total.", history, "") == ["1,700.5"]


def test_a_scaled_figure_is_a_rounding_of_a_printed_one():
    """Live retail: 27.9k, 35.1k and 28.9k were refused, because the tool printed 27912.4, 35108.0 and 28901.2."""
    history = [{"kind": "tool", "tool": "sandbox_python", "args": {}, "result": {"stdout": "mean 27912.4 high 35108.0 wkend 28901.2 total 1250000"}}]
    assert unverified("Weekday means run £27.9k to £35.1k and weekends £28.9k.", history, "") == []
    assert unverified("Total about 1.25m.", history, "") == []
    assert unverified("Weekday means run £27.8k.", history, "") == ["27.8k"]          # 27,800 is not a rounding of 27,912
    assert unverified("The cell is 5m wide.", [{"kind": "tool", "tool": "t", "args": {}, "result": {"stdout": "5"}}], "") == []
    assert unverified("It ran 5 min.", [{"kind": "tool", "tool": "t", "args": {}, "result": {"stdout": "5"}}], "") == []


def test_a_copying_slip_in_a_large_figure_is_corrected_to_the_printed_value():
    """Live retail round 15: 557,050 for the printed 557051.21 and 352,702 for 352602.3."""
    from prophet_agent.runtime.grounding import snap, snap_note
    history = [{"kind": "tool", "tool": "forecast_report", "args": {}, "result": {
        "centre_line_total": 557051.21, "q10_together": 352602.3, "rows": 10000, "q90": 843784.23}}]
    text = "Total **557,050**, range 352,702 to 843,784. We read 10,001 rows."
    fixed_text, fixed = snap(text, history, "")
    assert fixed_text == "Total **557,051**, range 352,602 to 843,784. We read 10,001 rows."
    assert fixed == [("557,050", "557,051"), ("352,702", "352,602")]
    assert unverified(fixed_text, history, "") == ["10,001"]                   # a whole-number count is never snapped
    assert "2 figure(s)" in snap_note(fixed) and snap_note([]) == ""
    # two printed values near the figure: no guess
    twin = [{"kind": "tool", "tool": "t", "args": {}, "result": {"a": 5000.4, "b": 5001.3}}]
    assert snap("It is 5,001.", twin, "")[1] == []
