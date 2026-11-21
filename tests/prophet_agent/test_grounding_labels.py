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
