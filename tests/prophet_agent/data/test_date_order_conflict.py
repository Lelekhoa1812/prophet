"""A date column that mixes day-first and month-first values is reported, never silently read as one order. Regression for
the malformed-dates pressure run (mission c7490cda): 13 rows only fit day-first and 13 only fit month-first; the planner
forced day-first, 13 month-first rows were dropped unnamed, and the timeline ran to December."""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src" / "prophet_agent" / "sandbox"))
import prophet_lib as pl  # noqa: E402


MIXED = pd.Series(["16/01/2025", "01/02/2025", "21/01/2025", "04/22/2025", "01/07/2025", "26/01/2025"])


def test_order_evidence_counts_values_that_fit_only_one_order():
    assert pl.order_evidence(MIXED) == (3, 1)
    assert pl.order_evidence(pd.Series(["03/04/2025", "05/06/2025"])) == (0, 0), "both fit: no evidence either way"


def test_a_mixed_column_reports_the_conflict_with_counts():
    got, info = pl.to_datetime(MIXED)
    assert info["order_conflict"] == {"day_first_only_rows": 3, "month_first_only_rows": 1}


def test_a_stated_order_that_contradicts_rows_says_how_many():
    got, info = pl.to_datetime(MIXED, dayfirst=True)
    assert info["rows_contradicting_stated_order"] == 1, "04/22/2025 only fits month-first and is not read"
    assert pd.isna(got.iloc[3])


def test_a_consistent_day_first_column_has_no_conflict():
    got, info = pl.to_datetime(pd.Series(["16/01/2025", "21/01/2025", "05/02/2025"]))
    assert "order_conflict" not in info and info["format"] == "day first"


def test_a_consistent_month_first_column_has_no_conflict():
    got, info = pl.to_datetime(pd.Series(["01/16/2025", "01/21/2025", "02/05/2025"]))
    assert "order_conflict" not in info and info["format"] == "month first"


def test_profile_raises_the_conflict_as_an_issue_and_keeps_the_column_as_dates(tmp_path):
    rows = ["when,value"]
    for i in range(30):
        rows.append(f"{MIXED.iloc[i % 6]},{i}")
    path = tmp_path / "mixed.csv"
    path.write_text("\n".join(rows) + "\n")
    prof = pl.profile(path)
    col = next(c for c in prof["columns"] if c["name"] == "when")
    assert col["kind"] == "time", "a mostly-parseable mixed column is a date column, not text"
    assert any("mixes both orders" in issue and "Ask which order" in issue for issue in prof["issues"])


def test_profile_of_a_consistent_column_raises_no_order_issue(tmp_path):
    path = tmp_path / "clean.csv"
    path.write_text("day,value\n" + "\n".join(f"{d:02d}/01/2025,{d}" for d in range(13, 30)) + "\n")
    prof = pl.profile(path)
    assert not any("mixes both orders" in issue for issue in prof["issues"])
