"""Round 3 calendar finding (A1-06): the scan names the weekdays a series never trades on.

Motivation vs Logic: the live retail scan had no calendar finding. 135 weekday periods were missing and the pooled
weekend share (0.52) sat under the business-day threshold, so rep4 built a Monday-to-Friday series and dropped 99
trading Sundays. Saturday alone has no record on 0.99 of its periods (measured-raw, scan of the example CSV).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from prophet_agent.sandbox import prophet_scan as sc

REPO = Path(__file__).resolve().parents[3]
RETAIL = REPO / "examples" / "sales" / "uci_online_retail_ii_daily_sales_2009_2011.csv"


def test_scan_names_closed_saturday():
    df = pd.read_csv(RETAIL)
    r = sc.scan(df, "date", "net_sales_gbp")
    found = [f for f in r["findings"] if f["id"] == "calendar.closed_days"]
    assert len(found) == 1
    assert found[0]["severity"] == "major"
    assert found[0]["evidence"]["closed"] == ["Saturday"]
    assert "Saturday" in found[0]["evidence"]["trading_weekdays"] or "Sunday" in found[0]["evidence"]["trading_weekdays"]
    assert found[0]["evidence"]["share_by_weekday"]["Saturday"] >= 0.9
    assert "closed=" in found[0]["next_step"]                  # the fix names the argument, not a business-day grid


def test_a_business_day_series_gets_no_major_closed_days_finding():
    # Synthetic weekday-only series: the weekend closure is already the business-day finding.
    idx = pd.bdate_range("2023-01-02", periods=300)
    df = pd.DataFrame({"d": idx, "v": 100 + np.sin(np.arange(300) / 5)})
    r = sc.scan(df, "d", "v")
    assert not [f for f in r["findings"] if f["id"] == "calendar.closed_days" and f["severity"] == "major"]
