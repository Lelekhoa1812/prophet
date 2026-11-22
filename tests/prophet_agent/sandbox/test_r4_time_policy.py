"""Round 4 finding A7-10: the person can state the time rule for a source, so a clock-change blocker has a decision path.

Motivation vs Logic: the scan rates a daylight-saving repeated hour as a major ``time.dst_shift`` and the contract blocks
until a decision is recorded. Nothing let the person state the rule (timezone and what to do with the repeated hour), so
the only resolver was the planner. ``prophet_lib.time_policy`` applies the rule to the loaded frame before the scan.
Scope of this slice: the transform and its validation. Storing the setting on the source and writing the ledger decision
are in the notebook store and the service, outside this slice (see the round 4 report).

A missing hour (spring forward) is not filled: the policy resolves repeated hours only, and the scan still reports the
missing hour. The fixture therefore covers an autumn fall-back window, which has repeated hours and no missing ones.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from prophet_agent.sandbox import prophet_lib as pl
from prophet_agent.sandbox import prophet_scan as sc

ZONE = "Australia/Sydney"


def _sydney_hourly() -> pd.DataFrame:
    """Hourly wall-clock stamps for a window that contains the 7 April 2024 fall-back (02:00 happens twice)."""
    utc = pd.date_range("2024-03-20 00:00", "2024-04-15 23:00", freq="h", tz="UTC")
    wall = utc.tz_convert(ZONE).tz_localize(None)
    vals = 100 + 10 * np.sin(np.arange(len(wall)) / 24.0)
    return pd.DataFrame({"when": wall, "kwh": vals})


def _dst_findings(df: pd.DataFrame) -> list:
    r = sc.scan(df, "when", "kwh", freq="h")
    return [f for f in r["findings"] if f["id"] == "time.dst_shift"]


def test_without_a_policy_the_scan_flags_the_repeated_hour_as_a_major_dst_shift():
    found = _dst_findings(_sydney_hourly())
    assert found and found[0]["severity"] == "major"
    assert found[0]["evidence"]["repeated_hour_days"] == 1


def test_the_sum_policy_sums_the_repeated_hour_and_the_scan_no_longer_flags_it():
    raw = _sydney_hourly()
    fixed, report = pl.time_policy(raw, "when", "kwh", timezone=ZONE, dst="sum_repeated_hour")
    assert _dst_findings(fixed) == []
    assert report["ambiguous_wall_times"] == 1
    assert report["rows_removed"] == 1
    twice = raw[raw["when"] == pd.Timestamp("2024-04-07 02:00")]["kwh"].sum()
    assert fixed.loc[fixed["when"] == pd.Timestamp("2024-04-07 02:00"), "kwh"].iloc[0] == pytest.approx(twice)


def test_the_drop_policy_keeps_the_first_occurrence_and_removes_the_second():
    raw = _sydney_hourly()
    first_value = raw["kwh"].iloc[(raw["when"] == pd.Timestamp("2024-04-07 02:00")).to_numpy().argmax()]
    fixed, report = pl.time_policy(raw, "when", "kwh", timezone=ZONE, dst="drop_repeated")
    assert len(fixed) == len(raw) - 1
    assert report["rows_removed"] == 1
    assert fixed.loc[fixed["when"] == pd.Timestamp("2024-04-07 02:00"), "kwh"].iloc[0] == pytest.approx(first_value)


def test_keep_as_is_changes_nothing():
    raw = _sydney_hourly()
    fixed, report = pl.time_policy(raw, "when", "kwh", timezone=ZONE, dst="keep_as_is")
    assert len(fixed) == len(raw)
    assert report["rows_removed"] == 0


def test_a_duplicate_stamp_outside_the_repeated_hour_is_not_summed():
    raw = _sydney_hourly()
    dup = pd.concat([raw, raw.iloc[[10]]], ignore_index=True)     # a genuine double entry in an ordinary hour
    fixed, report = pl.time_policy(dup, "when", "kwh", timezone=ZONE, dst="sum_repeated_hour")
    assert report["rows_removed"] == 1                           # only the fall-back's second 02:00 goes
    assert (fixed["when"] == raw["when"].iloc[10]).sum() == 2   # both rows kept: the ordinary hour is not ambiguous


def test_an_aware_column_is_read_in_the_given_zone_before_the_policy():
    utc = pd.date_range("2024-04-06 12:00", periods=4, freq="h", tz="UTC")
    df = pd.DataFrame({"when": utc, "kwh": [1.0, 2.0, 3.0, 4.0]})
    fixed, _ = pl.time_policy(df, "when", "kwh", timezone=ZONE, dst="keep_as_is")
    assert str(fixed["when"].iloc[0]) == "2024-04-06 23:00:00"   # 12:00 UTC is 23:00 in Sydney (AEDT, +11) that day
    assert fixed["when"].dt.tz is None


@pytest.mark.parametrize("kwargs", [
    {"dst": "round_it"},
    {"timezone": "Mars/Olympus"},
    {"grain": "bogus"},
])
def test_an_unknown_policy_zone_or_grain_is_refused(kwargs):
    base = {"dst": "keep_as_is"}
    base.update(kwargs)
    with pytest.raises(ValueError):
        pl.time_policy(_sydney_hourly(), "when", "kwh", **base)


def test_the_grain_is_reported_so_the_scan_can_be_given_the_same_frequency():
    _, report = pl.time_policy(_sydney_hourly(), "when", "kwh", grain="h")
    assert report["grain"] == "h"
