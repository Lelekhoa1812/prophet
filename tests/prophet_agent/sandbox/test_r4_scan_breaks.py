"""Round 4 finding A4-09: a level shift on a calendar boundary, or shared by several entities on the same date, is a
recording hypothesis to test, not a regime.

Motivation vs Logic: the live Beijing run kept shifts on 1 January for three stations and a step shared by all of them,
and called them real regimes with no test. The scan now marks each break with ``on_calendar_boundary`` (midnight on 1
January or on the first of a month) and ``coincident_entities`` (other entities with a break on the same date). A break
with either flag is a ``recording_hypothesis`` and the finding's decision says to test it before keeping it. Severity is
unchanged: the flag asks for a test, it does not block.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from prophet_agent.sandbox import prophet_scan as sc

START = "2011-07-01"  # 184 days before 1 January 2012
N = 500


def _pattern(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(N)
    return 50 * (1 + 0.3 * np.sin(2 * np.pi * t / 7)) * (1 + 0.05 * rng.standard_normal(N))


def _panel(steps: dict, factor: float = 1.8) -> pd.DataFrame:
    """One daily series per entity. A non-None date is where that entity's level is multiplied by ``factor``."""
    idx = pd.date_range(START, periods=N, freq="D")
    frames = []
    for i, (name, when) in enumerate(steps.items()):
        vals = _pattern(300 + i)
        if when is not None:
            vals[idx.get_loc(pd.Timestamp(when)):] *= factor
        frames.append(pd.DataFrame({"sku": name, "date": idx, "units": vals}))
    return pd.concat(frames, ignore_index=True)


def _shift_rows(result: dict, scope: str) -> list:
    found = [f for f in result["findings"] if f["id"] == "changepoints.level_shift" and f["scope"] == scope]
    assert found, f"no level shift finding for {scope}"
    return found[0]["evidence"]["breaks"]


def test_a_common_new_year_step_is_a_recording_hypothesis_for_every_entity():
    r = sc.scan(_panel({"A": "2012-01-01", "B": "2012-01-01", "C": "2012-01-01"}), "date", "units", entity_cols="sku")
    for scope, others in (("A", ["B", "C"]), ("B", ["A", "C"]), ("C", ["A", "B"])):
        row = _shift_rows(r, scope)[0]
        assert row["at"].startswith("2012-01-01")
        assert row["on_calendar_boundary"] is True
        assert row["coincident_entities"] == others
        assert row["recording_hypothesis"] is True


def test_the_finding_says_to_test_a_recording_hypothesis_before_keeping_the_shift():
    r = sc.scan(_panel({"A": "2012-01-01", "B": "2012-01-01", "C": "2012-01-01"}), "date", "units", entity_cols="sku")
    found = [f for f in r["findings"] if f["id"] == "changepoints.level_shift" and f["scope"] == "A"][0]
    assert found["evidence"]["recording_hypothesis_breaks"] == 1
    assert "recording hypothesis" in found["decision"].lower()
    assert "truncate" in found["decision"].lower() and "flag" in found["decision"].lower()


def test_a_mid_year_shift_on_one_entity_alone_is_not_a_recording_hypothesis():
    r = sc.scan(_panel({"D": "2012-03-15", "E": None}), "date", "units", entity_cols="sku")
    row = _shift_rows(r, "D")[0]
    assert row["at"].startswith("2012-03-15")
    assert row["on_calendar_boundary"] is False
    assert row["coincident_entities"] == []
    assert row["recording_hypothesis"] is False
    found = [f for f in r["findings"] if f["id"] == "changepoints.level_shift" and f["scope"] == "D"][0]
    assert found["evidence"].get("recording_hypothesis_breaks", 0) == 0
    assert "recording hypothesis" not in found["decision"].lower()


def test_a_mid_year_shift_shared_by_two_entities_is_flagged_with_the_other_one_named():
    r = sc.scan(_panel({"F": "2012-06-10", "G": "2012-06-10", "H": None}), "date", "units", entity_cols="sku")
    row = _shift_rows(r, "F")[0]
    assert row["on_calendar_boundary"] is False
    assert row["coincident_entities"] == ["G"]
    assert row["recording_hypothesis"] is True
    assert _shift_rows(r, "G")[0]["coincident_entities"] == ["F"]
