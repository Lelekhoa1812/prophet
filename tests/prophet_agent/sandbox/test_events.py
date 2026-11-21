"""Event, holiday and anomaly helpers in the analysis library, and covariates saved beside a series."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from prophet_agent.sandbox import prophet_lib as pl


def test_event_covariate_daily_is_flags_and_runs_into_the_future():
    idx = pd.date_range("2025-01-01", periods=60, freq="D")
    c = pl.event_covariate(idx, [("2025-01-10", "2025-01-12"), ("2025-03-05", "2025-03-05")], horizon=10)
    assert len(c) == 70
    assert c[9:12].tolist() == [1, 1, 1] and c[:9].sum() == 0
    assert c[63] == 1 and c.sum() == 4                        # 2025-03-05 is index 63, inside the horizon


def test_event_covariate_weekly_is_a_share_of_the_period():
    idx = pd.date_range("2025-01-06", periods=20, freq="W-MON")
    c = pl.event_covariate(idx, [("2025-02-04", "2025-02-04")], horizon=4)
    assert len(c) == 24
    assert c.max() == pytest.approx(1 / 7) and np.count_nonzero(c) == 1


def test_holiday_covariate_matches_the_calendar():
    idx = pd.date_range("2025-12-01", periods=40, freq="D")
    c = pl.holiday_covariate(idx, "AU", "VIC", horizon=10)
    days = {h["date"] for h in pl.list_holidays("AU", "VIC", "2025-12-01", "2026-01-19")}
    assert {"2025-12-25", "2025-12-26", "2026-01-01"} <= days
    assert c[24] == 1 and c[25] == 1 and c[31] == 1          # Dec 25, Dec 26, Jan 1
    assert c.sum() == len(days)


def test_detect_anomalies_finds_injected_errors_and_leaves_clean_data_alone():
    rng = np.random.default_rng(3)
    t = np.arange(300)
    y = 100 + 0.2 * t + 20 * np.sin(2 * np.pi * t / 7) + rng.normal(0, 3, 300)
    clean = pl.detect_anomalies(y, season=7)
    assert clean["ignore_ranges"] == []
    y[[3, 100, 205, 206]] *= 10
    for season in (7, None):
        r = pl.detect_anomalies(y, season=season)
        assert r["ignore_ranges"] == [[3, 4], [100, 101], [205, 207]]
    assert pl.detect_anomalies([1.0, 2.0], season=None)["ignore_ranges"] == []


def test_save_series_keeps_covariates_and_refuses_missing(tmp_path):
    path = str(tmp_path / "s.json")
    pl.save_series([1.0, 2.0, 3.0], path, covariates={"promo": [0, 1, 0, 0]})
    doc = json.loads((tmp_path / "s.json").read_text())
    assert doc["covariates"] == {"promo": [0.0, 1.0, 0.0, 0.0]} and doc["series"] == [1.0, 2.0, 3.0]
    with pytest.raises(ValueError, match="covariate 'p'"):
        pl.save_series([1.0, 2.0], path, covariates={"p": [1.0, float("nan")]})


def _csv(tmp_path, values, dates=None, name="s.csv"):
    idx = pd.date_range("2024-01-01", periods=len(values), freq="D") if dates is None else dates
    path = tmp_path / name
    pd.DataFrame({"Date": idx.strftime("%Y-%m-%d"), "Units": values}).to_csv(path, index=False)
    return path


def test_profile_reports_entry_error_candidates_as_a_fact(tmp_path):
    rng = np.random.default_rng(5)
    t = np.arange(300)
    y = np.round(100 + 20 * np.sin(2 * np.pi * t / 7) + rng.normal(0, 3, 300))
    y[[40, 200]] *= 10
    r = pl.profile(_csv(tmp_path, y))
    col = next(c for c in r["columns"] if c["name"] == "Units")
    assert col["outlier_candidates"]["count"] == 2
    assert col["outlier_candidates"]["ignore_ranges_if_confirmed"] == [[40, 41], [200, 201]]
    assert {p["row"] for p in col["outlier_candidates"]["largest"]} == {41, 201}      # 1-based table rows
    assert any("far from their neighbours" in i and "real events" in i for i in r["issues"])


def test_profile_stays_quiet_on_clean_and_on_spiky_series(tmp_path):
    rng = np.random.default_rng(6)
    clean = np.round(100 + rng.normal(0, 3, 200))
    assert not any("neighbours" in i for i in pl.profile(_csv(tmp_path, clean))["issues"])
    lumpy = np.where(rng.random(200) < 0.3, rng.integers(50, 500, 200), 0).astype(float)      # intermittent demand
    assert not any("neighbours" in i for i in pl.profile(_csv(tmp_path, lumpy, name="l.csv"))["issues"])


def test_profile_does_not_offer_positions_that_could_be_wrong_after_dropping_gaps(tmp_path):
    rng = np.random.default_rng(7)
    y = np.round(100 + rng.normal(0, 3, 120)).astype(object)
    y[10] = ""                      # a blank cell shifts positions once it is dropped
    y[80] = 5000
    col = next(c for c in pl.profile(_csv(tmp_path, y))["columns"] if c["name"] == "Units")
    assert col["outlier_candidates"]["count"] == 1 and "ignore_ranges_if_confirmed" not in col["outlier_candidates"]
