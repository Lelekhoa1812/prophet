"""Round 9 repairs: each test names the panel mission and the audit finding it comes from."""
from __future__ import annotations

import json

import pytest

from prophet_agent.runtime import contracts, goal, grounding
from prophet_agent.tools.sports_tool import VIEW_BUDGET, observe_simulate


def _row(team: str, i: int) -> dict:
    return {"team": team, "current": float(20 - i), "expected_final": 46.377 - i, "expected_rank_in_group": 6.14 + i,
            "final": {"median": 46.0 - i, "p10": 41.0 - i, "p90": 52.0 - i}, "playoff": 0.733}


# NBA: the call named 15 teams; the view showed the first six by standing (three of them named) and the planner paged
# 25 times through the full result, then stated a Brier score no tool produced.
def test_simulation_view_shows_the_teams_the_call_named():
    teams = [f"Team {i}" for i in range(30)]
    rows = [_row(t, i) for i, t in enumerate(teams)]
    named = teams[10:24]
    paths = {t: [{"game": g, "measured": g <= 20, "median": g * 0.5, "p10": g * 0.4, "p90": g * 0.6} for g in range(1, 83)]
             for t in named}
    view = observe_simulate({"teams": rows, "points_paths": paths, "standings": rows, "ratings": rows})
    shown = [r["team"] for r in view["teams"]] + view.get("named_teams_not_shown", [])
    assert shown == named
    assert len(json.dumps(view, separators=(",", ":"))) <= VIEW_BUDGET
    assert view["teams"][0]["final"] == {"median": 36, "p10": 31, "p90": 42}       # whole numbers without a decimal point
    assert view["teams"][0]["expected_final"] == 36.4


def test_simulation_view_defaults_to_the_top_rows_when_every_team_has_a_path():
    rows = [_row(f"T{i}", i) for i in range(8)]
    paths = {r["team"]: [] for r in rows}
    view = observe_simulate({"teams": rows, "points_paths": paths})
    assert [r["team"] for r in view["teams"]] == [f"T{i}" for i in range(6)]


SCAN = {"findings": [
    {"id": "changepoints.level_shift", "severity": "major",
     "evidence": {"count": 4, "breaks": [{"at": "1979-01-01 00:00:00"}, {"at": "1990-10-01 00:00:00"},
                                         {"at": "2010-10-01 00:00:00"}, {"at": "2021-10-01 00:00:00"}]}},
    {"id": "outliers.share", "severity": "minor", "evidence": {"share_by_method": {"hampel": 0.031}, "severity_uses": "hampel"}},
    {"id": "values.missing", "severity": "minor", "evidence": {"missing_values": 65, "stamps": 1799}},
]}


def _saved_scan(tmp_path, scan=SCAN) -> list[dict]:
    run = "a" * 32
    (tmp_path / run / "out").mkdir(parents=True, exist_ok=True)
    (tmp_path / run / "out" / "scan.json").write_text(json.dumps(scan))
    return [{"kind": "tool", "tool": "sandbox_python", "result": {"run_id": run, "outputs": [{"path": "out/scan.json"}]}}]


# CPI, AUUNEMP, CO2 and SPX: the answer denied what the saved scan had flagged.
@pytest.mark.parametrize("answer, topic", [
    ("- No level shifts, outliers, or regime changes were flagged by the scan.", "level shifts"),
    ("- **Level shifts / regime changes**: None detected;", "level shifts"),
    ("Scan findings: no outliers, no missing values.", "outliers"),
    ("No gaps flagged in the series.", "missing values or gaps"),
])
def test_an_answer_may_not_deny_a_scan_finding(tmp_path, answer, topic):
    checks = contracts._scan_contradictions(answer, _saved_scan(tmp_path), tmp_path)
    bad = [c for c in checks if not c["ok"]]
    assert bad and bad[0]["severity"] == "block" and f"no {topic}" in bad[0]["detail"]


@pytest.mark.parametrize("answer", [
    "No gaps were filled; the 65 missing closes were left as holidays.",
    "Checks not run: no structural-break test beyond the scan's level shifts.",
    "No structural break is modelled.",
    "There were no major outliers.",                       # the outlier finding is minor
    "Four level shifts (1979-01, 1990-10, 2010-10 and 2021-10) were kept as observed.",
    "No outliers needing treatment.",
])
def test_statements_that_do_not_deny_a_finding_pass(tmp_path, answer):
    checks = contracts._scan_contradictions(answer, _saved_scan(tmp_path), tmp_path)
    assert all(c["ok"] for c in checks), checks


# NGSTORAGE "one level shift in early 2012" against breaks from 2013-12; SPX "level shift (2020-03-23)" against breaks from 2021.
@pytest.mark.parametrize("answer, year", [("One level shift in early 2000 was kept.", 2000),
                                          ("one level shift (2020-03-23, COVID crash)", 2020)])
def test_a_shift_date_must_match_a_scan_break(tmp_path, answer, year):
    bad = [c for c in contracts._scan_contradictions(answer, _saved_scan(tmp_path), tmp_path) if not c["ok"]]
    assert bad and bad[0]["name"] == "answer_shift_dates_match" and str(year) in bad[0]["detail"]


def test_a_shift_date_near_a_break_passes(tmp_path):
    for answer in ("A level shift in late 1978 lifted the level.", "the break at 2010-09-20 is kept",
                   "level shifts since 1950 were all kept"):
        assert all(c["ok"] for c in contracts._scan_contradictions(answer, _saved_scan(tmp_path), tmp_path)), answer


# GISTEMP "7 level shifts" against 4: a warning did not stop it.
def test_a_wrong_shift_count_blocks_for_one_series(tmp_path):
    checks = contracts.check_answer("The scan found 7 level shifts.", _saved_scan(tmp_path), run_root=tmp_path)
    bad = [c for c in checks if c["name"] == "answer_shift_count_matches"]
    assert bad and not bad[0]["ok"] and bad[0]["severity"] == "block"


# EPL: expected clubs from memory (last season's) made the claim, the planner and the evaluator reject this season's files.
def test_expected_members_from_memory_are_left_out():
    args = {"series_name": "epl", "expected_entities": ["Arsenal", "Burnley", "Leicester City"], "table_kind": "league"}
    out, unsourced = goal._drop_unsourced_entities(args, [], "Forecast the Premier League; Arsenal is one club")
    assert "expected_entities" not in out and out["expected_rows"] == 3 and unsourced == ["Burnley", "Leicester City"]
    hist = [{"kind": "tool", "tool": "download_web_data",
             "result": {"first_rows": [["1", "Arsenal", "Coventry"], ["2", "Hull", "Man Utd"]]}}]
    kept, none = goal._drop_unsourced_entities({"expected_entities": ["Arsenal", "Hull", "Coventry"]}, hist, "")
    assert kept["expected_entities"] == ["Arsenal", "Hull", "Coventry"] and none == []


def test_denials_and_dates_are_read_from_prose():
    assert ("outliers", "No level shifts, outliers or missing values were flagged.", False) in \
        grounding.denied_findings("No level shifts, outliers or missing values were flagged.")
    assert grounding.claimed_shift_dates("No level shift in 2012.") == []


# NEM: the 5-minute month file ended at 00:00 on 27 November, so the last daily mean was one reading and the forecast repeated it.
def test_a_partial_last_period_is_left_out(capsys):
    import numpy as np
    import pandas as pd

    from prophet_agent.sandbox import prophet_lib as pl
    t = pd.date_range("2026-11-01 00:05", "2026-11-27 00:00", freq="5min")
    df = pd.DataFrame({"t": t.astype(str), "v": np.arange(len(t)).astype(float)})
    s = pl.build_series(df, "v", "t", freq="D", agg="mean")
    assert str(s.index[-1].date()) == "2026-11-26" and len(s) == 26       # 1 November holds 287 of 288 rows: kept
    assert "partial edge periods: 2026-11-27" in capsys.readouterr().out
    closes = pl.build_series(df, "v", "t", freq="D", agg="last")
    assert str(closes.index[-1].date()) == "2026-11-27"                    # the last close of a day is still that close
    daily = pd.DataFrame({"t": pd.date_range("2026-11-01", "2026-11-26").astype(str), "v": 1.0})
    assert len(pl.build_series(daily, "v", "t", freq="D", agg="mean")) == 26


def _fc(date: str, value: float) -> list[dict]:
    return [{"kind": "tool", "tool": "timesfm_forecast", "result": {"point": [1.0], "last_observed": {"date": date,
                                                                                                    "value": value}}}]


# AUUNEMP: "Latest observed value: 4.1% (October 2026)" where October was 4.86 and 4.1 was January's.
def test_the_latest_value_is_the_last_observed_value():
    bad = contracts._latest_and_end("- Latest observed value: **4.1%** (October 2026).", _fc("2026-10-01", 4.864833))
    assert bad and bad[0]["name"] == "answer_latest_value" and bad[0]["severity"] == "block"
    for ok in ("- Latest observed value: **4.86%** (October 2026).", "The latest value is 4.9%.",
               "| Naive (last value) | 0.442 | baseline |", "The ABS release reported a latest value of 5.1%."):
        assert contracts._latest_and_end(ok, _fc("2026-10-01", 4.864833)) == [], ok


# SPX: "1,799 trading days, 2020-01-02 to 2026-11-27" with the last close on 2026-11-24.
def test_the_data_span_may_not_end_after_the_last_observation():
    said = "- Source: FRED series SP500 (1,799 trading days, 2020-01-02 to 2026-11-27)."
    bad = contracts._latest_and_end(said, _fc("2026-11-24", 7201.09))
    assert bad and bad[0]["name"] == "answer_data_end" and "2026-11-24" in bad[0]["detail"]
    for ok in ("- Source: FRED SP500, 2020-01-02 to 2026-11-24.", "The forecast window runs 2026-11-25 to 2026-12-22."):
        assert contracts._latest_and_end(ok, _fc("2026-11-24", 7201.09)) == [], ok


def test_a_forecast_records_where_the_observations_stop():
    from prophet_agent.tools.timesfm_tool import _last_observed
    assert _last_observed([1.0, 2.0, float("nan")], ["2026-11-01", "2026-11-02", "2026-11-03"]) == \
        {"value": 2.0, "date": "2026-11-02"}
    assert _last_observed([1.0, 3.5], None) == {"value": 3.5}
