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
    table = view["teams"]
    shown = [r[0] for r in table["rows"]] + view.get("named_teams_not_shown", [])
    assert shown == named
    assert len(json.dumps(view, separators=(",", ":"))) <= VIEW_BUDGET
    first = dict(zip(table["columns"], table["rows"][0]))
    assert (first["p10"], first["median"], first["p90"]) == (31, 36, 42)          # whole numbers without a decimal point
    assert first["expected_final"] == 36.4 and "expected_rank_in_group" not in table["columns"]


def test_simulation_view_defaults_to_the_top_rows_when_every_team_has_a_path():
    rows = [_row(f"T{i}", i) for i in range(8)]
    paths = {r["team"]: [] for r in rows}
    view = observe_simulate({"teams": rows, "points_paths": paths})
    assert [r[0] for r in view["teams"]["rows"]] == [f"T{i}" for i in range(6)]


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


# Round 10 (SPX, CO2): an answer that listed the five shifts and then wrote "no regime change that would break the seasonal
# pattern" was blocked; a phrase about regimes does not deny findings the answer reports. AUUNEMP's denial still blocks.
def test_a_reported_finding_is_not_denied_by_a_later_phrase(tmp_path):
    said = ("The scan flagged 4 level shifts (1979-01-01, 1990-10-01, 2010-10-01, 2021-10-01); all were kept with no regime "
            "change detected that would break the seasonal pattern.")
    assert all(c["ok"] for c in contracts._scan_contradictions(said, _saved_scan(tmp_path), tmp_path))
    denied = "Scan flagged an outlier share of 3.1%; no missing months, no level shifts detected."
    bad = [c for c in contracts._scan_contradictions(denied, _saved_scan(tmp_path), tmp_path) if not c["ok"]]
    assert bad and "no level shifts" in bad[0]["detail"]
    assert not grounding.acknowledges("- **Level shifts / regime changes**: None detected;", "level_shift")


# Round 10 (EPL): six questions were refused for a question mark mid-text or "Top 4" against "Top 6", and four more for
# lacking "high-risk:" although the request said "Ask me which clubs to keep".
def test_club_questions_reach_the_person():
    from prophet_agent.runtime.questions import repair_questions, validate_for_goal
    q = {"header": "Clubs", "multiSelect": True, "question": "Which clubs should I keep? The table has 20 teams.",
         "options": [{"label": "Top 4 by current points", "description": "a"},
                     {"label": "Top 6 by current points", "description": "b"}]}
    fixed, notes = repair_questions([q])
    assert fixed[0]["question"] == "The table has 20 teams. Which clubs should I keep?" and notes
    assert validate_for_goal({"questions": fixed}, "") == []
    assert goal.ASKED_TO_ASK.search("Ask me which clubs to keep if the table has many rows.")


# Round 10 (EPL): a 22-row standings table typed from memory into validate_table.
def test_a_typed_table_must_trace_to_fetched_data():
    hist = [{"kind": "tool", "tool": "download_web_data",
             "result": {"first_rows": [["1", "Arsenal", "Coventry", "3 - 0"], ["2", "Hull", "Man Utd", "2 - 0"]]}}]
    typed = "Team,Pts\nArsenal,3\nLeicester,9\n"
    why = goal._typed_table_refusal("validate_table", {"csv": typed}, hist, "")
    assert why and "Leicester" in why
    assert goal._typed_table_refusal("validate_table", {"csv": "Team,Pts\nArsenal,3\nHull,3\n"}, hist, "") is None
    assert goal._typed_table_refusal("validate_table", {"dataset_id": "x"}, hist, "") is None


# Round 10 (NGSTORAGE): pd.read_csv on a file with title lines failed four times and the run stalled.
def test_a_preamble_parse_error_points_to_pl_load():
    from prophet_agent.tools.sandbox_tool import observe
    out = observe({"ok": False, "run_id": "r", "stdout": "", "outputs": [], "timed_out": False,
                   "stderr": "pandas.errors.ParserError: Error tokenizing data. C error: Expected 1 fields in line 5, saw 14\n"})
    assert "pl.load(path)" in out["hint"]


# Round 10 (NVDA): "closed at $148.82 on 2026-11-25", where the close that day was 149.30 (148.82 was a four-day mean).
def test_a_value_stated_for_the_last_day_is_that_days_value():
    bad = contracts._latest_and_end("Nvidia (NVDA) closed at $148.82 on 2026-11-25.", _fc("2026-11-25", 149.30))
    assert bad and bad[0]["name"] == "answer_latest_value"
    assert contracts._latest_and_end("Nvidia (NVDA) closed at $149.30 on 2026-11-25.", _fc("2026-11-25", 149.30)) == []
    assert contracts._latest_and_end("The mean of the last 4 closes on 2026-11-25 was $148.82.", _fc("2026-11-25", 149.3)) == []


# Round 10 (CPI): outside figures cited to philadelphiafed.org and fred.stlouisfed.org that neither fetched text holds.
def test_a_cited_source_must_hold_the_figure():
    hist = [{"kind": "tool", "tool": "web_research", "result": {"sources": [
        {"id": "S1", "final_url": "https://www.bls.gov/schedule/cpi.htm", "passages": [{"text": "Mar. 10, 2027 08:30 AM"}]},
        {"id": "S2", "final_url": "https://fred.stlouisfed.org/series/EXPINF27YR", "passages": [{"text": "Notes only."}]}]}},
        {"kind": "tool", "tool": "sandbox_python", "result": {"stdout": "2.3 2.5 2.7"}}]
    said = ("- **Cleveland Fed expected inflation (fred.stlouisfed.org, 10 Nov 2026):** long-run expectation at 2.3%.\n"
            "- **BLS schedule (bls.gov, undated):** the February CPI is released on 10 Mar 2027 at 08:30.\n"
            "- All sources (bls.gov and fred.stlouisfed.org) were read; 7 passages were kept.")
    assert grounding.source_bound_mismatches(said, hist) == ["2.3%"]


# Round 10 (CPI): an outside figure cited to one site but printed only by another is left out.
def test_an_outside_line_is_checked_against_the_site_it_cites(tmp_path):
    from prophet_agent.runtime.goal import _check_outside
    from prophet_agent.runtime.store import Store

    class C:
        pass
    store = Store(tmp_path / "s.sqlite")
    mid = store.create_mission("goal", {})
    c = C(); c.store, c.mission_id = store, mid
    store.tool_start(mid, 0, "web_research", {"queries": ["x"]}, "k0")
    store.tool_set("k0", "done", {"sources": [
        {"final_url": "https://fred.stlouisfed.org/graph/fredgraph.csv?id=CPIAUCSL", "passages": [{"text": "2026-01-01,2.5\n2.7"}]},
        {"final_url": "https://www.philadelphiafed.org/surveys", "passages": [{"text": "The survey is published quarterly."}]},
        {"final_url": "https://www.bls.gov/schedule/cpi.htm", "passages": [{"text": "Mar. 10, 2027 08:30 AM"}]}]})
    answer = ("Flat.\n\n## Outside context\n- **SPF (philadelphiafed.org, latest Q4 2026):** \"The survey is published "
              "quarterly\"; a pace near 2.5–2.7%.\n"
              "- **BLS (bls.gov, undated):** \"Mar. 10, 2027 08:30 AM\" for the February CPI.\n\n## Files\n- out/forecast.png")
    out, removed = _check_outside(c, answer)
    assert "2.5" in removed and "philadelphiafed" not in out and "bls.gov" in out and "1 outside claim(s)" in out


# Round 10 (GISTEMP): "ENSO-neutral conditions expected" where the cited CPC passage says a strong El Nino is likely.
def test_an_outside_line_needs_a_quote_from_its_site(tmp_path):
    from prophet_agent.runtime.goal import _check_outside
    from prophet_agent.runtime.store import Store

    class C:
        pass
    store = Store(tmp_path / "s.sqlite")
    mid = store.create_mission("goal", {})
    c = C(); c.store, c.mission_id = store, mid
    store.tool_start(mid, 0, "web_fetch", {"url": "x"}, "k0")
    store.tool_set("k0", "done", {"final_url": "https://www.cpc.ncep.noaa.gov/enso.html",
                                  "text": "El Niño continues to strengthen, with a strong-to-very strong El Niño likely."})
    answer = ("Flat.\n\n## Outside context\n"
              "- **CPC (cpc.ncep.noaa.gov, 12 Nov 2026):** ENSO-neutral conditions expected through early 2027.\n"
              "- **CPC (cpc.ncep.noaa.gov, 12 Nov 2026):** \"ENSO-neutral conditions expected\".\n"
              "- **CPC (cpc.ncep.noaa.gov, 12 Nov 2026):** \"a strong-to-very strong El Nino likely\", so warm months.\n"
              "\n## Files\n- out/forecast.png")
    out, removed = _check_outside(c, answer)
    assert out.count("cpc.ncep.noaa.gov") == 1 and "strong-to-very strong" in out and "2 outside claim(s)" in out
