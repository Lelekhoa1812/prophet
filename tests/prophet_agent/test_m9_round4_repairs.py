"""Panel round 4 repairs (2026-11-27): each test names the mission whose failure it pins."""
from __future__ import annotations

import json
import os

import pytest

from prophet_agent.forecast.sports import parse_matches, placeholder_rows
from prophet_agent.opendata import sources
from prophet_agent.runtime import goal
from prophet_agent.runtime.grounding import unverified
from prophet_agent.sandbox import prophet_stats as ps
from prophet_agent.tools import series_io


# EPL: the planner sent its question as a tool call named ask_user and got "unknown tool" three times.
def test_ask_user_tool_call_becomes_the_question_action():
    qs = [{"id": "contenders", "text": "Which clubs?", "type": "multi_select", "options": ["A", "B"]}]
    plan = {"action": "call_tool", "tool": "ask_user", "args": {"questions": qs}, "rationale": "high-risk: x" * 3}
    out = goal.as_question_action(plan)
    assert out["action"] == "ask_user" and out["questions"] == qs and out["rationale"] == plan["rationale"]
    assert "tool" not in out and "args" not in out


def test_other_tool_calls_are_left_alone():
    plan = {"action": "call_tool", "tool": "sandbox_python", "args": {"code": "print(1)"}, "rationale": "r"}
    assert goal.as_question_action(plan) is plan


# EPL and NBA: "95% CI" and "95% bootstrap interval" were flagged as unverified figures.
@pytest.mark.parametrize("phrase", ["the 95% CI crossed zero", "a 95% bootstrap interval that crossed zero",
                                    "the 90% confidence interval", "an 80% PI"])
def test_interval_level_names_are_labels(phrase):
    assert unverified(f"The model differs from the baseline: {phrase}.", [], "forecast") == []


def test_a_bare_percentage_is_still_checked():
    assert unverified("Arsenal win the title with 95% probability.", [], "forecast") == ["95%"]


# NBA: the fixturedownload file holds cup knockout rows "To be announced" on both sides; the simulation failed on them.
NBA = ("Match Number,Round Number,Date,Location,Home Team,Away Team,Result\n"
       "1,1,01/11/2026 00:00,A,Boston Celtics,New York Knicks,121 - 110\n"
       "2,1,02/11/2026 00:00,B,New York Knicks,Boston Celtics,\n"
       "3,7,04/12/2026 05:00,TBA,To be announced,To be announced,\n")


def test_placeholder_rows_are_left_out_and_counted():
    matches = parse_matches(NBA, "nba.csv")
    assert [(m.home, m.away, m.played) for m in matches] == [("Boston Celtics", "New York Knicks", True),
                                                             ("New York Knicks", "Boston Celtics", False)]
    assert placeholder_rows(NBA, "nba.csv") == 1


def test_a_played_placeholder_row_is_an_error():
    text = NBA.replace("To be announced,To be announced,", "To be announced,Boston Celtics,100 - 90")
    with pytest.raises(ValueError, match="needs both team names"):
        parse_matches(text, "nba.csv")


def test_a_whole_score_written_as_a_float_is_read():
    text = "home,away,home_goals,away_goals\nA,B,121.0,110.0\n"
    assert (parse_matches(text, "x.csv")[0].home_goals, parse_matches(text, "x.csv")[0].away_goals) == (121, 110)
    with pytest.raises(ValueError, match="whole number"):
        parse_matches("home,away,home_goals,away_goals\nA,B,1.5,2\n", "x.csv")


# NEM: aemo.com.au pages answer with a bot check; the monthly price and demand file is reachable.
def test_aemo_price_demand_url():
    url = sources.build_url("aemo_price_demand", {"month": "2026-10", "region": "NSW1"})
    assert url == "https://www.aemo.com.au/aemo/data/nem/priceanddemand/PRICE_AND_DEMAND_202610_NSW1.csv"
    with pytest.raises(ValueError):
        sources.build_url("aemo_price_demand", {"month": "202613", "region": "NSW1"})
    with pytest.raises(ValueError):
        sources.build_url("aemo_price_demand", {"month": "202610", "region": "WA1"})


# GISTEMP: a cell saved four stages with ps.Ledger, then copied the carried ledger over out/ledger.json.
def test_a_ledger_written_over_after_save_is_restored(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(ps, "_SAVED_LEDGERS", {})
    (tmp_path / "in").mkdir()
    old = [{"index": 1, "stage": "frame", "finding": "f", "decision": "d"}]
    (tmp_path / "in" / "ledger.json").write_text(json.dumps(old))
    led = ps.Ledger.open("in/ledger.json")
    led.add("coverage", "no gaps", "kept")
    led.add("seasonality", "period 12", "season 12")
    led.save()
    (tmp_path / "out" / "ledger.json").write_text((tmp_path / "in" / "ledger.json").read_text())   # shutil.copy
    assert ps.keep_saved_ledgers() == [os.path.abspath("out/ledger.json")]
    stages = [e["stage"] for e in json.loads((tmp_path / "out" / "ledger.json").read_text())]
    assert stages == ["frame", "coverage", "seasonality"]
    assert "written back" in capsys.readouterr().out
    assert ps.keep_saved_ledgers() == []          # nothing lost a second time


# NVDA: Ledger.add(rationale=...) raised and the ledger step stalled the run.
def test_ledger_add_takes_a_rationale():
    led = ps.Ledger()
    led.add("coverage", "no gaps", "kept", rationale="the series is complete")
    led.add("seasonality", "period 5", rationale="weekly trading cycle")
    assert [e["decision"] for e in led.entries] == ["kept Why: the series is complete", "weekly trading cycle"]


# NVDA: origin "market" was refused five times.
def test_market_origin_reads_as_retrieved(monkeypatch):
    monkeypatch.setattr(series_io, "_fetch_record_exists", lambda url: True)
    prov = {"origin": "market", "source_name": "NASDAQ", "sources": [{"id": "S1", "url": "https://api.nasdaq.com/x"}]}
    assert series_io.check_provenance(prov)["origin"] == "retrieved"
    monkeypatch.setattr(series_io, "_fetch_record_exists", lambda url: False)
    with pytest.raises(ValueError, match="no fetch record"):
        series_io.check_provenance(prov)


# AUUNEMP: a source named as text was refused six times; the refusal now names the downloaded address.
def test_refusal_names_the_downloaded_address(tmp_path, monkeypatch):
    run_id = "a" * 32
    run = tmp_path / "sandbox" / run_id
    (run / "in").mkdir(parents=True)
    (run / "out").mkdir()
    body = b"TIME_PERIOD,OBS_VALUE\n2026-10,4.3\n"
    (run / "in" / "abs_sdmx.csv").write_bytes(body)
    import hashlib
    ds = tmp_path / "datasets" / "d1"
    ds.mkdir(parents=True)
    url = "https://data.api.abs.gov.au/rest/data/ABS,LF,1.0.0/M13.3.1599.20.AUS.M?format=csv"
    (ds / "meta.json").write_text(json.dumps({"sha256": hashlib.sha256(body).hexdigest(), "provenance": {"url": url}}))
    prov = {"origin": "retrieved", "source_name": "ABS", "sources": [{"id": "S1", "url": "ABS SDMX flow ABS,LF,1.0.0"}]}
    (run / "out" / "series.json").write_text(json.dumps({"series": [4.1, 4.2, 4.3], "provenance": prov}))
    monkeypatch.setattr(series_io, "SANDBOX_ROOT", tmp_path / "sandbox")
    monkeypatch.setattr(series_io, "DATASETS_ROOT", tmp_path / "datasets")
    assert series_io.input_source_urls(run_id) == {"abs_sdmx.csv": url}
    # Round 6: a retrieved claim traced to the download is kept, with the download as its source.
    assert len(series_io.resolve_series({"source": {"run_id": run_id, "file": "out/series.json"}})[0][1]) == 3
    kept = series_io.series_provenance({"source": {"run_id": run_id, "file": "out/series.json"}})
    assert kept == {"origin": "retrieved", "source_name": "ABS", "sources": [{"id": "S1", "url": url}], "traced": True}
    # A claim with no origin is still refused, and the refusal names the address.
    (run / "out" / "series.json").write_text(json.dumps({"series": [4.1, 4.2, 4.3],
                                                         "provenance": {"source_name": "ABS", "sources": []}}))
    with pytest.raises(ValueError, match=r"abs_sdmx\.csv came from https://data\.api\.abs\.gov\.au"):
        series_io.resolve_series({"source": {"run_id": run_id, "file": "out/series.json"}})


# NVDA: "the next 2 months" on a trading-day series ran 40 steps to 26 January and was blocked as 60.9 periods.
def test_calendar_request_is_checked_by_dates():
    rec = {"config": {"horizon": 40, "series_step": 1.0}, "window": {"first": "2026-11-27", "last": "2027-01-26"}}
    check = goal._dated_span_check("the request asks for 2 months", 2, "month", rec, [])
    assert check["ok"] is True
    short = {"config": {"horizon": 20}, "window": {"first": "2026-11-27", "last": "2026-12-24"}}
    assert goal._dated_span_check("the request asks for 2 months", 2, "month", short, [])["ok"] is False
    assert goal._dated_span_check("x", 20, "trading_day", rec, []) is None
    assert goal._dated_span_check("x", 2, "month", {"config": {}}, []) is None


# EPL: the planner paged eight times through a simulation result whose game-by-game paths it never used.
def test_simulation_view_leaves_out_points_paths():
    from prophet_agent.tools.sports_tool import observe_simulate
    result = {"teams": [{"team": "A"}], "points_paths": {"A": [{"game": 1, "median": 3.0}]}, "standings": [], "ratings": []}
    view = observe_simulate(result)
    assert "points_paths" not in view and "points_paths" in result and "do not page" in view["note"]


# EPL (round 6): the goal asks for the points path, so a compact path stays in the view.
def test_simulation_view_keeps_a_compact_path():
    from prophet_agent.tools.sports_tool import observe_simulate
    path = [{"game": g, "measured": g <= 10, "median": 3.0 * g, "p10": 2.0 * g, "p90": 4.0 * g} for g in range(1, 39)]
    view = observe_simulate({"teams": [{"team": "A"}], "points_paths": {"A": path}, "standings": [], "ratings": []})
    rows = view["points_path_summary"]["A"]
    assert [r[0] for r in rows] == [20, 30, 38]
    assert rows[-1] == [38, 114.0, 76.0, 152.0]


# CPI (round 5): six monthly steps dated at month starts end on 1 April; the dated check read 152 days and asked for 8 months.
def test_a_monthly_forecast_keeps_the_period_count():
    rec = {"config": {"horizon": 6, "series_step": "monthly"},
           "window": {"first": "2026-11-01", "last": "2027-04-01", "steps": 6}}
    assert goal._dated_span_check("the request asks for 6 months", 6, "month", rec, []) is None
    no_step = {"config": {"horizon": 6}, "window": {"first": "2026-11-01", "last": "2027-04-01", "steps": 6}}
    assert goal._dated_span_check("the request asks for 6 months", 6, "month", no_step, []) is None


# AUUNEMP and NEM (round 5): the series run read carried files; the download sat in an earlier run's in/ folder.
def test_the_address_is_found_through_carried_files(tmp_path, monkeypatch):
    import hashlib
    first, second = "b" * 32, "c" * 32
    body = b"REGION,SETTLEMENTDATE,TOTALDEMAND\nNSW1,2026/11/01 00:05:00,6829.48\n"
    (tmp_path / "sandbox" / first / "in").mkdir(parents=True)
    (tmp_path / "sandbox" / first / "in" / "aemo.csv").write_bytes(body)
    (tmp_path / "sandbox" / second / "in" / "carried").mkdir(parents=True)
    (tmp_path / "sandbox" / second / "carried_from.json").write_text(json.dumps({"carried/daily.csv": first}))
    url = "https://www.aemo.com.au/aemo/data/nem/priceanddemand/PRICE_AND_DEMAND_202611_NSW1.csv"
    ds = tmp_path / "datasets" / "d2"
    ds.mkdir(parents=True)
    (ds / "meta.json").write_text(json.dumps({"sha256": hashlib.sha256(body).hexdigest(), "provenance": {"url": url}}))
    monkeypatch.setattr(series_io, "SANDBOX_ROOT", tmp_path / "sandbox")
    monkeypatch.setattr(series_io, "DATASETS_ROOT", tmp_path / "datasets")
    assert series_io.input_source_urls(second) == {"aemo.csv": url}


def test_carried_origins_are_recorded(tmp_path, monkeypatch):
    from prophet_agent.tools import sandbox_tool
    monkeypatch.setattr(sandbox_tool, "SANDBOX_ROOT", tmp_path)
    run, src = "d" * 32, "e" * 32
    (tmp_path / run).mkdir()
    sandbox_tool._record_carried_from(run, {"carried/series.json": tmp_path / src / "out" / "series.json"})
    assert json.loads((tmp_path / run / "carried_from.json").read_text()) == {"carried/series.json": src}


# NGSTORAGE (round 5): each cell carried a different earlier ledger, so no save held all the stages.
def test_a_carried_ledger_is_the_newest_of_its_name():
    old, new = "1" * 32, "2" * 32
    hist = [{"kind": "tool", "tool": "sandbox_python", "result": {"run_id": old, "outputs": [{"path": "out/ledger.json"}]}},
            {"kind": "tool", "tool": "sandbox_python", "result": {"run_id": new, "outputs": [{"path": "out/ledger.json"},
                                                                                             {"path": "out/scan.json"}]}}]
    out = goal._auto_carry({"code": "x", "carry": [{"run_id": old, "file": "out/ledger.json"},
                                                   {"run_id": old, "file": "out/series.json"}]}, hist)
    assert {"run_id": new, "file": "out/ledger.json"} in out["carry"]
    assert {"run_id": old, "file": "out/series.json"} in out["carry"]      # other files stay as the planner chose


# NEM (round 5): a sub-agent fetched aemo.com.au five times after its first page answered with a bot check.
def test_a_walled_host_is_not_fetched_again():
    from prophet_agent.tools import refuse_walled_hosts
    calls = []

    def fetch(args, key):
        calls.append(args["url"])
        return {"ok": False, "note": "interstitial bot_check (bot check), not read: title: 'just a moment...'"}
    guarded = refuse_walled_hosts(fetch)
    guarded({"url": "https://aemo.com.au/a"}, "k1")
    with pytest.raises(ValueError, match="aemo.com.au answered with a bot check"):
        guarded({"url": "https://aemo.com.au/b"}, "k2")
    guarded({"url": "https://www.aemo.com.au/data.csv"}, "k3")     # another host name is still fetched
    assert calls == ["https://aemo.com.au/a", "https://www.aemo.com.au/data.csv"]


# CO2 (round 6): a request in days counts daily values; the dated check left it alone.
def test_a_request_in_days_keeps_the_period_count():
    rec = {"config": {"horizon": 30, "series_step": 1.0}, "window": {"first": "2026-11-20", "last": "2026-12-31", "steps": 30}}
    assert goal._dated_span_check("the request asks for 30 days", 30, "day", rec, []) is None


# NGSTORAGE (round 6): eight weekly steps are eight weeks; the dated check read 50 days against 56 and blocked.
def test_a_weekly_forecast_keeps_the_period_count():
    rec = {"config": {"horizon": 8, "series_step": "weekly"}, "window": {"first": "2026-11-27", "last": "2027-01-15", "steps": 8}}
    assert goal._dated_span_check("the request asks for 8 weeks", 8, "week", rec, []) is None


# GISTEMP (round 6): a file landed before download records existed was reused, so its address had no record.
def test_a_stored_download_counts_as_fetched(tmp_path, monkeypatch):
    url = "https://data.giss.nasa.gov/gistemp/tabledata_v4/GLB.Ts+dSST.txt"
    ds = tmp_path / "d3"
    ds.mkdir()
    (ds / "meta.json").write_text(json.dumps({"sha256": "ab", "provenance": {"url": url, "final_url": url}}))
    assert series_io._stored_download(url, tmp_path) is True
    assert series_io._stored_download("https://data.giss.nasa.gov/other.txt", tmp_path) is False


def test_a_missing_page_is_not_fetched_again():
    from prophet_agent.tools import refuse_walled_hosts
    guarded = refuse_walled_hosts(lambda args, key: {"ok": False, "note": "HTTP 404"})
    guarded({"url": "https://data.giss.nasa.gov/gistemp/GMSTA/"}, "k1")
    with pytest.raises(ValueError, match="already answered HTTP 404"):
        guarded({"url": "https://data.giss.nasa.gov/gistemp/GMSTA/"}, "k2")
    guarded({"url": "https://data.giss.nasa.gov/gistemp/"}, "k3")       # the host is not refused


# NBA (round 7): a 13,592-character view was clipped, the planner paged 18 times through the full result, and the answer
# stated backtest errors no tool produced. The view fits one observation.
def test_simulation_view_fits_one_observation():
    from prophet_agent.runtime.goal import OBS_CHARS
    from prophet_agent.tools.sports_tool import observe_simulate
    teams = [f"Team {i}" for i in range(30)]
    rows = [{"team": t, "current": 4.0, "expected_final": 41.123456, "expected_rank_in_group": 12.3456,
             "final": {"median": 41.0, "p10": 35.0, "p90": 47.0}, "playoff": 0.512345} for t in teams]
    paths = {t: [{"game": g, "measured": g <= 26, "median": g * 0.5, "p10": g * 0.4, "p90": g * 0.6}
                 for g in range(1, 83)] for t in teams}
    bt = {"models": {"m": {"brier": 0.2492, "log_loss": 0.6916, "brier_home": 0.1}}, "n_test": 131, "origins": 6,
          "comparisons": [{"comparison": f"m vs b{i}", "verdict": "no_clear_difference", "mean_difference": -0.0016,
                           "ci95": [-0.005, 0.002]} for i in range(4)]}
    full = {"teams": rows, "points_paths": paths, "standings": rows, "ratings": rows, "backtest": bt,
            "settings": {"model": "normal score-margin model", "n_played": 261, "margin_sd": 15.9}}
    view = observe_simulate(full)
    assert len(json.dumps(view)) < OBS_CHARS
    assert len(view["teams"]["rows"]) == 6 and view["teams_not_shown"] == 24 and view["points_path_summary"]
    assert view["backtest"]["models"] == {"m": {"brier": 0.249, "log_loss": 0.692}}
    assert view["settings"] == {"model": "normal score-margin model", "n_played": 261}


# Round 7: "10/90 quantiles" (CO2) and "10–90% ranges" (EPL) are labels; finish sent as a tool call is the finish action.
@pytest.mark.parametrize("phrase", ["point forecasts and 10/90 quantiles for each day", "their median paths and 10–90% ranges"])
def test_more_band_labels(phrase):
    assert unverified(phrase, [], "forecast") == []


def test_finish_tool_call_becomes_the_finish_action():
    out = goal.as_question_action({"action": "call_tool", "tool": "finish", "args": {}, "rationale": "done"})
    assert out == {"action": "finish", "rationale": "done"}


# AUUNEMP (round 7): Ledger.open() read the empty out/ of a new cell while the ledger sat at in/carried/ledger.json.
def test_ledger_open_falls_back_to_the_carried_file(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "in" / "carried").mkdir(parents=True)
    (tmp_path / "in" / "carried" / "ledger.json").write_text(json.dumps(
        [{"index": 1, "stage": "coverage", "finding": "no gaps", "decision": "kept"}]))
    led = ps.Ledger.open()
    assert [e["stage"] for e in led.entries] == ["coverage"]
    assert "opened in/carried/ledger.json (1 entries)" in capsys.readouterr().out
    (tmp_path / "in" / "carried" / "ledger.json").unlink()
    assert ps.Ledger.open().entries == [] and "starting empty" in capsys.readouterr().out


# NBA (round 8, independent audit): 18 wrong figures in standings and forecast rows passed, because each number was printed
# somewhere else in the evidence. A number in a row that names an entity is checked against that entity's own records.
def _sim_history():
    teams = [{"team": "Boston Celtics", "current": 12.0, "final": {"median": 43.0, "p10": 38.0, "p90": 49.0}, "playoff": 0.62},
             {"team": "Detroit Pistons", "current": 11.0, "final": {"median": 44.0, "p10": 39.0, "p90": 48.0}, "playoff": 0.58}]
    standings = [{"team": "Boston Celtics", "played": 18, "wins": 12}, {"team": "Detroit Pistons", "played": 17, "wins": 11}]
    paths = {"Boston Celtics": [{"game": 30, "median": 20.0, "p10": 18.0, "p90": 22.0}]}
    stdout = "        Boston Celtics    12       6     18 0.666667\n       Detroit Pistons    11       6     17 0.647059\n"
    return [{"kind": "tool", "tool": "simulate_season", "result": {"teams": teams, "standings": standings,
                                                                   "points_paths": paths}},
            {"kind": "tool", "tool": "sandbox_python", "result": {"stdout": stdout}}]


def test_entity_rows_match_their_own_records():
    from prophet_agent.runtime.grounding import entity_row_mismatches
    good = ("| Team | W | L | GP | Pct |\n|---|---|---|---|---|\n| Boston Celtics | 12 | 6 | 18 | .667 |\n"
            "| Detroit Pistons | 11 | 6 | 17 | .647 |\n\n| Team | Median | Range | Playoff |\n|---|---|---|---|\n"
            "| Boston Celtics | 43 | 38 – 49 | 62% |\n")
    assert entity_row_mismatches(good, _sim_history()) == []


def test_a_figure_from_another_entity_is_caught():
    from prophet_agent.runtime.grounding import entity_row_mismatches
    # 44 and 39 are Detroit's numbers, and .706 is nobody's: in Boston's row they are wrong.
    bad = "| Team | Median | Range | Pct |\n|---|---|---|---|\n| Boston Celtics | 44 | 39 – 49 | .706 |\n"
    assert entity_row_mismatches(bad, _sim_history()) == ["44", "39", ".706"]
    assert set(unverified(bad, _sim_history(), "forecast")) >= {"44", "39"}


def test_path_values_count_only_in_a_path_table():
    from prophet_agent.runtime.grounding import entity_row_mismatches
    path_table = "| Team | After game 30 |\n|---|---|\n| Boston Celtics | 20 (18 – 22) |\n"
    assert entity_row_mismatches(path_table, _sim_history()) == []
    other = "| Team | Wins |\n|---|---|\n| Boston Celtics | 20 |\n"
    assert entity_row_mismatches(other, _sim_history()) == ["20"]


def test_method_names_are_not_entities():
    from prophet_agent.runtime.grounding import entity_row_mismatches
    hist = [{"kind": "tool", "tool": "backtest_series", "result": {"models": {"timesfm": {"mase": 0.313}}, "gain_pct": 29.16}}]
    assert entity_row_mismatches("| Method | MASE | Gain |\n|---|---|---|\n| TimesFM | 0.313 | 29% |\n", hist) == []
