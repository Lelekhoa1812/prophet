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
    assert "points_paths" not in view and "points_paths" in result and "points paths" in view["note"]


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
