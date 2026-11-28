"""Round 5: ps.Ledger.decide records a scan finding by its id, and a save with no new entry says so.

Live retail_biz rep5 (measured, 2026-11-23): the repair cell guarded each add on scan["calendar"]["closed_days"], a key the scan
file does not have (it is {"findings": [...], "summary": ...}), so both adds were skipped and the saved ledger had the same sha256
as before. The finished answer was held back twice and left partial. Electricity rep9 and Beijing rep5 guessed other layouts.
"""
from __future__ import annotations

import json

import pytest

from prophet_agent.sandbox import prophet_stats as ps


@pytest.fixture()
def run(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "in" / "carried").mkdir(parents=True)
    (tmp_path / "out").mkdir()
    scan = {"findings": [
        {"id": "calendar.closed_days", "severity": "major", "scope": "all", "evidence": {"closed_days": 135}},
        {"id": "censoring.plateau", "severity": "major", "scope": "MT_001", "evidence": {"run": 9}},
        {"id": "censoring.plateau", "severity": "major", "scope": "MT_002", "evidence": {"run": 4}},
        {"id": "values.spikes", "severity": "minor", "scope": "all", "evidence": {}}], "summary": {}}
    (tmp_path / "in" / "carried" / "scan.json").write_text(json.dumps(scan))
    return tmp_path


def test_decide_reads_the_stage_scope_and_measure_from_the_scan(run):
    led = ps.Ledger()
    led.decide("calendar.closed_days", "trading-day series; non-trading days not filled")
    led.decide("censoring.plateau", "kept as observed")
    e = led.entries
    assert e[0]["finding"].startswith("calendar.closed_days (all): ") and "135" in e[0]["finding"]
    assert e[1]["stage"] == "zeros and censoring"
    assert e[1]["finding"].startswith("censoring.plateau (2 series: MT_001, MT_002)")


def test_an_unknown_id_is_kept_and_the_note_names_the_real_ones(run, capsys):
    # live SPX (round 20261128-110827): the raise failed the cell and the model resent it until the mission stalled
    led = ps.Ledger()
    led.decide("calendar.closed", "x")
    out = capsys.readouterr().out
    assert "calendar.closed_days" in out and "values.spikes" not in out
    assert led.entries[0]["finding"] == "calendar.closed" and led.entries[0]["decision"] == "x"


def test_without_a_scan_the_id_alone_is_the_finding(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    led = ps.Ledger()
    led.decide("stationarity.verdict", "forecast on the original scale")
    assert led.entries[0]["finding"] == "stationarity.verdict" and led.entries[0]["stage"] == "stationarity"


def test_a_save_that_adds_nothing_says_so(run, capsys):
    led = ps.Ledger()
    led.add("time index", "ok", "kept")
    led.save()
    capsys.readouterr()
    again = ps.Ledger.open("out/ledger.json")
    again.save()
    assert "no new entry in this cell" in capsys.readouterr().out
    fresh = ps.Ledger()
    fresh.add("values", "x", "y")
    fresh.save()
    assert "no new entry" not in capsys.readouterr().out


def test_the_scan_check_and_the_ledger_agree_on_decide(run):
    # the check passes on a ledger written with decide: each entry names its finding id
    from prophet_agent.runtime import contracts
    led = ps.Ledger()
    led.decide("calendar.closed_days", "trading days only")
    led.decide("censoring.plateau", "kept")
    led.save()
    texts = [" ".join(str(x.get(k, "")) for k in ("stage", "finding", "decision")) for x in led.entries]
    assert all(any(i in t for t in texts) for i in ("calendar.closed_days", "censoring.plateau"))
    assert contracts.STAGE_OF == ps._STAGE_OF
