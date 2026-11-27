"""RBA statistics tables (round 21, AURATE 2026-11-27): the RBA's own table is read (one-cell title line, metadata rows, three
date formats), a DBnomics copy whose last period is a day is still judged by its frequency, and a stale RBA copy is fetched
from the RBA, from the live successor when the mirror's series was stopped. Offline: the download is replaced."""
from __future__ import annotations

import pytest

from prophet_agent.opendata import fetch, sources
from prophet_agent.opendata.store import rba_csv
from prophet_agent.tools import opendata_tools as ot

F11 = ("﻿F1.1 INTEREST RATES AND YIELDS – MONEY MARKET\r\n"
       "Title,Cash Rate Target,Interbank Overnight Cash Rate\r\n"
       "Description,Cash Rate Target; monthly average,Interbank Overnight Cash Rate; monthly average\r\n"
       "Frequency,Monthly,Monthly\r\n"
       "Type,Original,Original\r\n"
       "Units,Per cent,Per cent\r\n"
       "\r\n"
       "Source,RBA,RBA\r\n"
       "Publication date,02-Nov-2026,02-Nov-2026\r\n"
       "Series ID,FIRMMCRT,FIRMMCRI\r\n"
       "30/06/1969,,\r\n"
       "30/09/2026,4.36,4.36\r\n"
       "31/10/2026,4.60,4.61\r\n").encode("utf-8")

F13 = ("Title,Australia Target Cash Rate\nFrequency,Monthly\nUnits,Per cent\nSeries ID,FOOIRATCR\n"
       "Jan-1980,14.00\nFeb-2024,4.35\nMay-2024,4.35\nJul-2024,\n").encode("utf-8")

F1 = ("Title,Cash Rate Target\nFrequency,Daily\nSeries ID,FIRMMCRTD\n25-Nov-2026,4.85\n26-Nov-2026,4.85\n").encode("utf-8")


def test_a_table_with_a_title_line_is_accepted_as_csv():
    assert fetch._csv_like(F11.decode("utf-8-sig"))
    assert not fetch._csv_like("just one line of prose\nand another\nand a third\nfourth, with a comma")


def test_the_table_becomes_date_and_series_columns_with_metadata_kept():
    body, extra = rba_csv(F11)
    lines = body.decode().splitlines()
    assert lines[0] == "date,FIRMMCRT,FIRMMCRI"
    assert lines[1:] == ["2026-09-30,4.36,4.36", "2026-10-31,4.60,4.61"], "the empty 1969 row is dropped"
    info = extra["rba"]["series"]["FIRMMCRT"]
    assert info["title"] == "Cash Rate Target" and info["units"] == "Per cent" and info["frequency"] == "Monthly"
    assert info["published"] == "02-Nov-2026" and info["last_date"] == "2026-10-31"


def test_one_series_is_kept_and_an_unknown_one_is_refused():
    body, extra = rba_csv(F11, "FIRMMCRI")
    assert body.decode().splitlines() == ["date,FIRMMCRI", "2026-09-30,4.36", "2026-10-31,4.61"]
    with pytest.raises(ValueError, match="FIRMMCRT, FIRMMCRI"):
        rba_csv(F11, "NOPE")


def test_month_labels_and_daily_dates_parse():
    assert rba_csv(F13)[0].decode().splitlines()[1:] == ["1980-01-31,14.00", "2024-02-29,4.35", "2024-05-31,4.35"]
    assert rba_csv(F1)[1]["rba"]["series"]["FIRMMCRTD"]["last_date"] == "2026-11-26"


def test_the_source_takes_a_table_and_keeps_the_series_off_the_url():
    assert sources.build_url("rba_cash_rate", {}) == "https://www.rba.gov.au/statistics/tables/csv/f1.1-data.csv"
    assert sources.build_url("rba_cash_rate", {"table": "f13", "series": "FOOIRATCR"}).endswith("/csv/f13-data.csv")
    with pytest.raises(ValueError):
        sources.build_url("rba_cash_rate", {"table": "../x"})


def test_a_mirror_copy_dated_by_day_is_judged_by_its_frequency(monkeypatch):
    from datetime import date
    monkeypatch.setattr(ot, "_today", lambda: date(2026, 11, 27))
    stale = ot._stale_mirror({"frequency": "monthly", "last_period": "2024-01-31"})
    assert stale and stale["frequency"] == "M" and stale["age_days"] > 1000
    assert ot._stale_mirror({"frequency": "monthly", "last_period": "2026-10-31"}) is None


class _Live:
    def __init__(self):
        self.failed = {}


def test_a_stopped_rba_series_is_fetched_from_its_live_successor(monkeypatch):
    asked = []

    def fake(source, params, store):
        asked.append((source, dict(params)))
        body, extra = rba_csv(F11, params.get("series"))
        return {"id": "d1", "provenance": {"params": params, **extra}}

    monkeypatch.setattr(ot, "download_to_dataset", fake)
    got = ot._from_rba(None, _Live(), {"provider": "RBA", "dataset": "F13", "series": "FOOIRATCR"},
                       {"last_period": "2024-01-31"})
    assert asked == [("rba_cash_rate", {"table": "f1.1", "series": "FIRMMCRT"})]
    assert got and got["provenance"]["params"]["series"] == "FIRMMCRT"


def test_a_direct_copy_no_newer_than_the_mirror_is_not_used(monkeypatch):
    monkeypatch.setattr(ot, "download_to_dataset", lambda s, p, store: {
        "id": "d2", "provenance": {"params": p, "rba": {"series": {p["series"]: {"last_date": "2024-05-31"}}}}})
    assert ot._from_rba(None, _Live(), {"provider": "RBA", "dataset": "F13", "series": "FOOIRUKOBR"},
                        {"last_period": "2024-05-31"}) is None
    newer = ot._from_rba(None, _Live(), {"provider": "RBA", "dataset": "F13", "series": "FOOIRUKOBR"},
                         {"last_period": "2024-01-31"})
    assert newer and newer["provenance"]["params"] == {"table": "f13", "series": "FOOIRUKOBR"}
