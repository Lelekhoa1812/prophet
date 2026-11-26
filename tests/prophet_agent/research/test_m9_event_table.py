"""M9-4: dated events from news text become an event table (typed, deduplicated, every source URL kept, date origin labelled).
Offline: the events are written by hand or come from events.dated_events on canned passages."""
import csv
import io

import pytest

from prophet_agent.research import events as ev
from prophet_agent.research.event_table import (classify_type, date_origin, event_table_csv, normalise_events,
                                                select_events)

NVIDIA = "NVIDIA will report its results on 19 November 2026."


def row(date, text, url, **extra):
    return {"date": date, "event": text, "source_url": url, **extra}


def test_one_event_from_three_sources_keeps_all_three_urls():
    rows, refused = normalise_events([
        row("2026-11-19", NVIDIA, "https://a.example/1", how_sure="high", date_found="stated in text"),
        row("2026-11-19", "Nvidia reports Q3 results on 19 November", "https://b.example/2", how_sure="medium",
            date_found="publication date"),
        row("2026-11-19", "Nvidia reported results Nov 19", "https://c.example/3", how_sure="low", date_found="inferred"),
    ])
    assert refused == []
    assert len(rows) == 1
    merged = rows[0]
    assert merged["source_count"] == 3
    assert merged["source_urls"] == ["https://a.example/1", "https://b.example/2", "https://c.example/3"]
    assert merged["certainty"] == "high" and merged["date_found"] == "stated in text" and merged["type"] == "earnings"


def test_same_day_different_fixtures_stay_separate_and_a_reordered_pair_merges():
    rows, _ = normalise_events([
        row("2026-03-14", "Arsenal vs Chelsea kick off 5pm", "https://d.example/1", how_sure="high"),
        row("2026-03-14", "Chelsea v Arsenal, kick-off 17:00", "https://e.example/2", how_sure="high"),
        row("2026-03-14", "Liverpool vs Everton kick off 3pm", "https://f.example/3", how_sure="high"),
    ])
    assert len(rows) == 2
    assert sorted(r["source_count"] for r in rows) == [1, 2]
    assert all(r["type"] == "fixture" for r in rows)


def test_different_companies_on_one_day_are_not_merged():
    rows, _ = normalise_events([
        row("2026-10-29", "Apple earnings due Thursday", "https://g.example/1", how_sure="high"),
        row("2026-10-29", "Microsoft earnings due Thursday", "https://h.example/2", how_sure="high"),
    ])
    assert len(rows) == 2


def test_an_explicit_entity_keeps_different_entities_apart_even_with_the_same_wording():
    rows, _ = normalise_events([
        row("2026-10-29", "Quarterly results due", "https://g.example/1", how_sure="high", entity="Apple"),
        row("2026-10-29", "Quarterly results due", "https://h.example/2", how_sure="high", entity="Microsoft"),
    ])
    assert len(rows) == 2 and {r["entity"] for r in rows} == {"Apple", "Microsoft"}


@pytest.mark.parametrize("text, kind", [
    ("Markets closed for the Thanksgiving holiday", "holiday"),
    ("Grid outage forces rolling shutdowns", "outage"),
    ("Cyclone expected to make landfall", "weather"),
    ("The RBA cash rate decision is due", "policy"),
    ("OPEC+ meeting on production quotas", "policy"),
    ("Company posts final results for the year", "earnings"),
    ("Dividend of 0.10 per share goes ex", "earnings"),
    ("Derby match kick-off at 3pm", "fixture"),
    ("The company will unveil a new model", "product"),
    ("Statistics bureau publishes the monthly series", "other"),
])
def test_type_comes_from_the_keywords(text, kind):
    assert classify_type(text) == kind


def test_date_origin_is_stated_only_when_the_year_is_written():
    assert date_origin("The meeting is on 29 November 2026.") == "stated in text"
    assert date_origin("The meeting is on 29 November.") == "inferred"
    assert date_origin("The meeting is on 2026-11-29.") == "stated in text"


def test_certainty_is_capped_by_how_the_date_was_found():
    rows, _ = normalise_events([
        row("2026-11-29", "OPEC+ meeting on production", "https://o.example/1", how_sure="high", date_found="publication date"),
        row("2026-11-30", "OPEC+ follow-up talks", "https://o.example/2", how_sure="high", date_found="inferred"),
    ])
    by_date = {r["date"]: r for r in rows}
    assert by_date["2026-11-29"]["certainty"] == "medium"
    assert by_date["2026-11-30"]["certainty"] == "low"


def test_rows_without_a_source_url_or_a_real_date_are_refused_with_the_reason():
    rows, refused = normalise_events([
        row("2026-02-30", "Results day", "https://x.example/1"),
        row("2026-02-03", "Results day", "ftp://x.example/1"),
        {"date": "2026-02-03", "event": "Results day"},
        row("2026-02-03", "Results day", "https://x.example/2", end_date="2026-02-01"),
        row("2026-02-03", "Results day", "https://x.example/3"),
    ])
    assert len(rows) == 1 and rows[0]["source_urls"] == ["https://x.example/3"]
    reasons = [r["reason"] for r in refused]
    assert len(refused) == 4
    assert "ISO date" in reasons[0] and "http(s)" in reasons[1] and "source_url" in reasons[2] and "before date" in reasons[3]


def test_an_end_date_is_kept_and_merged_as_the_later_end():
    rows, _ = normalise_events([
        row("2026-06-01", "Festival of lights runs", "https://f.example/1", end_date="2026-06-03"),
        row("2026-06-01", "Festival of lights runs for three days", "https://f.example/2", end_date="2026-06-05"),
    ])
    assert len(rows) == 1 and rows[0]["end_date"] == "2026-06-05" and rows[0]["source_count"] == 2


def test_dated_events_from_research_carry_their_date_origin_into_the_table():
    sources = [{"id": "S1", "url": "https://www.eia.gov/steo", "tier": 1, "date": "2026-11-10",
                "passages": [{"text": "The OPEC+ meeting is scheduled for 29 November 2026. The meeting is on 3 December."}]}]
    found = ev.dated_events(sources)
    labels = {e["date"]: e["date_found"] for e in found}
    assert labels == {"2026-11-29": "stated in text", "2026-12-03": "publication date"}
    rows, refused = normalise_events(found)
    assert refused == []
    assert all(r["source_urls"] == ["https://www.eia.gov/steo"] for r in rows)


def test_select_events_and_csv_keep_the_urls_and_the_labels():
    rows, _ = normalise_events([
        row("2026-11-29", "OPEC+ meeting on production", "https://o.example/1", how_sure="high", date_found="stated in text"),
        row("2026-11-29", "OPEC+ meeting on production quotas", "https://o.example/2", how_sure="high",
            date_found="stated in text"),
        row("2026-12-02", "Rate decision by the central bank", "https://c.example/1", date_found="inferred"),
    ])
    kept = select_events(rows, "medium")
    assert [r["date"] for r in kept] == ["2026-11-29"]
    with pytest.raises(ValueError, match="min_certainty"):
        select_events(rows, "sure")
    parsed = list(csv.DictReader(io.StringIO(event_table_csv(rows).decode("utf-8"))))
    assert len(parsed) == 2
    assert parsed[0]["source_urls"] == "https://o.example/1 | https://o.example/2"
    assert parsed[0]["source_count"] == "2" and parsed[0]["type"] == "policy"
    assert parsed[1]["certainty"] == "low" and parsed[1]["date_found"] == "inferred"
    assert set(parsed[0]) == {"date", "end_date", "event", "type", "entity", "date_found", "certainty", "source_count",
                              "source_urls"}
