"""Expected cardinality for table claims: short, complete and long tables, named missing clubs, and count inference.
Offline. The league is the synthetic standings fixture (fictional clubs, invented figures)."""
import csv
import io
from pathlib import Path

import pytest
from bs4 import BeautifulSoup

from prophet_agent.research.tables import html_tables
from prophet_agent.source_table.claim import build_claim
from prophet_agent.source_table.validate import validate_table

FIXTURE = Path(__file__).resolve().parents[1] / "opendata" / "fixtures" / "standings_20.html"
URL = "https://clubs.example.org/harbour"
TAIL = " The Harbour League is played between its member clubs."


def league_csv() -> str:
    return html_tables(FIXTURE.read_text(encoding="utf-8"))["candidates"][0]["csv"]


def club_names() -> list[str]:
    return [row[1] for row in list(csv.reader(io.StringIO(league_csv())))[1:]]


def page_with(count_phrase: str = "The league has 20 clubs.") -> str:
    text = BeautifulSoup(FIXTURE.read_text(encoding="utf-8"), "lxml").get_text(" ")
    return " ".join(text.split()) + " " + count_phrase + TAIL


def league_claim(**extra) -> dict:
    raw = {"series_name": "harbour_points", "entity": "Harbour League", "aliases": ["Harbour League"],
           "measure": "Pts", "table_kind": "league"}
    raw.update(extra)
    return build_claim(raw)


def first_rows(count: int) -> bytes:
    lines = league_csv().splitlines()
    return ("\n".join(lines[:count + 1]) + "\n").encode("utf-8")


def results_csv(teams: list[str]) -> bytes:
    lines = ["Date,Home Team,Away Team,Score"]
    for index in range(0, len(teams) - 1, 2):
        lines.append(f"2026-08-{index + 1:02d},{teams[index]},{teams[index + 1]},1-0")
    return ("\n".join(lines) + "\n").encode("utf-8")


# --- a league table of 20 clubs ---------------------------------------------------------------------------------------

def test_a_20_row_league_table_is_accepted_as_complete_with_the_count_from_the_page():
    verdict = validate_table(league_claim(), league_csv().encode("utf-8"), final_url=URL, page_text=page_with())
    assert verdict["status"] == "accepted" and verdict["bound_name"] == "harbour_points"
    assert verdict["cardinality"] == "complete"
    assert verdict["expected_rows"] == 20 and verdict["expected_source"] == "page_text"
    assert verdict["row_count"] == 20 and verdict["missing_entities"] == []


def test_an_11_row_league_table_is_refused_as_short_and_names_the_missing_clubs():
    clubs = club_names()
    verdict = validate_table(league_claim(expected_entities=clubs), first_rows(11), final_url=URL,
                             page_text=page_with())
    assert verdict["status"] == "short" and verdict["bound_name"] is None
    assert verdict["cardinality"] == "short" and verdict["row_count"] == 11 and verdict["expected_rows"] == 20
    assert verdict["missing_entities"] == clubs[11:], "the nine clubs not in the table are named, in table order"
    assert all(name in verdict["detail"] for name in clubs[11:])


def test_a_page_count_alone_sets_the_expected_rows_for_a_short_table():
    verdict = validate_table(league_claim(), first_rows(11), final_url=URL,
                             page_text=page_with("The league has 20 teams."))
    assert verdict["status"] == "short" and verdict["expected_rows"] == 20
    assert verdict["expected_source"] == "page_text" and verdict["missing_entities"] == []


def test_a_table_with_more_rows_than_expected_is_long_and_not_accepted():
    body = league_csv().rstrip("\n") + "\n21,Extra FC,12,0,0,12,-30,0,LLLLL\n"
    verdict = validate_table(league_claim(), body.encode("utf-8"), final_url=URL, page_text=page_with())
    assert verdict["status"] == "long" and verdict["cardinality"] == "long" and verdict["bound_name"] is None


def test_a_league_claim_with_no_count_source_is_not_accepted_as_complete():
    verdict = validate_table(league_claim(), league_csv().encode("utf-8"), final_url=URL,
                             page_text=page_with("The top 4 teams qualify for Europe."))
    assert verdict["status"] == "count_unstated" and verdict["bound_name"] is None
    assert verdict["cardinality"] == "unknown" and verdict["expected_rows"] is None


def test_a_position_such_as_top_four_teams_is_not_read_as_the_league_size():
    verdict = validate_table(league_claim(), first_rows(4), final_url=URL,
                             page_text=page_with("The top 4 teams qualify. The league has 20 clubs."))
    assert verdict["expected_rows"] == 20 and verdict["status"] == "short"


def test_teams_in_a_results_table_of_the_same_competition_give_the_expected_rows():
    clubs = club_names()
    reference = results_csv(clubs)
    short = validate_table(league_claim(), first_rows(11), final_url=URL,
                           page_text=page_with("The Harbour League table is below."), reference_csvs=[reference])
    assert short["status"] == "short" and short["expected_rows"] == 20
    assert short["expected_source"] == "results_table"
    complete = validate_table(league_claim(), league_csv().encode("utf-8"), final_url=URL,
                              page_text=page_with("The Harbour League table is below."), reference_csvs=[reference])
    assert complete["status"] == "accepted" and complete["expected_source"] == "results_table"


def test_sources_that_disagree_give_no_count_and_no_acceptance():
    clubs = club_names()
    verdict = validate_table(league_claim(), league_csv().encode("utf-8"), final_url=URL,
                             page_text=page_with("The league has 20 clubs."),
                             reference_csvs=[results_csv(clubs[:18])])
    assert verdict["expected_rows"] is None and verdict["expected_conflict"] is True
    assert verdict["status"] == "count_unstated" and verdict["bound_name"] is None
    assert "disagree" in verdict["detail"]


# --- the same check for a claim that is not a league ------------------------------------------------------------------

def test_an_explicit_row_count_applies_to_a_dated_series_too():
    body = b"date,name,close\n2024-01-01,Alpha,3\n2024-01-02,Alpha,4\n"
    base = {"series_name": "alpha_daily", "entity": "Alpha", "aliases": ["Alpha"], "measure": "close"}
    short = validate_table(build_claim({**base, "expected_rows": 3}), body, final_url=URL)
    assert short["status"] == "short" and short["cardinality"] == "short"
    exact = validate_table(build_claim({**base, "expected_rows": 2}), body, final_url=URL)
    assert exact["status"] == "accepted" and exact["cardinality"] == "complete"
    unstated = validate_table(build_claim(base), body, final_url=URL)
    assert unstated["status"] == "accepted" and unstated["cardinality"] == "unknown"


# --- claim fields ------------------------------------------------------------------------------------------------------

def test_a_claim_without_the_new_fields_keeps_its_original_shape():
    claim = build_claim({"series_name": "alpha_daily", "entity": "Alpha"})
    assert "expected_rows" not in claim and "expected_entities" not in claim and "table_kind" not in claim


@pytest.mark.parametrize("extra", [
    {"expected_rows": 0}, {"expected_rows": True}, {"expected_rows": "20"}, {"table_kind": "cup"},
    {"expected_rows": 3, "expected_entities": ["A", "B"]},
])
def test_build_claim_refuses_a_bad_count_or_kind(extra):
    with pytest.raises(ValueError):
        build_claim({"series_name": "alpha_daily", "entity": "Alpha", **extra})


def test_the_entity_list_is_stored_and_its_length_is_the_count():
    claim = build_claim({"series_name": "alpha_daily", "entity": "Alpha", "expected_entities": ["A", "B", "A"]})
    assert claim["expected_entities"] == ["A", "B"]
    assert validate_table(claim, b"date,name,close\n2024-01-01,Alpha,3\n", final_url=URL)["expected_rows"] == 2
