"""Tables from HTML: embedded JSON (embedded.py) and markup (tables.py), with the league-table shape and the layout rules.
Offline: the fixtures are synthetic (fictional clubs, invented figures)."""
import csv
import io
import json
from pathlib import Path

import pytest

from prophet_agent.research import tables
from prophet_agent.research.embedded import MAX_ROWS, embedded_tables
from prophet_agent.research.tables import html_tables

FIXTURES = Path(__file__).resolve().parents[1] / "opendata" / "fixtures"
LEAGUE_HEADER = ["#", "Club", "P", "W", "D", "L", "GD", "Pts", "Form"]


def page(*scripts: str, body: str = "") -> str:
    return "<!doctype html><html><body>" + body + "".join(scripts) + "</body></html>"


def records(count: int) -> list[dict]:
    return [{"rank": i + 1, "name": f"Club {i + 1}", "score": 100 - i, "flag": True} for i in range(count)]


# --- markup: league table shape and layout rules ----------------------------------------------------------------------

def test_a_20_row_league_table_passes_with_text_form_and_signed_goal_difference():
    html = (FIXTURES / "standings_20.html").read_text(encoding="utf-8")
    result = html_tables(html, base_url="https://clubs.example.org/t")
    assert len(result["candidates"]) == 1, "the navigation table is layout and is not a candidate"
    cand = result["candidates"][0]
    assert cand["columns"] == LEAGUE_HEADER
    assert cand["row_count"] == 20
    rows = list(csv.reader(io.StringIO(cand["csv"])))
    assert len(rows) == 21
    assert any(row[6].startswith("+") for row in rows[1:]), "a signed GD such as +10 is kept as text"
    assert all(row[8].isalpha() for row in rows[1:]), "the Form column is text"


def test_a_table_that_holds_another_table_is_layout_and_is_skipped():
    html = page(body=(
        "<table><tr><td><table><tr><th>a</th><th>b</th></tr><tr><td>1</td><td>2</td></tr></table></td>"
        "<td>x</td></tr><tr><td>y</td><td>z</td></tr></table>"))
    result = html_tables(html)
    assert result["skipped_layout"] >= 1
    assert all(cand["columns"] == ["a", "b"] for cand in result["candidates"])


def test_a_table_of_prose_cells_is_layout_even_with_a_number():
    long = "The squad trained hard all week before the next match. " * 5
    body = (f"<table><tr><th>Note</th><th>Other</th><th>Count</th></tr>"
            f"<tr><td>{long}</td><td>{long}</td><td>3</td></tr>"
            f"<tr><td>{long}</td><td>{long}</td><td>4</td></tr></table>")
    result = html_tables(page(body=body))
    assert result["candidates"] == [] and result["skipped_layout"] == 1


def test_the_candidate_cap_is_twelve_tables():
    body = "".join(f"<table><tr><th>n</th><th>v</th></tr><tr><td>{i}</td><td>{i * 2}</td></tr></table>"
                   for i in range(20))
    result = html_tables(page(body=body))
    assert len(result["candidates"]) == tables.MAX_TABLES == 12


def test_the_scan_cap_is_sixty_tables():
    assert tables.MAX_SCAN == 60
    body = "".join(f"<table><tr><th>a</th><th>b</th></tr><tr><td>{i}</td><td>x</td></tr></table>" for i in range(70))
    assert len(html_tables(page(body=body))["candidates"]) == 12


# --- embedded JSON ----------------------------------------------------------------------------------------------------

def test_next_data_records_become_a_candidate_with_its_json_path():
    html = (FIXTURES / "next_data_standings.html").read_text(encoding="utf-8")
    cands = embedded_tables(html, base_url="https://clubs.example.org/t")["candidates"]
    assert len(cands) == 1
    cand = cands[0]
    assert cand["method"] == "next_data"
    assert cand["locator"] == "__NEXT_DATA__:$.props.pageProps.standings.table"
    assert cand["row_count"] == 20 and "team.name" in cand["columns"]
    assert cand["final_url"] == "https://clubs.example.org/t" and cand["truncated"] is False


def test_json_ld_dataset_measures_become_a_candidate():
    html = (FIXTURES / "json_ld_dataset.html").read_text(encoding="utf-8")
    cand = embedded_tables(html)["candidates"][0]
    assert cand["method"] == "json_ld" and cand["locator"] == "ld+json:$.variableMeasured"
    assert cand["row_count"] == 4 and "value" in cand["columns"]


def test_an_assigned_window_state_object_is_read():
    state = {"standings": records(3)}
    html = page(f"<script>window.__INITIAL_STATE__ = {json.dumps(state)};</script>")
    cand = embedded_tables(html)["candidates"][0]
    assert cand["method"] == "inline_state" and cand["locator"] == "__INITIAL_STATE__:$.standings"
    assert cand["row_count"] == 3


def test_an_inline_array_of_records_is_read():
    html = page('<script>var rows = [{"a": "x", "n": 1}, {"a": "y", "n": 2}, {"a": "z", "n": 3}];</script>')
    cand = embedded_tables(html)["candidates"][0]
    assert cand["method"] == "inline_json" and cand["row_count"] == 3
    assert cand["csv"].splitlines()[1] == "x,1"


def test_a_list_of_links_with_no_numbers_is_not_a_table():
    menu = [{"label": "Home", "href": "/"}, {"label": "Fixtures", "href": "/fixtures"}, {"label": "News", "href": "/news"}]
    html = page(f'<script type="application/json">{json.dumps({"menu": menu})}</script>')
    assert embedded_tables(html)["candidates"] == []


def test_a_nuxt_function_call_is_skipped_without_error():
    html = page("<script>window.__NUXT__=(function(a){return {data:[]}})(1);</script>")
    assert embedded_tables(html)["candidates"] == []


def test_malformed_embedded_json_is_ignored():
    html = page('<script type="application/ld+json">{not json</script>')
    assert embedded_tables(html) == {"candidates": []}


def test_records_over_the_row_cap_are_truncated_and_marked():
    big = json.dumps({"props": {"rows": records(MAX_ROWS + 1)}})
    html = page(f'<script id="__NEXT_DATA__" type="application/json">{big}</script>')
    cand = embedded_tables(html)["candidates"][0]
    assert cand["row_count"] == MAX_ROWS and cand["truncated"] is True


def test_the_same_table_in_two_places_is_listed_once():
    state = json.dumps({"rows": records(3)})
    html = page(f'<script id="__NEXT_DATA__" type="application/json">{state}</script>',
                f"<script>window.__INITIAL_STATE__ = {state};</script>")
    assert len(embedded_tables(html)["candidates"]) == 1
