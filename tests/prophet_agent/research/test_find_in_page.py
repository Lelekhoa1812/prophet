"""find_in_page searches a cached page with no network, returns exact offsets and the page's table list, and says so when the
page is not cached. web_fetch lists the same table candidates, bounded."""
import json
from dataclasses import asdict

import pytest

from prophet_agent.research.fetch import Page
from prophet_agent.research.search import cache_key, cache_root, write_cache
from prophet_agent.research.tables import html_tables
from prophet_agent.tools import research_tools as rt

URL = "https://stats.example.org/league"
TEXT = ("Season overview for the league table.\n\n"
        "Brent crude is forecast to average 78.5 dollars per barrel in 2027.\n\n"
        "Inventories rose by 3.2 million barrels in the latest week.\n\n"
        "Unrelated remarks about shipping insurance and labour contracts.")
HTML = ("<html><body><p>Table of standings.</p>"
        "<table><tr><th>Team</th><th>Played</th><th>Points</th></tr>"
        "<tr><td>Arsenal</td><td>12</td><td>30</td></tr>"
        "<tr><td>Chelsea</td><td>12</td><td>28</td></tr></table></body></html>")


@pytest.fixture(autouse=True)
def isolated_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("PROPHET_RESEARCH_CACHE", str(tmp_path / "cache"))


def _cache(url: str, page: Page) -> None:
    """Writes a page the way fetch_page does, so find_in_page reads what a real web_fetch would have stored."""
    path = cache_root() / "pages" / f"{cache_key('page', url)}.json"
    write_cache(path, asdict(page))


def _page(url: str = URL, text: str = TEXT, html: str = HTML) -> Page:
    tables = html_tables(html, base_url=url)["candidates"] if html else []
    return Page(url=url, final_url=url, title="League", text=text, sha256="s", fetched_at="2026-11-26T00:00:00",
                status=200, ok=True, media="text/html", tables=tables)


def _no_network(*args, **kwargs):
    raise AssertionError("find_in_page must not fetch")


def test_queries_return_passages_with_exact_offsets(monkeypatch):
    _cache(URL, _page())
    monkeypatch.setattr(rt, "fetch_page", _no_network)
    out = rt.find_in_page_tool({"url": URL, "queries": ["brent crude", "inventories barrels"]}, "k")
    assert out["ok"] is True and out["in_cache"] is True and out["mode"] == "queries"
    assert out["passages"], "a query that appears in the page must match"
    for passage in out["passages"]:
        assert TEXT[passage["start"]:passage["end"]] == passage["text"]
    assert out["passages"][0]["text"].startswith("Brent crude") or "Inventories" in out["passages"][0]["text"]


def test_regex_returns_each_match_with_context_and_exact_offsets(monkeypatch):
    _cache(URL, _page())
    monkeypatch.setattr(rt, "fetch_page", _no_network)
    out = rt.find_in_page_tool({"url": URL, "regex": r"\d+\.\d+ million barrels"}, "k")
    assert out["mode"] == "regex" and len(out["passages"]) == 1
    (passage,) = out["passages"]
    assert passage["match"] == "3.2 million barrels"
    assert TEXT[passage["start"]:passage["end"]] == passage["text"]
    assert "Inventories rose" in passage["text"]


def test_table_list_names_index_header_rows_and_locator(monkeypatch):
    _cache(URL, _page())
    monkeypatch.setattr(rt, "fetch_page", _no_network)
    out = rt.find_in_page_tool({"url": URL, "queries": ["standings"]}, "k")
    assert out["table_count"] == 1
    (table,) = out["tables"]
    assert table == {"index": 0, "header": ["Team", "Played", "Points"], "row_count": 2, "locator": "table[0]"}
    assert "download_web_data" in out["note"]


def test_a_page_not_in_the_cache_is_said_so_and_nothing_is_fetched(monkeypatch):
    monkeypatch.setattr(rt, "fetch_page", _no_network)
    out = rt.find_in_page_tool({"url": "https://stats.example.org/never-read", "queries": ["x"]}, "k")
    assert out["ok"] is False and out["in_cache"] is False
    assert "web_fetch" in out["note"]


def test_a_failed_page_is_not_served_from_the_cache(monkeypatch):
    failed = Page(url=URL, final_url=URL, note="HTTP 503", ok=False, status=503)
    _cache(URL, failed)
    out = rt.find_in_page_tool({"url": URL, "queries": ["brent"]}, "k")
    assert out["in_cache"] is False


def test_no_query_match_says_so_and_gives_no_passage(monkeypatch):
    _cache(URL, _page())
    out = rt.find_in_page_tool({"url": URL, "queries": ["zeppelin hangar"]}, "k")
    assert out["ok"] is True and out["passages"] == []
    assert "no passage matches" in out["note"]


@pytest.mark.parametrize("args", [
    {"url": URL},
    {"url": URL, "queries": ["a"], "regex": "b"},
    {"url": URL, "queries": []},
    {"url": URL, "queries": [None]},
    {"url": URL, "regex": "("},
    {"url": URL, "regex": "a" * 201},
    {"url": "ftp://stats.example.org/x", "queries": ["a"]},
    {"url": URL, "queries": ["a"], "depth": 2},
])
def test_bad_arguments_are_refused(args):
    with pytest.raises(ValueError):
        rt.find_in_page_tool(args, "k")


def test_result_stays_within_the_observation_budget(monkeypatch):
    big = " ".join(f"Row {i} records 1.{i} million barrels of crude." for i in range(3000))
    _cache(URL, _page(text=big, html=HTML))
    out = rt.find_in_page_tool({"url": URL, "regex": r"million barrels"}, "k")
    seen = rt.observe(out)
    assert len(json.dumps(seen, ensure_ascii=False)) <= rt.OBSERVE_CHARS + 500
    assert seen["passages"], "the budget drops passages from the end, it does not empty the result"


def test_page_level_passages_are_not_clipped_to_the_research_cap():
    """web_fetch and find_in_page passages are what the planner asked to read (a table, often): only the size bound applies."""
    text = "Club Played Points " + " ".join(f"Team{i} 13 {30 - i}" for i in range(80))
    out = {"url": URL, "passages": [{"text": text, "start": 0, "end": len(text), "reason": "query"}]}
    seen = rt.observe(out)
    assert seen["passages"][0]["text"] == text and len(text) > rt.PASSAGE_SEEN_CHARS


def test_web_fetch_lists_the_page_tables(monkeypatch):
    page = _page()
    monkeypatch.setattr(rt, "fetch_page", lambda url, render=False: page)
    out = rt.web_fetch_tool({"url": URL}, "k")
    assert out["table_count"] == 1 and out["tables"][0]["header"] == ["Team", "Played", "Points"]
    assert out["tables"][0]["row_count"] == 2
    assert "download_web_data" in out["note"]


def test_web_fetch_without_tables_adds_no_table_key(monkeypatch):
    page = _page(html="")
    monkeypatch.setattr(rt, "fetch_page", lambda url, render=False: page)
    out = rt.web_fetch_tool({"url": URL, "queries": ["brent"]}, "k")
    assert "tables" not in out


def test_web_fetch_table_list_is_bounded(monkeypatch):
    tables = [{"index": i, "columns": [f"col{j}_" + "x" * 80 for j in range(20)], "row_count": i + 1,
               "locator": f"table[{i}]", "csv": "", "sha256": "", "preview_rows": [], "final_url": URL}
              for i in range(12)]
    page = Page(url=URL, final_url=URL, text=TEXT, ok=True, media="text/html", tables=tables)
    monkeypatch.setattr(rt, "fetch_page", lambda url, render=False: page)
    out = rt.web_fetch_tool({"url": URL}, "k")
    assert out["table_count"] == 12 and len(out["tables"]) == rt.TABLES_LISTED
    assert len(out["tables"][0]["header"]) == rt.TABLE_HEADER_COLUMNS + 1
    assert all(len(name) <= rt.TABLE_NAME_CHARS for name in out["tables"][0]["header"][:-1])


def test_find_in_page_is_registered_as_an_idempotent_research_tool():
    pairs = {tool.name: (tool, spec) for tool, spec in rt.tools()}
    tool, spec = pairs["find_in_page"]
    assert tool.idempotent is True and set(spec["args"]) == {"url", "queries", "regex"}
    assert spec["title"] and spec["activity"]
