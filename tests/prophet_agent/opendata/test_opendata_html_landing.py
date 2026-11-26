"""An HTML page lands its best data table as a CSV dataset, with provenance. The network is replaced (fetch.open_client returns
an httpx.MockTransport, names resolve to a public address), and the store lives under tmp_path. The fixtures are synthetic:
fictional clubs and invented figures."""
import csv
from pathlib import Path

import httpx
import pytest

from prophet_agent.datasets import DatasetStore
from prophet_agent.opendata import fetch
from prophet_agent.research import netguard
from prophet_agent.runtime.runner import ArgumentRejected
from prophet_agent.tools import default_tools
from prophet_agent.tools.opendata_tools import OpenDataContext

FIXTURES = Path(__file__).resolve().parent / "fixtures"
PUBLIC = "93.184.216.34"
PAGE_URL = "https://clubs.example.org/tables/premier-league"
HEADER = ["#", "Club", "P", "W", "D", "L", "GD", "Pts", "Form"]


def fixture(name: str) -> bytes:
    return (FIXTURES / name).read_bytes()


@pytest.fixture
def page(monkeypatch):
    """Serve ``body`` at every request and count them."""
    state = {"body": b"", "content_type": "text/html; charset=utf-8", "calls": 0}

    def handler(request):
        state["calls"] += 1
        return httpx.Response(200, headers={"content-type": state["content_type"]}, content=state["body"])

    monkeypatch.setattr(fetch, "open_client",
                        lambda timeout_s: httpx.Client(transport=httpx.MockTransport(handler), follow_redirects=False))
    monkeypatch.setattr(netguard, "_default_resolver", lambda host, port: [PUBLIC])
    return state


class Mission:
    def __init__(self, root: Path):
        self.store = DatasetStore(root / "datasets")
        self.ids: list[str] = []

    def download(self, **args):
        tool = default_tools(opendata=OpenDataContext(store=self.store, attach=self.ids.append))["download_web_data"]
        return tool.fn({"url": PAGE_URL, **args}, "k")

    def csv_rows(self, out: dict) -> list[list[str]]:
        return list(csv.reader(self.store.path(out["dataset_id"]).read_text(encoding="utf-8").splitlines()))


@pytest.fixture
def mission(tmp_path):
    return Mission(tmp_path)


def test_a_20_club_league_table_lands_every_club(page, mission):
    page["body"] = fixture("standings_20.html")
    out = mission.download()
    rows = mission.csv_rows(out)
    assert rows[0] == HEADER
    assert len(rows) - 1 == 20 and out["row_count"] == 20
    assert len({row[1] for row in rows[1:]}) == 20, "every club is a distinct row"
    assert any(row[6].startswith("+") for row in rows[1:]), "signed goal difference is kept as text"
    assert out["extraction"]["method"] == "html_table" and out["extraction"]["locator"] == "table[0]"
    assert out["extraction"]["row_count"] == 20 and out["extraction"]["column_count"] == 9


def test_the_provenance_records_the_page_table_and_counts(page, mission):
    page["body"] = fixture("standings_20.html")
    out = mission.download()
    provenance = mission.store.get(out["dataset_id"])["provenance"]
    assert provenance["url"] == PAGE_URL and provenance["html"] is True and provenance["format"] == "csv"
    assert provenance["content_type"] == "text/html" and provenance["series_bound"] is False
    assert provenance["extraction"]["method"] == "html_table"
    assert provenance["extraction"]["row_count"] == 20 and provenance["extraction"]["column_count"] == 9
    assert len(provenance["page_sha256"]) == 64


def test_the_results_list_is_not_chosen_over_the_standings(page, mission):
    page["body"] = fixture("two_tables.html")
    out = mission.download()
    assert out["row_count"] == 20
    others = out["other_tables"]
    assert [other["row_count"] for other in others] == [3]
    assert others[0]["header"] == ["Date", "Home", "Score", "Away"] and others[0]["index"] == 0


def test_the_table_argument_picks_a_listed_candidate(page, mission):
    page["body"] = fixture("two_tables.html")
    best = mission.download()
    picked = mission.download(table=0)
    assert picked["row_count"] == 3 and picked["dataset_id"] != best["dataset_id"]
    assert len(mission.csv_rows(picked)) - 1 == 3
    assert page["calls"] == 2, "a different table request is a different download"


def test_the_same_page_and_table_request_return_the_stored_copy(page, mission):
    page["body"] = fixture("standings_20.html")
    first = mission.download()
    second = mission.download()
    assert second["dataset_id"] == first["dataset_id"] and page["calls"] == 1


def test_next_data_records_land_with_their_json_path_and_flattened_names(page, mission):
    page["body"] = fixture("next_data_standings.html")
    out = mission.download()
    assert out["extraction"]["method"] == "next_data"
    assert out["extraction"]["locator"] == "__NEXT_DATA__:$.props.pageProps.standings.table"
    assert out["row_count"] == 20
    rows = mission.csv_rows(out)
    assert "team.name" in rows[0] and "points" in rows[0]
    assert rows[1][rows[0].index("team.name")] != "", "the nested team name is flattened into a column"
    assert rows[1][rows[0].index("position")] == "1"


def test_a_json_ld_dataset_lands_its_measures(page, mission):
    page["body"] = fixture("json_ld_dataset.html")
    out = mission.download()
    assert out["extraction"]["method"] == "json_ld"
    assert out["extraction"]["locator"] == "ld+json:$.variableMeasured"
    assert out["row_count"] == 4
    rows = mission.csv_rows(out)
    assert rows[1][rows[0].index("value")] == "12.5"


def test_a_page_with_no_data_table_is_refused_with_a_way_forward(page, mission):
    page["body"] = fixture("no_table.html")
    with pytest.raises(ValueError, match="has no data table") as raised:
        mission.download()
    assert "web_fetch" in str(raised.value) and "extract_page_tables" in str(raised.value)
    assert mission.ids == [] and list(mission.store.root.glob("*")) == []


def test_an_unknown_table_argument_lists_what_the_page_offers(page, mission):
    page["body"] = fixture("two_tables.html")
    with pytest.raises(ValueError, match="no table matches '9'.*table\\[") as raised:
        mission.download(table="9")
    assert "rows" in str(raised.value)
    assert mission.ids == []


@pytest.mark.parametrize("bad", [True, 1.5, ["table"]])
def test_a_table_argument_of_the_wrong_type_is_rejected(page, mission, bad):
    page["body"] = fixture("standings_20.html")
    with pytest.raises(ArgumentRejected, match="table must be"):
        mission.download(table=bad)
    assert page["calls"] == 0
