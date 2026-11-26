"""The planner tools bind a series only after the check. No network."""
from prophet_agent.datasets import DatasetStore
from prophet_agent.tools import default_tools
from prophet_agent.tools.opendata_tools import OpenDataContext


class _Mission:
    def __init__(self, root):
        self.store = DatasetStore(root / "datasets")
        self.ids = []

    def context(self):
        return OpenDataContext(store=self.store, attach=self.ids.append)


def test_bind_series_refuses_the_requested_name_for_a_substitute_file(tmp_path):
    tools = default_tools(opendata=_Mission(tmp_path).context())
    claim = tools["state_table_claim"].fn(
        {"series_name": "alpha_daily", "entity": "Alpha", "aliases": ["Alpha", "AAA"], "measure": "close"}, "k1")
    refused = tools["bind_series"].fn({
        "series_name": claim["series_name"], "entity": claim["entity"], "aliases": claim["aliases"],
        "measure": "close", "final_url": "https://example.test/beta.csv",
        "csv": "date,ticker,name,close\n2020-01-02,BBB,Beta,10\n",
    }, "k2")
    assert refused["status"] == "wrong_entity"
    assert refused["bound_name"] is None
    assert refused["dataset_id"] is None
    page = ("<table><tr><th>date</th><th>name</th><th>close</th></tr>"
            "<tr><td>2020-01-02</td><td>Alpha</td><td>10</td></tr></table>")
    extracted = tools["extract_page_tables"].fn({"html": page, "url": "https://example.test/alpha"}, "k3")
    assert extracted["candidates"][0]["series_bound"] is False
    accepted = tools["bind_series"].fn({
        "series_name": claim["series_name"], "entity": claim["entity"], "measure": "close",
        "dataset_id": extracted["candidates"][0]["dataset_id"],
        "final_url": "https://example.test/alpha",
    }, "k4")
    assert accepted["status"] == "accepted"
    assert accepted["name"] == "alpha_daily.csv"
