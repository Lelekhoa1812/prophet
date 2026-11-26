"""extract_page_charts mounts each landed series for the mission's sandbox: the result names in/<file> and sandbox_python
reads that file in the same run. A full mission gets no new file."""
import pytest

from prophet_agent.datasets import DatasetStore
from prophet_agent.research import chart_fetch
from prophet_agent.tools import default_tools
from prophet_agent.tools.opendata_tools import OpenDataContext

PAGE = """<html><body>
<script>Highcharts.chart('c', { xAxis: { categories: ['Jan', 'Feb', 'Mar'] },
  series: [{ name: 'Revenue', data: [3, 4.5, 6] }] });</script>
</body></html>"""
READ_SERIES = ("import csv\nrows = list(csv.DictReader(open('in/chart-Revenue.csv')))\n"
               "print(sum(float(r['y']) for r in rows))\n")


@pytest.fixture
def fake_page(monkeypatch):
    def fetch(url, *, max_bytes, media, **kwargs):
        return {"ok": True, "status": 200, "media": media[0], "body": PAGE.encode(), "final_url": url, "note": ""}
    monkeypatch.setattr(chart_fetch, "fetch_bytes", fetch)


def test_landed_series_is_read_by_the_sandbox_in_the_same_run(fake_page, tmp_path):
    attached = []
    store = DatasetStore(tmp_path / "datasets")
    tools = default_tools(None, opendata=OpenDataContext(store=store, attach=attached.append))
    out = tools["extract_page_charts"].fn({"url": "https://example.com/p"}, "chart-key")
    (series,) = out["series"]
    assert series["landed"] is True
    assert series["sandbox_file"] == "in/chart-Revenue.csv"
    assert attached == [series["dataset_id"]]
    res = tools["sandbox_python"].fn({"code": READ_SERIES}, "sandbox-key")
    assert res["ok"], res["stderr"]
    assert res["stdout"].strip() == "13.5"


def test_a_full_mission_gets_no_new_file(fake_page, tmp_path):
    store = DatasetStore(tmp_path / "datasets")
    files = {f"file{i}.csv": tmp_path / f"file{i}.csv" for i in range(10)}
    for path in files.values():
        path.write_text("a,b\n1,2\n")
    attached = []
    tools = default_tools(files, opendata=OpenDataContext(store=store, attach=attached.append))
    out = tools["extract_page_charts"].fn({"url": "https://example.com/p"}, "chart-key")
    (series,) = out["series"]
    assert series["landed"] is False
    assert "already holds 10 files" in series["note"]
    assert "sandbox_file" not in series
    assert attached == []
