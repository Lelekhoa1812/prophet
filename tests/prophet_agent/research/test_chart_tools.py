"""extract_page_charts: embedded data lands exactly, chart pictures are digitised, uploads are read, bad arguments refused."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from prophet_agent.datasets import DatasetStore
from prophet_agent.research import chart_fetch
from prophet_agent.runtime.runner import ArgumentRejected
from prophet_agent.tools import default_tools
from prophet_agent.tools.chart_tools import chart_tool_pair
from prophet_agent.tools.opendata_tools import OpenDataContext
from tests.prophet_agent.research.chart_fixtures import known_series, render

PAGE = """<html><body>
<script>Highcharts.chart('c', { xAxis: { categories: ['Jan', 'Feb'] },
  series: [{ name: 'Revenue', data: [3, 4.5] }] });</script>
</body></html>"""
BARE_PAGE = '<html><body><img src="/img/chart-sales.png" alt="Sales chart"><p>text</p></body></html>'


class FakeClient:
    def __init__(self, proposal):
        self.text = json.dumps(proposal)

    def complete(self, models, messages, **kwargs):
        return SimpleNamespace(text=self.text, model="nvidia/nemotron-3.5-super-vl-120b-a12b")


def _fetch_factory(pages: dict):
    def fetch(url, *, max_bytes, media, **kwargs):
        if url not in pages:
            return {"ok": False, "status": 404, "media": "", "body": b"", "final_url": url, "note": "HTTP 404"}
        body = pages[url]
        return {"ok": True, "status": 200, "media": media[0], "body": body, "final_url": url, "note": ""}
    return fetch


@pytest.fixture
def store(tmp_path):
    return DatasetStore(tmp_path / "datasets")


@pytest.fixture
def ctx(store):
    attached = []
    return OpenDataContext(store=store, attach=attached.append), attached


def _run(pair, args):
    tool, _spec = pair["extract_page_charts"]
    return tool.fn(args, "key")


def test_embedded_page_series_lands_exact_dataset(monkeypatch, ctx):
    context, attached = ctx
    monkeypatch.setattr(chart_fetch, "fetch_bytes", _fetch_factory({"https://example.com/p": PAGE.encode()}))
    out = _run(chart_tool_pair(None, context), {"url": "https://example.com/p"})
    assert out["ok"] is True
    (s,) = out["series"]
    assert s["method"] == "highcharts" and s["exactness"] == "embedded"
    assert s["first_rows"] == [{"x": "Jan", "y": 3}, {"x": "Feb", "y": 4.5}]
    assert s["landed"] is True and attached == [s["dataset_id"]]
    meta = context.store.get(s["dataset_id"])
    assert meta["provenance"]["exactness"] == "embedded"
    assert meta["provenance"]["licence_key"] == "unverified"


def test_no_embedded_data_digitises_a_chart_picture(monkeypatch, ctx, store):
    context, _ = ctx
    x, y = known_series()
    png, proposal = render(x, y)
    monkeypatch.setattr(chart_fetch, "fetch_bytes", _fetch_factory({
        "https://example.com/bare": BARE_PAGE.encode(),
        "https://example.com/img/chart-sales.png": png,
    }))
    import prophet_agent.nim.transport as transport
    monkeypatch.setattr(transport, "shared_client", lambda name=None: FakeClient(proposal))
    out = _run(chart_tool_pair(None, context), {"url": "https://example.com/bare"})
    assert out["chart_images_seen"] == ["https://example.com/img/chart-sales.png"]
    (s,) = out["series"]
    assert s["exactness"] == "read_from_image"
    assert s["error_bound_y"] > 0
    assert s["landed"] is True


def test_uploaded_image_is_read_from_the_mission_file(tmp_path, monkeypatch, ctx):
    context, _ = ctx
    x, y = known_series()
    png, proposal = render(x, y)
    path = tmp_path / "chart.png"
    path.write_bytes(png)
    import prophet_agent.nim.transport as transport
    monkeypatch.setattr(transport, "shared_client", lambda name=None: FakeClient(proposal))
    out = _run(chart_tool_pair({"chart.png": path}, context), {"image": "in/chart.png", "series": "known"})
    (s,) = out["series"]
    assert s["exactness"] == "read_from_image" and s["source"]["kind"] == "upload"


def test_series_filter_and_skipped_reasons(monkeypatch, ctx):
    context, _ = ctx
    page = PAGE.replace("name: 'Revenue'", "name: 'Revenue'") + "<script>Plotly.newPlot('d', [{x:[1], y:[2], name:'Cost'}]);</script>"
    monkeypatch.setattr(chart_fetch, "fetch_bytes", _fetch_factory({"https://example.com/p": page.encode()}))
    out = _run(chart_tool_pair(None, context), {"url": "https://example.com/p", "series": "cost"})
    assert [s["name"] for s in out["series"]] == ["Cost"]


def test_refused_page_is_returned_not_counted(monkeypatch, ctx):
    context, _ = ctx
    monkeypatch.setattr(chart_fetch, "fetch_bytes", _fetch_factory({}))
    out = _run(chart_tool_pair(None, context), {"url": "https://example.com/missing"})
    assert out["ok"] is False and "404" in out["note"]


@pytest.mark.parametrize("args", [
    {},
    {"url": "https://a.example/x", "image": "in/c.png"},
    {"url": "ftp://a.example/x"},
    {"url": "https://a.example/x", "max_images": 9},
    {"url": "https://a.example/x", "extra": 1},
])
def test_bad_arguments_are_refused(args, ctx):
    context, _ = ctx
    with pytest.raises(ArgumentRejected):
        _run(chart_tool_pair(None, context), args)


def test_unknown_mission_file_is_refused(tmp_path, ctx):
    context, _ = ctx
    with pytest.raises(ArgumentRejected, match="no mission file"):
        _run(chart_tool_pair({}, context), {"image": "in/nothing.png"})


def test_tool_is_in_the_default_catalogue():
    tools = default_tools()
    assert "extract_page_charts" in tools
    assert "args" not in tools["extract_page_charts"].doc or "url" in tools["extract_page_charts"].doc
    assert "url" in tools["extract_page_charts"].doc and "image" in tools["extract_page_charts"].doc
