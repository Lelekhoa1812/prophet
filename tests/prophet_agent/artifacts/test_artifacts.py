"""Chart and document pack (GOAL G11): generate in the sandbox, reopen each file with its own library, check hashes."""
from __future__ import annotations

from pathlib import Path

import pytest

from prophet_agent.artifacts import REQUIRED_KINDS, build_forecast_pack
from prophet_agent.sandbox.runner import verify_manifest


@pytest.fixture(scope="module")
def pack():
    history = [10 + (i % 7) * 0.5 for i in range(60)]
    point = [12.0 + 0.1 * i for i in range(12)]
    q10 = [p - 2 for p in point]
    q90 = [p + 2 for p in point]
    res = build_forecast_pack(history=history, point=point, q10=q10, q90=q90,
                              title="Weekly demand forecast (synthetic)", note="Synthetic data.")
    assert res.ok, res.stderr
    return res


def _out(pack, name):
    return Path(pack.workdir) / "out" / name


def test_every_required_kind_is_produced(pack):
    names = {Path(o.path).name for o in pack.outputs}
    assert {"forecast.png", "forecast.svg", "forecast.xlsx", "report.docx", "report.pdf"} <= names
    kinds = {Path(o.path).suffix.lstrip(".") for o in pack.outputs}
    assert set(REQUIRED_KINDS) <= kinds


def test_manifest_hashes_match_the_files_on_disk(pack):
    assert verify_manifest(Path(pack.workdir) / "manifest.json") == []


def test_png_reopens_with_pillow(pack):
    from PIL import Image
    with Image.open(_out(pack, "forecast.png")) as im:
        im.load()
        assert im.format == "PNG" and im.width > 500 and im.height > 300


def test_svg_is_well_formed_xml(pack):
    import xml.etree.ElementTree as ET
    root = ET.parse(_out(pack, "forecast.svg")).getroot()
    assert root.tag.endswith("svg")


def test_xlsx_reopens_with_the_forecast_rows(pack):
    import openpyxl
    wb = openpyxl.load_workbook(_out(pack, "forecast.xlsx"), read_only=True)
    rows = list(wb["Forecast"].iter_rows(values_only=True))
    wb.close()
    assert rows[0] == ("Periods ahead", "Forecast", "Low (P10)", "High (P90)")
    assert len(rows) == 13
    assert rows[1][0] == 1 and abs(rows[1][1] - 12.0) < 1e-9
    assert rows[1][2] < rows[1][1] < rows[1][3]


def test_docx_reopens_with_title_and_picture(pack):
    from docx import Document
    doc = Document(_out(pack, "report.docx"))
    texts = [p.text for p in doc.paragraphs]
    assert "Weekly demand forecast (synthetic)" in texts
    assert any("Synthetic data." in t for t in texts)
    assert len(doc.inline_shapes) == 1


def test_pdf_has_a_valid_header_and_a_page(pack):
    data = _out(pack, "report.pdf").read_bytes()
    assert data.startswith(b"%PDF-")
    assert b"/Type /Page" in data or b"/Type/Page" in data


def test_mismatched_inputs_are_refused():
    with pytest.raises(ValueError):
        build_forecast_pack(history=[1.0], point=[1.0, 2.0], q10=[1.0], q90=[1.0], title="t", note="n")


def _dated_pack():
    import pandas as pd
    from prophet_agent.tools.report_tool import _future_dates
    n, h = 200, 14
    hist_dates = [d.strftime("%Y-%m-%d") for d in pd.date_range("2025-01-01", periods=n)]
    ahead = _future_dates(hist_dates, h)
    point = [100.0 + i for i in range(h)]
    res = build_forecast_pack(
        history=[90.0 + (i % 7) for i in range(n)], point=point, q10=[p - 9 for p in point], q90=[p + 9 for p in point],
        q30=[p - 3 for p in point], q70=[p + 3 for p in point], title="Dated", note="n", dates=hist_dates, future_dates=ahead,
        events=[{"name": "Promo", "spans": [[n + 3, n + 6]]}], ignored=[[10, 11]], unit="Units")
    assert res.ok, res.stderr
    return res, ahead


def test_future_dates_continue_the_histories_own_rhythm():
    import pandas as pd
    from prophet_agent.tools.report_tool import _future_dates
    daily = [d.strftime("%Y-%m-%d") for d in pd.date_range("2025-01-30", periods=10)]
    assert _future_dates(daily, 3) == ["2025-02-09", "2025-02-10", "2025-02-11"]
    month_end = [d.strftime("%Y-%m-%d") for d in pd.date_range("2024-01-31", periods=12, freq="ME")]
    assert _future_dates(month_end, 2) == ["2025-01-31", "2025-02-28"]
    assert _future_dates(["2025-01-01", "2025-01-02"], 3) is None                              # too short to know the step
    assert _future_dates(["2025-01-01", "2025-01-02", "2025-03-09", "2025-03-10", "2025-06-01"], 3) is None  # irregular
    assert _future_dates(["not a date", "x", "y"], 2) is None


def test_event_spans_come_only_from_on_off_drivers_of_the_right_length(tmp_path, monkeypatch):
    import json
    from prophet_agent.sandbox import runner
    from prophet_agent.tools import report_tool, series_io
    run_id = "a" * 32
    out = tmp_path / run_id / "out"
    out.mkdir(parents=True)
    n, h = 10, 4
    (out / "series.json").write_text(json.dumps({"series": {"s": [1.0] * n}, "covariates": {
        "promo": [0] * 8 + [1, 1] + [0, 1, 1, 0], "price": [1.5] * (n + h), "short": [1] * 3}}))
    monkeypatch.setattr(series_io, "SANDBOX_ROOT", tmp_path)
    spans = report_tool._event_spans({"source": {"run_id": run_id}, "future_covariates": ["promo", "price", "short", "missing"]}, n, h)
    assert spans == [{"name": "promo", "spans": [[8, 10], [11, 13]]}]


def test_dates_events_and_ignored_points_reach_the_chart_and_workbook():
    import openpyxl
    res, ahead = _dated_pack()
    assert _out(res, "forecast.png").stat().st_size > 30_000                                  # 200 dpi, not a thumbnail
    ws = openpyxl.load_workbook(_out(res, "forecast.xlsx")).active
    rows = list(ws.iter_rows(values_only=True))
    assert rows[0] == ("Date", "Periods ahead", "Forecast", "Low (P10)", "High (P90)", "Promo")
    assert rows[1][0] == ahead[0] and rows[1][1] == 1
    assert [r[5] for r in rows[1:]] == [0, 0, 0, 1, 1, 1] + [0] * 8                           # spans [n+3, n+6)


def test_a_pack_without_dates_still_builds_with_period_numbers():
    res = build_forecast_pack(history=[1.0] * 20, point=[1.0, 2.0], q10=[0.5, 1.5], q90=[1.5, 2.5], title="t")
    assert res.ok, res.stderr
    import openpyxl
    ws = openpyxl.load_workbook(Path(res.workdir) / "out" / "forecast.xlsx").active
    assert [c.value for c in ws[1]] == ["Periods ahead", "Forecast", "Low (P10)", "High (P90)"]
