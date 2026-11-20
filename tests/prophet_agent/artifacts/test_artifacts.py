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
                              title="Weekly demand forecast (synthetic)", note="Research use only. Synthetic data.")
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
    assert any("Research use only" in t for t in texts)
    assert len(doc.inline_shapes) == 1


def test_pdf_has_a_valid_header_and_a_page(pack):
    data = _out(pack, "report.pdf").read_bytes()
    assert data.startswith(b"%PDF-")
    assert b"/Type /Page" in data or b"/Type/Page" in data


def test_mismatched_inputs_are_refused():
    with pytest.raises(ValueError):
        build_forecast_pack(history=[1.0], point=[1.0, 2.0], q10=[1.0], q90=[1.0], title="t", note="n")
