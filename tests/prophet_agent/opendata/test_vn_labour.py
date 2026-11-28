"""Mission 10, M10-4: Vietnam's quarterly unemployment rate assembled from the statistics office's releases.

Workbooks and pages here are synthetic, built in the layout the releases used on 2026-11-28; the values are made up.
"""
import importlib
import io
import json

import openpyxl
import pytest

from prophet_agent.opendata import vn_labour as v
from prophet_agent.runtime import goal

ost = importlib.import_module("prophet_agent.tools.official_series_tool")

PAGE_2023 = v.HOST + "/du-lieu-va-so-lieu-thong-ke/2023/04/thong-cao-bao-chi-tinh-hinh-lao-dong-viec-lam-quy-i-nam-2023/"
PAGE_2026 = v.HOST + "/tin-tuc-thong-ke/2026/10/thong-cao-bao-chi-ve-tinh-hinh-lao-dong-viec-lam-quy-iii-va-9-thang-nam-2026/"
BOOK_2026 = v.HOST + "/wp-content/uploads/2026/10/Bieu-so-lieu-Lao-dong-viec-lam-QIII.2026.xlsx"


def _book(columns: list[str], rate: list[float]) -> bytes:
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "LĐ Quy"
    ws.append(["Biểu 1. Một số chỉ tiêu lao động, việc làm"])
    ws.append([None] + columns)
    ws.append(["Lực lượng lao động (Nghìn người)"] + [50000.0] * len(columns))
    ws.append(["- Số người thất nghiệp trong độ tuổi lao động "] + [1050.0] * len(columns))
    ws.append(["Tỷ lệ thất nghiệp trong độ tuổi lao động (%) "] + rate)
    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()


def test_workbook_reads_quarter_columns_only():
    data = _book(["Quý III \nnăm 2025", "9 tháng\nnăm 2025", "Quý II \nnăm 2026", "Quý III \nnăm 2026"],
                 [2.21, 2.22, 2.23, 2.217])
    assert v.parse_workbook(data) == {(2025, 3): 2.21, (2026, 2): 2.23, (2026, 3): 2.217}


def test_release_text_and_quarter_from_address():
    page = "<p>Tỷ lệ thất nghiệp của lực lượng lao động trong độ tuổi quý I năm 2023 là 2,25%, giảm 0,07 điểm</p>"
    assert v.parse_release_text(page) == 2.25
    assert v.release_quarter(PAGE_2023) == (2023, 1)
    assert v.release_quarter(PAGE_2026) == (2026, 3)
    assert v.release_quarter(v.HOST + "/tin-tuc-thong-ke/2026/10/gia-vang/") is None


def _site(extra_search=()):
    old_book = _book(["Quý III năm 2024", "Quý III năm 2025"], [2.23, 2.30])          # an older figure for 2025-Q3
    pages = {
        PAGE_2023: "<p>Tỷ lệ thất nghiệp trong độ tuổi lao động quý I năm 2023 là 2,25%.</p>".encode(),
        PAGE_2026: f'<a href="{BOOK_2026}">Biểu số liệu</a>'.encode(),
        BOOK_2026: _book(["Quý III năm 2025", "Quý II năm 2026", "Quý III năm 2026"], [2.21, 2.23, 2.217]),
    }
    known = v.HOST + "/tin-tuc-thong-ke/2025/10/thong-cao-bao-chi-ve-tinh-hinh-lao-dong-viec-lam-quy-iii-va-9-thang-nam-2025/"
    book25 = v.HOST + "/wp-content/uploads/2025/10/Bieu_baocao_Q3.2025.xlsx"
    pages[known] = f'<a href="{book25}">x</a>'.encode()
    pages[book25] = old_book
    asked = []

    def get(url):
        asked.append(url)
        if "wp-json" in url:
            return json.dumps([{"link": PAGE_2026}, {"link": PAGE_2023}, *extra_search]).encode()
        if url not in pages:
            raise RuntimeError("HTTP 404")
        return pages[url]
    return get, asked, known


def test_series_merges_releases_newest_wins_and_keeps_sources(monkeypatch):
    get, asked, known = _site()
    monkeypatch.setattr(v, "KNOWN_RELEASES", (known,))
    monkeypatch.setattr(v.time, "sleep", lambda s: None)
    res = v.quarterly_unemployment(get)
    vals = {r["period"]: r for r in res["series"]}
    assert vals["2023-Q1"]["value"] == 2.25 and vals["2023-Q1"]["source"] == PAGE_2023
    assert vals["2025-Q3"]["value"] == 2.21 and vals["2025-Q3"]["source"] == BOOK_2026     # the 2026 release revises it
    assert res["last"] == "2026-Q3" and vals["2026-Q3"]["date"] == "2026-07-01"
    assert "2024-Q1" in res["gaps"]
    assert "not the ILO" in res["measure"]


def test_tool_saves_dated_series_and_observes_the_grain(monkeypatch):
    get, _, known = _site()
    monkeypatch.setattr(v, "KNOWN_RELEASES", (known,))
    monkeypatch.setattr(v.time, "sleep", lambda s: None)
    saved = {}

    def save(name, dates, values):
        saved.update(name=name, dates=dates, values=values)
        return {"run_id": "r1", "file": ost.OUT, "name": name}

    res = ost._run({"series": "vn_unemployment_quarterly"}, "k", get=get, save=save)
    assert saved["dates"][-1] == "2026-07-01" and saved["values"][-1] == 2.217
    view = ost.observe(res)
    assert view["source"]["run_id"] == "r1" and "derived" in view["next"]
    with pytest.raises(ValueError, match="series must be"):
        ost._run({"series": "nope"}, "k", get=get, save=save)


def test_harness_assembles_the_vietnam_series_first_and_once():
    tools = {"official_series": object()}
    g = "Find Vietnam's monthly unemployment rate from public sources and forecast it month by month over the next 2 years"
    step = goal._official_series_step(g, [], tools)
    assert step["args"] == {"series": "vn_unemployment_quarterly"}
    assert goal._official_series_step(g, [{"kind": "tool", "tool": "official_series"}], tools) is None
    assert goal._official_series_step("Forecast Australia's unemployment rate", [], tools) is None
    plan = goal._harness_plan(g, [], tools)
    assert plan and plan["tool"] == "official_series"
