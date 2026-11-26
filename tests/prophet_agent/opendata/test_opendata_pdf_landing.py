"""A PDF lands its best data table as a CSV dataset, with provenance. The PDFs are built in the test with reportlab, so no
fixture file is checked in. The network is replaced (fetch.open_client returns an httpx.MockTransport, names resolve to a
public address), and the store lives under tmp_path. The club names and figures are invented."""
import csv
import io

import httpx
import pytest
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.pdfencrypt import StandardEncryption
from reportlab.pdfgen import canvas
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet

from prophet_agent.datasets import DatasetStore
from prophet_agent.opendata import fetch, pdf_landing
from prophet_agent.research import netguard
from prophet_agent.runtime.runner import ArgumentRejected
from prophet_agent.tools import default_tools
from prophet_agent.tools.opendata_tools import OpenDataContext

PUBLIC = "93.184.216.34"
PDF_URL = "https://league.example.org/reports/table-2026.pdf"
HEADER = ["#", "Club", "P", "W", "D", "L", "Pts"]
STANDINGS = [
    ["#", "Club", "P", "W", "D", "L", "Pts"],
    ["1", "Harbour FC", "10", "7", "2", "1", "23"],
    ["2", "Riverside", "10", "6", "3", "1", "21"],
    ["3", "Northgate", "10", "5", "2", "3", "17"],
]
RESULTS = [
    ["Home", "Goals", "Goals_2", "Away"],
    ["Harbour FC", "2", "1", "Riverside"],
    ["Northgate", "0", "0", "Riverside"],
]


def ruled_pdf(*tables: list[list[str]], prose: str | None = None) -> bytes:
    """One page per table: each table is drawn with a full grid of rules."""
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4)
    flow = []
    for index, rows in enumerate(tables):
        if index:
            flow.append(PageBreak())
        table = Table(rows)
        table.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.5, colors.black)]))
        flow.append(table)
    if prose is not None:
        flow.append(Paragraph(prose, getSampleStyleSheet()["Normal"]))
    doc.build(flow)
    return buf.getvalue()


def unruled_pdf(rows: list[list[str]]) -> bytes:
    """One page, no rules: each cell is drawn at a fixed x position, so the columns are aligned on the page."""
    buf = io.BytesIO()
    pdf = canvas.Canvas(buf, pagesize=A4)
    xs = [50, 80, 240, 290, 330, 370, 410]
    for line, row in enumerate(rows):
        for x, cell in zip(xs, row):
            pdf.drawString(x, 720 - line * 18, cell)
    pdf.showPage()
    pdf.save()
    return buf.getvalue()


@pytest.fixture
def page(monkeypatch):
    """Serve ``body`` at every request and count them."""
    state = {"body": b"", "content_type": "application/pdf", "calls": 0}

    def handler(request):
        state["calls"] += 1
        return httpx.Response(200, headers={"content-type": state["content_type"]}, content=state["body"])

    monkeypatch.setattr(fetch, "open_client",
                        lambda timeout_s: httpx.Client(transport=httpx.MockTransport(handler), follow_redirects=False))
    monkeypatch.setattr(netguard, "_default_resolver", lambda host, port: [PUBLIC])
    return state


class Mission:
    def __init__(self, root):
        self.store = DatasetStore(root / "datasets")
        self.ids: list[str] = []

    def download(self, url=PDF_URL, **args):
        tool = default_tools(opendata=OpenDataContext(store=self.store, attach=self.ids.append))["download_web_data"]
        return tool.fn({"url": url, **args}, "k")

    def csv_rows(self, out: dict) -> list[list[str]]:
        return list(csv.reader(self.store.path(out["dataset_id"]).read_text(encoding="utf-8").splitlines()))


@pytest.fixture
def mission(tmp_path):
    return Mission(tmp_path)


def test_a_ruled_table_lands_its_values_with_the_lines_strategy(page, mission):
    page["body"] = ruled_pdf(STANDINGS)
    out = mission.download()
    assert mission.csv_rows(out) == [HEADER, ["1", "Harbour FC", "10", "7", "2", "1", "23"],
                                     ["2", "Riverside", "10", "6", "3", "1", "21"],
                                     ["3", "Northgate", "10", "5", "2", "3", "17"]]
    assert out["row_count"] == 3
    extraction = out["extraction"]
    assert extraction["method"] == "pdf_table" and extraction["strategy"] == "lines"
    assert extraction["locator"] == "page1.table1" and extraction["page"] == 1
    assert extraction["row_count"] == 3 and extraction["column_count"] == 7


def test_an_unruled_aligned_table_is_read_with_the_text_strategy(page, mission):
    page["body"] = unruled_pdf(STANDINGS)
    out = mission.download()
    rows = mission.csv_rows(out)
    assert rows[0] == HEADER
    assert rows[1][1] == "Harbour FC" and rows[1][-1] == "23"
    assert len(rows) - 1 == 3
    assert out["extraction"]["strategy"] == "text" and out["extraction"]["locator"] == "page1.table1"


def test_the_provenance_records_the_pdf_page_table_and_counts(page, mission):
    page["body"] = ruled_pdf(STANDINGS)
    out = mission.download()
    provenance = mission.store.get(out["dataset_id"])["provenance"]
    assert provenance["url"] == PDF_URL and provenance["pdf"] is True and provenance["format"] == "csv"
    assert provenance["content_type"] == "application/pdf" and provenance["series_bound"] is False
    assert provenance["extraction"]["page"] == 1 and provenance["extraction"]["strategy"] == "lines"
    assert len(provenance["page_sha256"]) == 64


def test_the_best_table_wins_and_the_others_are_listed(page, mission):
    page["body"] = ruled_pdf(RESULTS, STANDINGS)
    out = mission.download()
    assert out["extraction"]["locator"] == "page2.table1" and out["row_count"] == 3
    others = out["other_tables"]
    assert [(other["locator"], other["row_count"]) for other in others] == [("page1.table1", 2)]
    assert others[0]["header"] == ["Home", "Goals", "Goals_2", "Away"] and others[0]["index"] == 0


def test_a_table_with_no_numeric_column_is_not_a_candidate(page, mission):
    page["body"] = ruled_pdf([["Date", "Venue"], ["2026-03-01", "Harbour Park"], ["2026-03-08", "Riverside Ground"]],
                             STANDINGS)
    out = mission.download()
    assert out["extraction"]["locator"] == "page2.table1"
    assert out["extraction"]["candidate_count"] == 1 and out["other_tables"] == []


def test_the_table_argument_picks_by_index_or_by_page_locator(page, mission):
    page["body"] = ruled_pdf(RESULTS, STANDINGS)
    by_index = mission.download(table=0)
    assert by_index["row_count"] == 2 and mission.csv_rows(by_index)[0] == ["Home", "Goals", "Goals_2", "Away"]
    by_locator = mission.download(table="page2.table1")
    assert by_locator["row_count"] == 3 and by_locator["dataset_id"] != by_index["dataset_id"]
    assert page["calls"] == 2, "a different table request is a different download"


def test_an_unknown_table_locator_is_refused_with_the_offer(page, mission):
    page["body"] = ruled_pdf(STANDINGS)
    with pytest.raises(ValueError, match="page1.table1"):
        mission.download(table="page9.table1")


def test_the_same_pdf_and_table_request_return_the_stored_copy(page, mission):
    page["body"] = ruled_pdf(STANDINGS)
    first = mission.download()
    second = mission.download()
    assert second["dataset_id"] == first["dataset_id"] and page["calls"] == 1


def test_a_pdf_with_no_table_is_refused_with_a_clear_error(page, mission):
    page["body"] = ruled_pdf(prose="The committee met in March and reviewed the season. No figures are given here.")
    with pytest.raises(ValueError, match="has no data table with numbers"):
        mission.download()
    assert mission.store.root.exists() is False or list(mission.store.root.iterdir()) == []


def test_an_encrypted_pdf_is_refused_with_a_clear_error(page, mission):
    buf = io.BytesIO()
    pdf = canvas.Canvas(buf, pagesize=A4, encrypt=StandardEncryption("userpw", "ownerpw"))
    pdf.drawString(50, 700, "Harbour FC 23")
    pdf.save()
    page["body"] = buf.getvalue()
    with pytest.raises(ValueError, match="encrypted and needs a password"):
        mission.download()


def test_a_broken_pdf_is_refused_with_a_clear_error(page, mission):
    page["body"] = b"%PDF-1.4\n1 0 obj\n<< /Type /Catalog garbage"
    with pytest.raises(ValueError, match="damaged or not a PDF"):
        mission.download()


def test_a_body_that_is_not_a_pdf_is_refused_at_the_fetch(page, mission):
    page["body"] = b"<html>not a pdf</html>"
    with pytest.raises(fetch.DownloadError, match="starts with %PDF-"):
        mission.download()


def test_the_name_guard_refuses_a_pdf_whose_columns_name_another_ticker(page, mission):
    page["body"] = ruled_pdf([["Date", "AAPL.Close", "AAPL.Open"], ["2026-03-01", "180.2", "179.0"],
                              ["2026-03-02", "181.4", "180.1"]])
    with pytest.raises(ValueError, match="names NVDA"):
        mission.download(name="NVDA_daily.csv")
    assert mission.ids == [], "the refused table is not mounted"


def test_pdf_landing_direct_scan_reports_pages_and_strategy():
    scan = pdf_landing.candidates(ruled_pdf(RESULTS, STANDINGS))
    assert scan["pages_total"] == 2 and scan["pages_read"] == 2 and scan["pages_truncated"] is False
    assert [candidate["strategy"] for candidate in scan["candidates"]] == ["lines", "lines"]
