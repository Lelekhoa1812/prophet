"""Panel NGSTORAGE (2026-11-26): EIA's weekly storage history is a legacy .xls whose first sheet is a Contents page. An
Excel content type is a workbook only when the body is one; a CSV that a server labels as Excel stays CSV."""
import io

import httpx
import pytest
from openpyxl import Workbook

from prophet_agent.opendata import containers, fetch

PUBLIC = "93.184.216.34"
URL = "https://www.eia.gov/dnav/ng/hist_xls/NW2_EPG0_SWO_R48_BCFw.xls"


def _resolver(host, port):
    return [PUBLIC]


@pytest.fixture
def serve(monkeypatch):
    def install(body, media):
        def handler(request):
            return httpx.Response(200, headers={"content-type": media}, content=body)
        monkeypatch.setattr(fetch, "open_client",
                            lambda timeout_s: httpx.Client(transport=httpx.MockTransport(handler), follow_redirects=False))
    return install


def test_an_ole_body_labelled_excel_is_kept_as_a_workbook(serve):
    body = fetch.OLE_MAGIC + b"\x00" * 600
    serve(body, "application/vnd.ms-excel")
    got = fetch.download(URL, resolver=_resolver)
    assert got["content_type"] == "application/vnd.ms-excel" and got["bytes"] == body


def test_a_csv_labelled_excel_is_read_as_csv(serve):
    serve(b"date,value\n2026-01-02,3117\n2026-01-09,2850\n2026-01-16,2700\n", "application/vnd.ms-excel")
    got = fetch.download(URL, resolver=_resolver)
    assert got["content_type"] == "text/csv"


def test_an_html_page_labelled_excel_is_refused(serve):
    serve(b"<html><body>Access denied</body></html>", "application/vnd.ms-excel")
    with pytest.raises(fetch.DownloadError, match="Excel workbook or CSV"):
        fetch.download(URL, resolver=_resolver)


def test_a_workbook_keeps_its_data_sheet_not_its_contents_sheet():
    book = Workbook()
    contents = book.active
    contents.title = "Contents"
    for row in (["Workbook Contents"], ["Weekly Lower 48 storage"], ["Worksheet Name", "Description"], ["Data 1", "weekly"]):
        contents.append(row)
    data = book.create_sheet("Data 1")
    data.append(["Sourcekey", "NW2_EPG0_SWO_R48_BCF"])
    data.append(["Date", "Weekly Lower 48 storage (Bcf)"])
    for day, value in ((2, 3117), (9, 2850), (16, 2700)):
        data.append([f"2026-01-{day:02d}", value])
    buf = io.BytesIO()
    book.save(buf)
    text = containers.to_csv(buf.getvalue(), "xlsx").decode()
    assert "3117" in text and "Workbook Contents" not in text


def test_a_body_that_is_not_an_xls_is_refused():
    with pytest.raises(containers.ContainerError, match="not an XLS workbook"):
        containers.to_csv(b"not a workbook", "xls")


def test_a_units_row_is_joined_to_the_names_above_it_and_wide_sheets_keep_their_last_columns():
    # Live GOLD (round 20261128-110827): the World Bank workbook landed with only its units row as the header and was cut at
    # 64 columns, so gold (the 69th) was gone and a "($/mt)" column was forecast as gold.
    book = Workbook()
    sheet = book.active
    sheet.title = "Monthly Prices"
    others = [f"Commodity {k}" for k in range(67)]
    sheet.append(["World Bank Commodity Price Data (The Pink Sheet)"])
    sheet.append(["Updated on November 04, 2026"])
    sheet.append([None, *others, "Gold", "Platinum", "Silver"])
    sheet.append([None, *(["($/mt)"] * 67), "($/troy oz)", "($/troy oz)", "($/troy oz)"])
    sheet.append(["2026M09", *([100.0] * 67), 3986.2, 1500.0, 46.1])
    sheet.append(["2026M10", *([101.0] * 67), 4105.1, 1550.0, 48.9])
    buf = io.BytesIO()
    book.save(buf)
    import csv as _csv
    rows = list(_csv.reader(io.StringIO(containers.to_csv(buf.getvalue(), "xlsx").decode())))
    assert rows[0][68] == "Gold ($/troy oz)" and rows[0][1] == "Commodity 0 ($/mt)"
    assert rows[-1][0] == "2026M10" and rows[-1][68] == "4105.1"
    assert "Updated on" not in ",".join(rows[0])
