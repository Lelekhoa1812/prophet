"""Turn a ZIP or XLSX download into one CSV table. Never unpacks onto disk.

Motivation vs Logic
Motivation: several publishers ship a series inside a ZIP or a workbook. The store can keep a table, and it should not
keep the archive.
Logic: a ZIP is read in memory. Names that leave the archive, encrypted members, and members that expand past the
download cap are refused. An EIA bulk file is JSON lines, and only the requested series id is kept. A workbook (XLSX, or
legacy XLS read by calamine) keeps the sheet with the most numeric rows, as CSV: EIA workbooks open on a Contents sheet.
"""
from __future__ import annotations

import csv
import io
import json
import re
from datetime import date
from pathlib import Path
from zipfile import ZipFile

from openpyxl import load_workbook

from prophet_agent.opendata.fetch import DEFAULT_MAX_BYTES

MAX_MEMBERS = 30
MAX_RATIO = 20
MAX_ROWS = 100_000
# Live GOLD (round 20261128-110827): the World Bank monthly workbook has 71 columns and gold, platinum and silver are the last
# ones; a 64-column cap cut them off and the mission forecast a $/mt column as gold.
MAX_COLS = 256
MAX_SHEETS = 40
TEXT_SUFFIXES = {".txt", ".csv", ".json"}


class ContainerError(ValueError):
    """The archive or workbook cannot be turned into one table."""


def to_csv(body: bytes, kind: str, series_id: str | None = None) -> bytes:
    """CSV bytes for a zip or xlsx body. ``series_id`` selects one JSON-lines series inside a ZIP."""
    if kind == "xlsx":
        return _xlsx_csv(body)
    if kind == "xls":
        return _xls_csv(body)
    if kind == "zip":
        return _zip_csv(body, series_id)
    raise ContainerError(f"unsupported container {kind}")


def _zip_csv(body: bytes, series_id: str | None) -> bytes:
    try:
        archive = ZipFile(io.BytesIO(body))
    except Exception:
        raise ContainerError("the body is not a ZIP archive") from None
    with archive:
        members = [info for info in archive.infolist() if not info.is_dir()]
        if not members or len(members) > MAX_MEMBERS:
            raise ContainerError("the archive does not hold one small set of files")
        texts = []
        for info in members:
            _refuse_unsafe(info)
            if Path(info.filename).suffix.lower() not in TEXT_SUFFIXES:
                continue
            raw = _read_member(archive, info)
            try:
                text = raw.decode("utf-8-sig")
            except UnicodeDecodeError:
                raise ContainerError("a file in the archive is not UTF-8 text") from None
            texts.append(text)
        if not texts:
            raise ContainerError("the archive has no CSV or JSON file")
        if series_id:
            return _series_csv(texts, series_id)
        if len(texts) != 1:
            raise ContainerError("the archive has more than one table; name the series id")
        return texts[0].encode("utf-8")


def _refuse_unsafe(info) -> None:
    name = info.filename
    if name.startswith("/") or "\\" in name or ".." in Path(name).parts:
        raise ContainerError("the archive has an unsafe file name")
    if info.flag_bits & 0x1:
        raise ContainerError("the archive is encrypted")
    if info.file_size > DEFAULT_MAX_BYTES:
        raise ContainerError("a file in the archive is larger than the download limit")
    if info.compress_size and info.file_size / info.compress_size > MAX_RATIO:
        raise ContainerError("a file in the archive expands too far")


def _read_member(archive: ZipFile, info) -> bytes:
    with archive.open(info) as handle:
        raw = handle.read(DEFAULT_MAX_BYTES + 1)
    if len(raw) > DEFAULT_MAX_BYTES:
        raise ContainerError("a file in the archive is larger than the download limit")
    return raw


def _series_csv(texts: list[str], series_id: str) -> bytes:
    rows = []
    for text in texts:
        for line in text.splitlines():
            stripped = line.strip()
            if not stripped or stripped[0] != "{":
                continue
            try:
                obj = json.loads(stripped)
            except json.JSONDecodeError:
                continue
            if not isinstance(obj, dict) or obj.get("series_id") != series_id:
                continue
            data = obj.get("data")
            if not isinstance(data, list):
                raise ContainerError("the series has no data list")
            for pair in data:
                if isinstance(pair, list) and len(pair) >= 2:
                    value = "" if pair[1] is None else pair[1]
                    rows.append((pair[0], value))
            if not rows:
                raise ContainerError("the series has no observations")
            buf = io.StringIO()
            writer = csv.writer(buf)
            writer.writerow(["period", "value"])
            writer.writerows(rows)
            return buf.getvalue().encode("utf-8")
    raise ContainerError(f"the archive has no series {series_id}")


def _numeric_rows(rows: list) -> int:
    return sum(1 for row in rows if any(isinstance(c, (int, float, date)) and not isinstance(c, bool) for c in row))


def _best_sheet(sheets: list[list]) -> list:
    """The sheet with the most rows holding a number or a date; the first of equals. A Contents or notes sheet loses."""
    best = max(range(len(sheets)), key=lambda i: (_numeric_rows(sheets[i]), -i))
    return sheets[best]


def _kept(rows_iter) -> list:
    rows = []
    for index, row in enumerate(rows_iter):
        if index >= MAX_ROWS:
            raise ContainerError("the sheet has too many rows")
        cells = list(row[:MAX_COLS])
        if any(cell is not None and str(cell).strip() for cell in cells):
            rows.append(["" if cell is None else cell for cell in cells])
    return rows


def _from_header(rows: list) -> list:
    """The rows from the header on: the text row just above the first row that holds a number or a date. Title and
    navigation rows above it ("Back to Contents", "Sourcekey") are dropped; a sheet with no such row is kept whole."""
    def numeric(row):
        return any(isinstance(c, (int, float, date)) and not isinstance(c, bool) for c in row)
    first = next((i for i, row in enumerate(rows) if numeric(row)), None)
    if not first:
        return rows
    head = rows[first - 1]
    if first >= 2 and _units_row(head) and not numeric(rows[first - 2]):
        # A units row under a names row (the World Bank workbook: "Gold" above "($/troy oz)"). Live GOLD (round
        # 20261128-110827) kept only the units row, so every column was named "($/mt)" or "($/kg)" and the mission took a
        # $/mt column for gold. The two rows are joined into one header: "Gold ($/troy oz)".
        names = rows[first - 2]
        width = max(len(head), len(names))
        joined = [" ".join(str(x).strip() for x in (names[k] if k < len(names) else "", head[k] if k < len(head) else "")
                           if str(x).strip()) for k in range(width)]
        return [joined] + rows[first:]
    return rows[first - 1:]


_UNIT = re.compile(r"^\(.{1,40}\)$")


def _units_row(row: list) -> bool:
    """True when most filled cells of the row are units in brackets, such as "($/bbl)" or "(2010=100)"."""
    filled = [str(c).strip() for c in row if str(c).strip()]
    return len(filled) >= 2 and sum(1 for c in filled if _UNIT.match(c)) >= 0.6 * len(filled)


def _rows_csv(rows: list) -> bytes:
    rows = _from_header(rows)
    if not rows:
        raise ContainerError("the sheet has no rows")
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerows(rows)
    return buf.getvalue().encode("utf-8")


def _xlsx_csv(body: bytes) -> bytes:
    try:
        book = load_workbook(io.BytesIO(body), read_only=True, data_only=True)
    except Exception:
        raise ContainerError("the body is not an XLSX workbook") from None
    try:
        sheets = [_kept(sheet.iter_rows(values_only=True)) for sheet in book.worksheets[:MAX_SHEETS]]
    finally:
        book.close()
    if not sheets:
        raise ContainerError("the workbook has no sheets")
    return _rows_csv(_best_sheet(sheets))


def _xls_csv(body: bytes) -> bytes:
    try:
        from python_calamine import CalamineWorkbook
        book = CalamineWorkbook.from_filelike(io.BytesIO(body))
        sheets = [_kept(book.get_sheet_by_name(name).to_python()) for name in book.sheet_names[:MAX_SHEETS]]
    except ContainerError:
        raise
    except Exception:
        raise ContainerError("the body is not an XLS workbook") from None
    if not sheets:
        raise ContainerError("the workbook has no sheets")
    return _rows_csv(_best_sheet(sheets))
