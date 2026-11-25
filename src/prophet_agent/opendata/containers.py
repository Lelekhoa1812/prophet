"""Turn a ZIP or XLSX download into one CSV table. Never unpacks onto disk.

Motivation vs Logic
Motivation: several publishers ship a series inside a ZIP or a workbook. The store can keep a table, and it should not
keep the archive.
Logic: a ZIP is read in memory. Names that leave the archive, encrypted members, and members that expand past the
download cap are refused. An EIA bulk file is JSON lines, and only the requested series id is kept. A workbook keeps
the first sheet, as CSV.
"""
from __future__ import annotations

import csv
import io
import json
from pathlib import Path
from zipfile import ZipFile

from openpyxl import load_workbook

from prophet_agent.opendata.fetch import DEFAULT_MAX_BYTES

MAX_MEMBERS = 30
MAX_RATIO = 20
MAX_ROWS = 100_000
MAX_COLS = 64
TEXT_SUFFIXES = {".txt", ".csv", ".json"}


class ContainerError(ValueError):
    """The archive or workbook cannot be turned into one table."""


def to_csv(body: bytes, kind: str, series_id: str | None = None) -> bytes:
    """CSV bytes for a zip or xlsx body. ``series_id`` selects one JSON-lines series inside a ZIP."""
    if kind == "xlsx":
        return _xlsx_csv(body)
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


def _xlsx_csv(body: bytes) -> bytes:
    try:
        book = load_workbook(io.BytesIO(body), read_only=True, data_only=True)
    except Exception:
        raise ContainerError("the body is not an XLSX workbook") from None
    try:
        sheet = book.worksheets[0]
        rows = []
        for index, row in enumerate(sheet.iter_rows(values_only=True)):
            if index >= MAX_ROWS:
                raise ContainerError("the sheet has too many rows")
            cells = list(row[:MAX_COLS])
            if any(cell is not None and str(cell).strip() for cell in cells):
                rows.append(["" if cell is None else cell for cell in cells])
        if not rows:
            raise ContainerError("the sheet has no rows")
    finally:
        book.close()
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerows(rows)
    return buf.getvalue().encode("utf-8")
