"""HTML tables as CSV candidates. No network.

Motivation vs Logic
Motivation: a page often holds the grid the file download refused. The agent needs that grid as bytes it can check,
not as a sentence a model typed.
Logic: each ``table`` element becomes one candidate when it has a header, one data row, and at least one numeric cell.
Layout tables fail that test and are counted, not returned. Cells copied across a span stay one measurement.
"""
from __future__ import annotations

import csv
import hashlib
import io
import re

from bs4 import BeautifulSoup

MAX_TABLES = 4
MAX_SCAN = 12
MAX_ROWS = 2000
MAX_COLS = 32
_NUMBER = re.compile(r"^[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?%?$")


def html_tables(html: str, *, base_url: str = "") -> dict:
    """Candidates from ``html``. ``base_url`` is recorded on each candidate and is not fetched."""
    if not html or not str(html).strip():
        return {"candidates": [], "skipped_layout": 0}
    soup = BeautifulSoup(str(html), "lxml")
    candidates = []
    skipped = 0
    for index, tag in enumerate(soup.find_all("table")):
        if index >= MAX_SCAN or len(candidates) >= MAX_TABLES:
            break
        grid = _grid(tag)
        if grid is None:
            skipped += 1
            continue
        header, rows = grid
        if not _has_number(rows):
            skipped += 1
            continue
        text = _csv(header, rows)
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        candidates.append({
            "index": len(candidates),
            "columns": header,
            "row_count": len(rows),
            "sha256": digest,
            "csv": text,
            "preview_rows": rows[:5],
            "final_url": base_url,
            "locator": f"table[{index}]",
        })
    return {"candidates": candidates, "skipped_layout": skipped}


def _grid(tag) -> tuple[list[str], list[list[str]]] | None:
    raw = []
    for tr in tag.find_all("tr"):
        cells = []
        for cell in tr.find_all(["th", "td"]):
            text = " ".join(cell.get_text(" ", strip=True).split())
            span = cell.get("colspan") or 1
            try:
                span = max(1, min(int(span), 8))
            except (TypeError, ValueError):
                span = 1
            cells.extend([text] * span)
        if any(cells):
            raw.append(cells[:MAX_COLS])
        if len(raw) > MAX_ROWS + 1:
            break
    if len(raw) < 2:
        return None
    width = max(len(row) for row in raw)
    if width < 2:
        return None
    header = [_clean_header(cell, i) for i, cell in enumerate(raw[0] + [""] * (width - len(raw[0])))]
    rows = []
    for row in raw[1:]:
        padded = row + [""] * (width - len(row))
        if any(cell.strip() for cell in padded):
            rows.append(padded[:width])
    if not rows:
        return None
    return header[:width], rows


def _clean_header(cell: str, index: int) -> str:
    text = cell.strip()
    return text or f"column_{index + 1}"


def _has_number(rows: list[list[str]]) -> bool:
    for row in rows[:50]:
        for cell in row:
            if _NUMBER.match(cell.replace(" ", "")):
                return True
    return False


def _csv(header: list[str], rows: list[list[str]]) -> str:
    buf = io.StringIO()
    writer = csv.writer(buf, lineterminator="\n")
    writer.writerow(header)
    writer.writerows(rows)
    return buf.getvalue()
