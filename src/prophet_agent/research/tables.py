"""HTML tables as CSV candidates. No network.

Motivation vs Logic
Motivation: a page often holds the grid the file download refused. The agent needs that grid as bytes it can check,
not as a sentence a model typed. A grid that is built from the markup naively shifts every cell after a span, drops a
header that sits in a thead, and keeps a nested layout table as if it were data. Each of those produced a short or
wrong table that still passed the checks.
Logic: each ``table`` element becomes one candidate when it has a header, one data row, and at least one numeric cell.
Cells are placed in a grid that honours rowspan and colspan: a spanning cell fills the positions it covers, and later
cells in the row move to the next free position, so no cell shifts into a span. A header is the thead rows, else the
leading rows of th cells, else the first row; multi-row headers are joined per column with " / ". A table that holds a
table is a page layout and is counted, not returned. A body that is more than 90% empty cells is a layout and is
counted too. Layout tables fail these tests and are counted, not returned. Cells copied across a span stay one
measurement. A league table (rank, club, played, points, text form, signed goal difference) passes: its numbers are plain.
"""
from __future__ import annotations

import csv
import hashlib
import io
import re

from bs4 import BeautifulSoup

MAX_TABLES = 12
MAX_SCAN = 60
MAX_ROWS = 2000
MAX_COLS = 32
MAX_COLSPAN = 8
MAX_ROWSPAN = 50
SPARSE = 0.9                # a body with more empty cells than this share is a layout, not a data table
HEADER_SEPARATOR = " / "
PROSE_CELL = 200            # a cell longer than this counts as prose
_NUMBER = re.compile(r"^[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?%?$")

# One parsed row: (sits in a thead, cells). One cell: (text, colspan, rowspan, is a th, scope attribute).
_Cell = tuple[str, int, int, bool, str]


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
        if tag.find("table") is not None:          # a table that holds a table is a page layout
            skipped += 1
            continue
        grid = _grid(tag)
        if grid is None:
            skipped += 1
            continue
        header, rows = grid
        if not _has_number(rows) or _is_prose(rows):
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
    """Header and body of one table, or None when the table is too small, too sparse or has no body.

    Motivation vs Logic
    Motivation: the header and the body must come from the same grid, or a rowspan header would push the body out of line.
    Logic: rows are read from this table only (nested rows are excluded), placed on the grid, then split at the header count.
    """
    rows = _rows(tag)
    if sum(1 for _, cells in rows if any(cell[0] for cell in cells)) < 2:
        return None
    grid, width = _place(rows)
    if width < 2:
        return None
    count = _header_count(rows)
    header_rows, body = grid[:count], grid[count:]
    if _is_sparse(body):             # the shape is judged on the whole body, spacer rows included
        return None
    header = [_join_header(header_rows, column) for column in range(width)]
    rows_out = [row for row in body if any(cell.strip() for cell in row)][:MAX_ROWS]
    if not rows_out:
        return None
    return header, rows_out


def _rows(tag) -> list[tuple[bool, list[_Cell]]]:
    """Rows of ``tag`` itself. A row that belongs to a nested table is never read, so nested text cannot reach the grid."""
    rows: list[tuple[bool, list[_Cell]]] = []
    for tr in tag.find_all("tr"):
        if tr.find_parent("table") is not tag:
            continue
        cells: list[_Cell] = []
        for cell in tr.find_all(["th", "td"], recursive=False):
            text = " ".join(cell.get_text(" ", strip=True).split())
            scope = str(cell.get("scope") or "").strip().lower()
            cells.append((text, _span(cell.get("colspan"), MAX_COLSPAN), _span(cell.get("rowspan"), MAX_ROWSPAN),
                          cell.name == "th", scope))
        in_head = tr.parent is not None and tr.parent.name == "thead"
        rows.append((in_head, cells))            # empty rows are kept: they are part of the shape the gate measures
        if len(rows) > MAX_ROWS + 1:
            break
    return rows


def _span(value, cap: int) -> int:
    try:
        return max(1, min(int(value), cap))
    except (TypeError, ValueError):
        return 1


def _place(rows: list[tuple[bool, list[_Cell]]]) -> tuple[list[list[str]], int]:
    """Place each cell on the grid, honouring rowspan and colspan.

    Motivation vs Logic
    Motivation: a cell that spans rows must not push the cells after it in the lower rows sideways. That shift turned a
    rank into a club name and a points total into a goal difference.
    Logic: a cell takes the first column of its row that no earlier cell or span covers. It fills every position its
    colspan and rowspan cover, with its own text, then the next cell starts after it. Rows beyond the table are clipped.
    """
    height = len(rows)
    taken: set[tuple[int, int]] = set()
    values: dict[tuple[int, int], str] = {}
    for r, (_, cells) in enumerate(rows):
        c = 0
        for text, colspan, rowspan, _, _ in cells:
            while (r, c) in taken and c < MAX_COLS:
                c += 1
            if c >= MAX_COLS:
                break
            for dr in range(rowspan):
                if r + dr >= height:
                    break
                for dc in range(colspan):
                    if c + dc >= MAX_COLS:
                        break
                    taken.add((r + dr, c + dc))
                    values[(r + dr, c + dc)] = text
            c += colspan
    width = max((column for _, column in taken), default=-1) + 1
    grid = [[values.get((r, column), "") for column in range(width)] for r in range(height)]
    return grid, width


def _header_count(rows: list[tuple[bool, list[_Cell]]]) -> int:
    """How many leading rows are the header.

    Motivation vs Logic
    Motivation: a multi-row header that is not recognised leaves its second row as a data row, and the table is short by one.
    Logic: thead rows are the header. Otherwise leading rows whose cells are all th (and not row headers, scope=row) are
    the header. Otherwise the first row is the header, as before. At least one body row always remains.
    """
    start = next((i for i, (_, cells) in enumerate(rows) if any(cell[0] for cell in cells)), 0)
    count = start
    while count < len(rows) and rows[count][0]:
        count += 1
    if count == start:
        while count < len(rows) and _is_header_row(rows[count][1]):
            count += 1
    return min(max(count, start + 1), len(rows) - 1)


def _is_header_row(cells: list[_Cell]) -> bool:
    return bool(cells) and all(is_th and scope != "row" for _, _, _, is_th, scope in cells)


def _join_header(header_rows: list[list[str]], column: int) -> str:
    """One column name from the header rows above it. Repeated neighbours (from a span) are kept once."""
    parts: list[str] = []
    for row in header_rows:
        text = row[column].strip() if column < len(row) else ""
        if text and (not parts or parts[-1] != text):
            parts.append(text)
    return HEADER_SEPARATOR.join(parts) if parts else f"column_{column + 1}"


def _is_sparse(rows: list[list[str]]) -> bool:
    """True when more than SPARSE of the body cells are empty. Integer form avoids float rounding at the boundary."""
    total = sum(len(row) for row in rows)
    if total == 0:
        return True
    empty = sum(1 for row in rows for cell in row if not cell.strip())
    return empty * 10 > total * 9  # SPARSE is 0.9; the integer form keeps the boundary exact


def _has_number(rows: list[list[str]]) -> bool:
    for row in rows[:50]:
        for cell in row:
            if _NUMBER.match(cell.replace(" ", "")):
                return True
    return False


def _is_prose(rows: list[list[str]]) -> bool:
    """True when most non-empty cells are running text rather than values: a layout built from paragraphs."""
    cells = [cell for row in rows[:50] for cell in row if cell.strip()]
    if not cells:
        return False
    return sum(1 for cell in cells if len(cell) > PROSE_CELL) * 2 > len(cells)


def _csv(header: list[str], rows: list[list[str]]) -> str:
    buf = io.StringIO()
    writer = csv.writer(buf, lineterminator="\n")
    writer.writerow(header)
    writer.writerows(rows)
    return buf.getvalue()
