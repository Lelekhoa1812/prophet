"""Land the best data table of a PDF as a CSV dataset. No network: the caller has already fetched the bytes.

Motivation vs Logic
Motivation: statistics agencies and leagues often publish their tables only inside PDFs. The planner needs the grid as a
file the sandbox reads, with its provenance, and it must be able to pick another table of the same document.
Logic: pdfplumber reads each page (up to MAX_PAGES). A page is searched for ruled tables first (the lines strategy); only
when it has none is the text strategy used, which finds aligned columns without rules. Each table is cleaned: multi-line
header rows are joined into one header, tables that are more than 90% empty or hold no numeric column are dropped, and
the rest are scored by html_landing.score (rows times numeric columns). The best wins unless the caller names one by index
or by locator, such as page3.table1. The chosen table is stored as CSV with the page address, the page, the table number,
the strategy and the row and column counts, and a list of the other candidates. An encrypted or damaged file is refused
with a clear message; nothing is stored.
"""
from __future__ import annotations

import csv
import hashlib
import io

from prophet_agent.opendata import html_landing
from prophet_agent.opendata.store import preview

MAX_PAGES = 40
MAX_CANDIDATES = 200
MAX_HEADER_ROWS = 3
MAX_EMPTY_SHARE = 0.9
LIST_LIMIT = html_landing.LIST_LIMIT
STRATEGIES = (
    ("lines", {"vertical_strategy": "lines", "horizontal_strategy": "lines"}),
    ("text", {"vertical_strategy": "text", "horizontal_strategy": "text"}),
)
NO_TABLE = ("the PDF at {url} has no data table with numbers. A scanned PDF has no text layer for tables to be read from. "
            "web_fetch returns the PDF's text, which can be searched but not computed on.")


def _open_pages(body: bytes):
    """A pdfplumber document, or a ValueError that names the reason the PDF cannot be read."""
    import pdfplumber
    try:
        pdf = pdfplumber.open(io.BytesIO(body))
    except Exception as exc:          # pdfminer raises several types for encrypted, truncated and malformed files
        if "password" in str(exc).lower() or type(exc).__name__ in ("PDFPasswordIncorrect",):
            raise ValueError("the PDF is encrypted and needs a password, so its tables were not read") from None
        raise ValueError(f"the PDF cannot be read: it is damaged or not a PDF ({type(exc).__name__})") from None
    return pdf


def _cell(value) -> str:
    return " ".join(str(value or "").split())


def _is_number(cell: str) -> bool:
    try:
        float(cell.replace(",", "").replace("%", "").replace(" ", ""))
    except ValueError:
        return False
    return True


def _header_and_data(grid: list[list[str]]) -> tuple[list[str], list[list[str]]]:
    """Split the grid into a header and data rows. Leading rows whose non-empty cells are mostly text are the header (up
    to MAX_HEADER_ROWS, joined per column, so a two-line header becomes one name). A grid that starts with numbers has no
    header, and the columns are named column_1, column_2 and so on."""
    width = len(grid[0]) if grid else 0
    header_rows: list[list[str]] = []
    for row in grid[:MAX_HEADER_ROWS]:
        if not _mostly_text(row):
            break
        header_rows.append(row)
    if len(header_rows) == len(grid):
        header_rows = header_rows[:1]                  # a grid of text only: its first row is the header
    data = grid[len(header_rows):]
    names = []
    for column in range(width):
        parts = [row[column] for row in header_rows if row[column]]
        names.append(" ".join(parts) if parts else f"column_{column + 1}")
    return names, data


def _mostly_text(row: list[str]) -> bool:
    """More than half of the filled cells are text. A row with exactly half numbers is data, not a header."""
    cells = [cell for cell in row if cell]
    return bool(cells) and sum(_is_number(cell) for cell in cells) * 2 < len(cells)


def _unique(names: list[str]) -> list[str]:
    seen: dict[str, int] = {}
    out = []
    for name in names:
        seen[name] = seen.get(name, 0) + 1
        out.append(name if seen[name] == 1 else f"{name}_{seen[name]}")
    return out


def _clean(raw: list[list]) -> tuple[list[str], list[list[str]]] | None:
    """The header and data of one pdfplumber table, or None when it is mostly empty or has no numeric column."""
    grid = [[_cell(value) for value in row] for row in raw if row and any(_cell(value) for value in row)]
    if len(grid) < 2:
        return None
    width = max(len(row) for row in grid)
    grid = [row + [""] * (width - len(row)) for row in grid]
    cells = [cell for row in grid for cell in row]
    if sum(1 for cell in cells if not cell) / len(cells) > MAX_EMPTY_SHARE:
        return None
    names, data = _header_and_data(grid)
    keep = [index for index in range(width) if names[index] != f"column_{index + 1}" or any(row[index] for row in data)]
    names = _unique([names[index] for index in keep])
    data = [[row[index] for index in keep] for row in data]
    if not data or len(names) == 0:
        return None
    if not any(_numeric_column(data, index) for index in range(len(names))):
        return None
    return names, data


def _numeric_column(data: list[list[str]], index: int) -> bool:
    """At least 80% of the filled cells in the column are numbers, as html_landing counts them."""
    cells = [row[index] for row in data if row[index]]
    return bool(cells) and sum(_is_number(cell) for cell in cells) * 5 >= len(cells) * 4


def _csv_text(names: list[str], data: list[list[str]]) -> str:
    buf = io.StringIO()
    writer = csv.writer(buf, lineterminator="\n")
    writer.writerow(names)
    writer.writerows(data)
    return buf.getvalue()


def _summary(candidate: dict) -> dict:
    return {"index": candidate["index"], "locator": candidate["locator"], "method": candidate["method"],
            "strategy": candidate["strategy"], "page": candidate["page"], "header": candidate["columns"][:12],
            "row_count": candidate["row_count"]}


def candidates(body: bytes) -> dict:
    """Every usable table of the PDF, numbered from 0 in page order, with the page count and whether pages were cut."""
    pdf = _open_pages(body)
    found: list[dict] = []
    try:
        pages_total = len(pdf.pages)
        pages = pdf.pages[:MAX_PAGES]
        for page in pages:
            page_number = page.page_number
            for strategy, settings in STRATEGIES:
                try:
                    tables = page.extract_tables(settings)
                except Exception as exc:
                    raise ValueError(f"the PDF's page {page_number} could not be read ({type(exc).__name__})") from None
                usable = [(number, _clean(raw)) for number, raw in enumerate(tables, start=1)]
                usable = [(number, cleaned) for number, cleaned in usable if cleaned is not None]
                if usable:
                    for number, (names, data) in usable:
                        if len(found) >= MAX_CANDIDATES:
                            break
                        text = _csv_text(names, data)
                        found.append({"index": len(found), "locator": f"page{page_number}.table{number}",
                                      "method": "pdf_table", "strategy": strategy, "page": page_number,
                                      "columns": names, "row_count": len(data), "csv": text})
                    break                              # the text strategy runs only when the lines strategy finds nothing
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(f"the PDF cannot be read: it is damaged or not a PDF ({type(exc).__name__})") from None
    finally:
        pdf.close()
    return {"candidates": found, "pages_total": pages_total, "pages_read": len(pages),
            "pages_truncated": pages_total > len(pages)}


def land_pdf(fetched: dict, *, store, name: str, url: str, licence_key: str, table=None) -> tuple[dict, dict]:
    """Store the chosen table of the fetched PDF as a CSV dataset. Returns (meta, extraction), where meta is the stored
    record with its preview and extraction holds the page, locator, strategy, counts and the other candidates. Raises
    ValueError when the PDF is encrypted, damaged or has no table, or when ``table`` names none of its tables."""
    scan = candidates(fetched["bytes"])
    cands = scan["candidates"]
    if not cands:
        raise ValueError(NO_TABLE.format(url=url))
    chosen = html_landing.choose(cands, table)
    body = html_landing._quoted(chosen["csv"]).encode("utf-8")
    table_info = preview(body, "csv")
    others = [_summary(c) for c in cands if c["index"] != chosen["index"]]
    extraction = {"method": chosen["method"], "strategy": chosen["strategy"], "page": chosen["page"],
                  "locator": chosen["locator"], "row_count": chosen["row_count"], "column_count": len(chosen["columns"]),
                  "candidate_count": len(cands), "pages_total": scan["pages_total"], "pages_read": scan["pages_read"],
                  "pages_truncated": scan["pages_truncated"], "other_tables": others[:LIST_LIMIT]}
    provenance = {"source": "web", "format": "csv", "url": url, "final_url": fetched["final_url"],
                  "content_type": fetched["content_type"], "retrieved_at": fetched["retrieved_at"],
                  "sha256": hashlib.sha256(body).hexdigest(), "bytes": len(body),
                  "page_sha256": hashlib.sha256(fetched["bytes"]).hexdigest(), "licence_key": licence_key,
                  "series_bound": False, "pdf": True,
                  "table_request": None if table is None else str(table), "extraction": extraction}
    meta = dict(store.add(name, body, provenance=provenance), preview=table_info)
    return meta, extraction
