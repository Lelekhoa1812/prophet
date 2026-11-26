"""Land the best data table of an HTML page as a CSV dataset. No network: the caller has already fetched the bytes.

Motivation vs Logic
Motivation: an HTML page often holds the grid the planner needs (a league table, a results list, a price table), and passages
of its text cannot be computed on. The grid must become a file the sandbox reads, with its provenance, and the planner must
be able to pick another table on the same page when the first is wrong.
Logic: candidates come from the markup (html_tables) and from the JSON the page embeds (embedded_tables). Each is scored by
rows times numeric columns, and the best wins unless the caller names one by index or by locator. The chosen table is stored
as CSV with the page address, the locator, the method, the row and column counts, and a short list of the other candidates.
The same page with the same table request returns the stored copy, as the other web downloads do.
"""
from __future__ import annotations

import csv
import hashlib
import io
import re

from prophet_agent.opendata.store import preview
from prophet_agent.research.embedded import embedded_tables
from prophet_agent.research.tables import html_tables

LIST_LIMIT = 12
_NUMBER = re.compile(r"^[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?%?$")
NO_TABLE = ("the page at {url} has no data table with numbers, in its markup or in JSON it embeds. Call web_fetch on the "
            "page, then extract_page_tables on that page. A page that builds its table in the browser may need rendering, "
            "which web_fetch does not do yet.")


def candidates(html: str, base_url: str) -> list[dict]:
    """Markup tables first, then embedded JSON tables, de-duplicated by content and numbered from 0 in that order."""
    markup = html_tables(html, base_url=base_url)["candidates"]
    for candidate in markup:
        candidate.setdefault("method", "html_table")
    combined: list[dict] = []
    seen: set[str] = set()
    for candidate in markup + embedded_tables(html, base_url=base_url)["candidates"]:
        if candidate["sha256"] in seen:
            continue
        seen.add(candidate["sha256"])
        combined.append(dict(candidate, index=len(combined)))
    return combined


def _numeric_columns(text: str) -> int:
    rows = list(csv.reader(io.StringIO(text)))
    if len(rows) < 2:
        return 0
    count = 0
    for column in range(len(rows[0])):
        cells = [row[column].strip().replace(" ", "") for row in rows[1:] if column < len(row) and row[column].strip()]
        if cells and sum(1 for cell in cells if _NUMBER.match(cell)) * 5 >= len(cells) * 4:
            count += 1
    return count


def score(candidate: dict) -> int:
    """Rows times numeric columns. A 20-row table with seven numeric columns outranks a 3-row results list."""
    return candidate["row_count"] * max(1, _numeric_columns(candidate["csv"]))


def choose(cands: list[dict], table) -> dict:
    """The candidate named by ``table`` (an index, or a locator such as table[2] or __NEXT_DATA__:$.props...), or the best."""
    if not cands:
        raise ValueError("the page has no candidate tables")
    if table is None:
        return max(cands, key=lambda candidate: (score(candidate), -candidate["index"]))
    if isinstance(table, int) and not isinstance(table, bool):
        for candidate in cands:
            if candidate["index"] == table:
                return candidate
    elif isinstance(table, str) and table.strip():
        wanted = table.strip()
        for candidate in cands:
            if wanted.isdigit() and candidate["index"] == int(wanted):
                return candidate
            if candidate["locator"] == wanted:
                return candidate
    raise ValueError(f"no table matches {table!r}; the page offers: {_offer(cands)}")


def _offer(cands: list[dict]) -> str:
    return "; ".join(f"{c['index']} {c['locator']} ({c['row_count']} rows, header {', '.join(c['columns'][:8])})"
                     for c in cands[:LIST_LIMIT])


def _summary(candidate: dict) -> dict:
    return {"index": candidate["index"], "locator": candidate["locator"], "method": candidate["method"],
            "header": candidate["columns"][:12], "row_count": candidate["row_count"]}


def _quoted(text: str) -> str:
    """The same CSV with every field quoted. The store's CSV reader treats a line that starts with '#' as a comment, so a
    header such as "#,Club,P" would lose its first row. A quoted field starts with a quote, which is not a comment."""
    buf = io.StringIO()
    writer = csv.writer(buf, lineterminator="\n", quoting=csv.QUOTE_ALL)
    writer.writerows(csv.reader(io.StringIO(text)))
    return buf.getvalue()


def _html_text(body: bytes) -> str:
    try:
        return body.decode("utf-8-sig")
    except UnicodeDecodeError:
        raise ValueError("the page is not UTF-8 text, so its tables cannot be read") from None


def land_html(fetched: dict, *, store, name: str, url: str, licence_key: str, table=None) -> tuple[dict, dict]:
    """Store the chosen table of the fetched page as a CSV dataset. Returns (meta, extraction), where meta is the stored
    record with its preview and extraction holds the method, locator, counts and the other candidates. Raises ValueError
    when the page has no table, or when ``table`` names none of its tables."""
    html = _html_text(fetched["bytes"])
    cands = candidates(html, base_url=fetched["final_url"])
    if not cands:
        raise ValueError(NO_TABLE.format(url=url))
    chosen = choose(cands, table)
    body = _quoted(chosen["csv"]).encode("utf-8")
    table_info = preview(body, "csv")
    others = [_summary(c) for c in cands if c["index"] != chosen["index"]]
    extraction = {"method": chosen["method"], "locator": chosen["locator"], "row_count": chosen["row_count"],
                  "column_count": len(chosen["columns"]), "truncated": chosen.get("truncated", False),
                  "candidate_count": len(cands), "other_tables": others[:LIST_LIMIT]}
    provenance = {"source": "web", "format": "csv", "url": url, "final_url": fetched["final_url"],
                  "content_type": fetched["content_type"], "retrieved_at": fetched["retrieved_at"],
                  "sha256": hashlib.sha256(body).hexdigest(), "bytes": len(body),
                  "page_sha256": hashlib.sha256(fetched["bytes"]).hexdigest(), "licence_key": licence_key,
                  "series_bound": False, "html": True,
                  "table_request": None if table is None else str(table), "extraction": extraction}
    meta = dict(store.add(name, body, provenance=provenance), preview=table_info)
    return meta, extraction
