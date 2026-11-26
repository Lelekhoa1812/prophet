"""Align accepted sources on the time key. A disagreement stays as two columns.

Motivation vs Logic
Motivation: two sources for one measure can both be real. Picking one in silence hides the other.
Logic: rows join on the time cell. Equal values collapse. Unequal values stay beside each other and the row is marked.
"""
from __future__ import annotations

import csv
import io


def fuse_sources(tables: list[dict], *, time_column: str, measure_column: str, tolerance: str = "exact") -> dict:
    """``tables`` items have ``source_id`` and ``rows`` (list of header-keyed dicts)."""
    if not tables:
        raise ValueError("fusion needs at least one accepted table")
    by_time: dict[str, dict] = {}
    order = []
    for table in tables:
        source = str(table.get("source_id") or "")
        if not source:
            raise ValueError("each table needs a source_id")
        for row in table.get("rows") or []:
            key = str(row.get(time_column, "")).strip()
            if not key:
                continue
            slot = by_time.setdefault(key, {})
            if key not in order:
                order.append(key)
            slot[source] = row.get(measure_column, "")
    rows = []
    conflicts = []
    sources = [str(table["source_id"]) for table in tables]
    for key in order:
        values = by_time[key]
        present = [values[source] for source in sources if source in values]
        conflict = len(present) > 1 and not _all_same(present, tolerance)
        if conflict:
            conflicts.append(key)
        out = {"time": key, "conflict": conflict}
        if not conflict and present:
            out[measure_column] = present[0]
        for source in sources:
            if source in values:
                out[f"{source}__{measure_column}"] = values[source]
        rows.append(out)
    return {"rows": rows, "conflict_times": conflicts, "csv": _csv(rows), "sources": sources,
            "measure_column": measure_column, "origin": "observed"}


def _all_same(values: list, tolerance: str) -> bool:
    first = values[0]
    return all(_same(first, other, tolerance) for other in values[1:])


def _same(left, right, tolerance: str) -> bool:
    if str(left).strip() == str(right).strip():
        return True
    if tolerance == "exact":
        return False
    try:
        a, b, band = float(left), float(right), float(tolerance)
    except (TypeError, ValueError):
        return False
    scale = max(abs(a), abs(b), 1e-12)
    return abs(a - b) / scale <= band


def _csv(rows: list[dict]) -> str:
    if not rows:
        return ""
    fields = list(rows[0])
    for row in rows[1:]:
        for key in row:
            if key not in fields:
                fields.append(key)
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=fields, lineterminator="\n", extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    return buf.getvalue()
