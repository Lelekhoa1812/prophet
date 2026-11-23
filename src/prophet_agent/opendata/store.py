"""Download one open-data source into the dataset store, with its provenance and a preview.

Motivation vs Logic
Motivation: a forecast is only as good as the record of where its series came from. Each stored download must say which
source, URL, parameters, time and bytes it came from, and asking for the same download again should return the stored
copy, not fetch it a second time.
Logic: the parameters are checked first (sources.normalise), the bytes come from fetch.download, and the preview is parsed
before anything is written, so a body that is not a table is never stored. The store is searched by source key and a
digest of the canonical parameters, which makes a repeat call idempotent.
"""
from __future__ import annotations

import csv
import hashlib
import json
from typing import Any

from prophet_agent.datasets import DatasetStore
from prophet_agent.opendata import fetch, sources

PREVIEW_ROWS = 5


def params_digest(canonical: dict) -> str:
    """sha256 of the canonical parameters, the same whatever order or spelling the caller used for equal values."""
    return hashlib.sha256(json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _format_of(canonical: dict) -> str:
    return "json" if canonical.get("format") == "json" else "csv"


def _is_column_block(value: Any) -> bool:
    if not isinstance(value, dict) or not value:
        return False
    if not all(isinstance(column, list) for column in value.values()):
        return False
    return len({len(column) for column in value.values()}) == 1


def _json_table(text: str) -> tuple[list, int, list]:
    data = json.loads(text)
    if isinstance(data, list) and data and all(isinstance(record, dict) for record in data):
        columns = list(dict.fromkeys(key for record in data for key in record))
        rows = [[record.get(column) for column in columns] for record in data]
        return columns, len(rows), rows[:PREVIEW_ROWS]
    if isinstance(data, dict):
        # Open-Meteo nests its table under "daily" or "hourly"; other blocks of equal-length lists are taken in order.
        blocks = sorted(((key, value) for key, value in data.items() if _is_column_block(value)),
                        key=lambda item: item[0] not in ("daily", "hourly"))
        block = blocks[0][1] if blocks else (data if _is_column_block(data) else None)
        if block is not None:
            columns = list(block)
            count = len(next(iter(block.values())))
            rows = [[block[column][i] for column in columns] for i in range(min(count, PREVIEW_ROWS))]
            return columns, count, rows
    raise ValueError("the JSON does not hold a table of equal-length columns or a list of records")


def _csv_table(text: str) -> tuple[list, int, list]:
    # Lines starting with '#' are comments. Open-Meteo prints a location block before its table, marked by a
    # 'latitude' first cell; the table then starts at the first row whose first cell is 'time' or 'date'.
    lines = [line for line in text.splitlines() if not line.startswith("#")]
    rows = [row for row in csv.reader(lines) if any(cell.strip() for cell in row)]
    if not rows:
        raise ValueError("the CSV has no rows")
    start = 0
    if rows[0][0].strip().lower() == "latitude":
        headers = [index for index, row in enumerate(rows) if row[0].strip().lower() in ("time", "date")]
        if not headers:
            raise ValueError("no time or date header follows the location block")
        start = headers[0]
    columns = [cell.strip() for cell in rows[start]]
    if not any(columns):
        raise ValueError("the CSV header row is empty")
    data = rows[start + 1:]
    return columns, len(data), [list(row) for row in data[:PREVIEW_ROWS]]


def split_csv(text: str) -> tuple[str, str]:
    """Split a CSV text into (preamble, table). Open-Meteo puts a 'latitude' location block before its table; the table
    starts at the first 'time' or 'date' row. A text without that block is returned whole, with an empty preamble."""
    lines = text.splitlines(keepends=True)
    first = next((line for line in lines if line.strip() and not line.startswith("#")), None)
    if first is None or first.split(",")[0].strip().lower() != "latitude":
        return "", text
    for index, line in enumerate(lines):
        if not line.startswith("#") and line.split(",")[0].strip().lower() in ("time", "date"):
            return "".join(lines[:index]), "".join(lines[index:])
    raise ValueError("no time or date header follows the location block")


def preview(body: bytes, fmt: str) -> dict:
    """columns, the first five rows and the total row count of a CSV or JSON body. Raises ValueError if it is not a table."""
    try:
        text = body.decode("utf-8-sig")
    except UnicodeDecodeError:
        raise ValueError("the body is not UTF-8 text") from None
    label = "JSON" if fmt == "json" else "CSV"
    try:
        columns, count, rows = _json_table(text) if fmt == "json" else _csv_table(text)
    except (json.JSONDecodeError, csv.Error, RecursionError) as exc:
        raise ValueError(f"the body does not parse as a {label} table ({type(exc).__name__})") from None
    return {"columns": columns, "rows": rows, "row_count": count}


def _find_existing(datasets: DatasetStore, source_key: str, digest: str) -> dict | None:
    if not datasets.root.is_dir():
        return None
    matches = []
    for folder in datasets.root.iterdir():
        try:
            meta = datasets.get(folder.name)
        except KeyError:
            continue
        provenance = meta.get("provenance") or {}
        if provenance.get("source") == source_key and provenance.get("params_sha256") == digest:
            matches.append(meta)
    return max(matches, key=lambda meta: meta["created"]) if matches else None


def download_to_dataset(source_key: str, params: dict, datasets: DatasetStore, *, name: str | None = None) -> dict:
    """Download one source and store it with its provenance. Returns the stored meta plus a 'preview' of columns, the
    first five rows and row_count. If the same source and parameters are already stored, returns that dataset without
    downloading again."""
    url = sources.build_url(source_key, params)          # validates every parameter first
    canonical = sources.normalise(source_key, params)
    digest = params_digest(canonical)
    fmt = _format_of(canonical)
    existing = _find_existing(datasets, source_key, digest)
    if existing is not None:
        body = datasets.path(existing["id"]).read_bytes()
        return dict(existing, preview=preview(body, fmt))
    fetched = fetch.download(url)
    raw = fetched["bytes"]
    table = preview(raw, fmt)     # nothing is stored unless the body parses as the table it claims to be
    # The stored file is the table alone, so a reader of the file sees the header first. The location block that came
    # before it is kept as text in provenance, and the hash of the raw download is kept too.
    preamble, body = "", raw
    if fmt == "csv":
        preamble, table_text = split_csv(raw.decode("utf-8-sig"))
        body = table_text.encode("utf-8")
    provenance = {
        "source": source_key,
        "url": url,
        "params": canonical,
        "params_sha256": digest,
        "retrieved_at": fetched["retrieved_at"],
        "final_url": fetched["final_url"],
        "content_type": fetched["content_type"],
        "sha256": hashlib.sha256(body).hexdigest(),
        "bytes": len(body),
        "raw_sha256": hashlib.sha256(raw).hexdigest(),
        "raw_bytes": len(raw),
        "preamble": preamble,
        "licence_key": sources.get(source_key).licence_key,
    }
    meta = datasets.add(name or f"{source_key}.{fmt}", body, provenance=provenance)
    return dict(meta, preview=table)
