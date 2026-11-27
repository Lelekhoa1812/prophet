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
from prophet_agent.opendata import containers, fetch, sources
from prophet_agent.opendata.csvlines import comment_flags, table_lines

PREVIEW_ROWS = 5


def params_digest(canonical: dict) -> str:
    """sha256 of the canonical parameters, the same whatever order or spelling the caller used for equal values."""
    return hashlib.sha256(json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _format_of(source_key: str, canonical: dict) -> str:
    source = sources.get(source_key)
    if canonical.get("format") in ("json", "csv"):
        return canonical["format"]
    if source.body_kind == "json":
        return "json"
    if source.body_kind == "text":
        return "text"
    return "csv"


def _is_column_block(value: Any) -> bool:
    if not isinstance(value, dict) or not value:
        return False
    if not all(isinstance(column, list) for column in value.values()):
        return False
    return len({len(column) for column in value.values()}) == 1


def _flatten_record(record: dict) -> dict:
    flat = {}
    for key, value in record.items():
        if isinstance(value, dict):
            flat[key] = value.get("id", value.get("value"))
        elif isinstance(value, (str, int, float)) or value is None:
            flat[key] = value
    return flat


def _is_match_list(value) -> bool:
    return (isinstance(value, list) and bool(value)
            and all(isinstance(m, dict) and isinstance(m.get("team1"), str) and isinstance(m.get("team2"), str) for m in value))


def _match_record(match: dict) -> dict:
    """One football.json match as a flat record. The score is {"ft": [h, a]} or a bare [h, a]; a fixture has none."""
    score = match.get("score")
    ft = score.get("ft") if isinstance(score, dict) else (score if isinstance(score, list) else None)
    played = isinstance(ft, list) and len(ft) == 2
    return {"date": match.get("date"), "round": match.get("round"), "team1": match["team1"], "team2": match["team2"],
            "home_goals": ft[0] if played else None, "away_goals": ft[1] if played else None}


def _json_table(text: str) -> tuple[list, int, list]:
    data = json.loads(text)
    # The World Bank indicator call returns [page metadata, list of observations].
    if (isinstance(data, list) and len(data) == 2 and isinstance(data[0], dict) and "page" in data[0]
            and isinstance(data[1], list)):
        data = [_flatten_record(record) for record in data[1] if isinstance(record, dict)]
    if isinstance(data, list) and data and all(isinstance(record, dict) for record in data):
        columns = list(dict.fromkeys(key for record in data for key in record))
        rows = [[record.get(column) for column in columns] for record in data]
        return columns, len(rows), rows[:PREVIEW_ROWS]
    if isinstance(data, dict) and _is_match_list(data.get("matches")):
        # openfootball football.json: one record per match, the full-time score split into two columns.
        records = [_match_record(match) for match in data["matches"]]
        columns = list(records[0])
        rows = [[record[column] for column in columns] for record in records]
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


def _filled(row: list) -> list:
    return [cell for cell in row if str(cell).strip()]


def _table_start(rows: list) -> int:
    """The header row. A title block of single-cell rows is skipped. An Open-Meteo location block starts at latitude
    and the table then starts at the first time or date row."""
    start = 0
    while start < len(rows) and len(_filled(rows[start])) < 2:
        start += 1
    if start >= len(rows):
        raise ValueError("the CSV has no header row")
    if rows[start][0].strip().lower() == "latitude":
        headers = [index for index, row in enumerate(rows) if row and row[0].strip().lower() in ("time", "date")]
        if not headers:
            raise ValueError("no time or date header follows the location block")
        return headers[0]
    return start


def _csv_table(text: str) -> tuple[list, int, list]:
    if text.lstrip()[:1] in ("{", "["):
        # Without this, a JSON body parses as a one-column CSV and is stored as a table. Live, 2026-11-24: a request with
        # no format came back as JSON.
        raise ValueError("the body is JSON, not CSV")
    # A '#' line is a comment unless it is the header (csvlines). Open-Meteo prints a location block before its table,
    # marked by a 'latitude' first cell; the table then starts at the first row whose first cell is 'time' or 'date'.
    lines = table_lines(text)
    rows = [row for row in csv.reader(lines) if any(cell.strip() for cell in row)]
    if not rows:
        raise ValueError("the CSV has no rows")
    start = _table_start(rows)
    columns = [cell.strip() for cell in rows[start]]
    if not any(columns):
        raise ValueError("the CSV header row is empty")
    data = rows[start + 1:]
    return columns, len(data), [list(row) for row in data[:PREVIEW_ROWS]]


def split_csv(text: str) -> tuple[str, str]:
    """Split a CSV text into (preamble, table). A title of single-cell rows, and an Open-Meteo 'latitude' block, stay in
    the preamble. The table starts at the header row. A text that already starts at the header is returned whole."""
    lines = text.splitlines(keepends=True)
    flags = comment_flags(text.splitlines())
    kept = []
    for index, line in enumerate(lines):
        if flags[index] or not line.strip():
            continue
        try:
            row = next(csv.reader([line]))
        except csv.Error:
            row = [line]
        kept.append((index, row))
    if not kept:
        return "", text
    start = _table_start([row for _, row in kept])
    cut = kept[start][0]
    if cut == 0:
        return "", text
    return "".join(lines[:cut]), "".join(lines[cut:])


def _numeric_cell(cell: str) -> bool:
    try:
        float(cell)
    except ValueError:
        return False
    return True


def _mostly_numeric(rows: list) -> bool:
    cells = [cell for row in rows for cell in row]
    if not cells:
        return False
    return sum(_numeric_cell(cell) for cell in cells) / len(cells) >= 0.7


def _whitespace_table(text: str) -> tuple[list, int, list, list]:
    """A plain-text table: '#' lines are comments, columns are split on whitespace.

    The NOAA daily CO2 file is this shape (text/plain, about 580,000 bytes). A comma reader sees each row as one
    field, so this parser is the one that counts its rows. A row whose width is not the common width is left out.
    Raises ValueError when the rows are not a numeric table."""
    candidates = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        cells = stripped.split()
        if len(cells) >= 2:
            candidates.append(cells)
    if not candidates:
        raise ValueError("the text has no data rows")
    width = max(set(len(row) for row in candidates), key=lambda n: sum(len(row) == n for row in candidates))
    if width < 2:
        raise ValueError("a table needs at least two columns")
    rows = [row for row in candidates if len(row) == width]
    if _mostly_numeric(rows[0]):
        columns = [f"column_{i + 1}" for i in range(width)]
        data = rows
    else:
        columns = [cell.strip() for cell in rows[0]]
        data = rows[1:]
    if not data or not _mostly_numeric(data):
        raise ValueError("the text is not a numeric table")
    return columns, len(data), [list(row) for row in data[:PREVIEW_ROWS]], list(data[-1])


def preview(body: bytes, fmt: str) -> dict:
    """columns, the first five rows and the total row count of a CSV, JSON or plain-text body.

    ``fmt`` is ``json``, ``csv`` or ``text``. Raises ValueError if it is not a table of that kind."""
    try:
        text = body.decode("utf-8-sig")
    except UnicodeDecodeError:
        raise ValueError("the body is not UTF-8 text") from None
    label = {"json": "JSON", "text": "plain-text"}.get(fmt, "CSV")
    try:
        last = None
        if fmt == "json":
            columns, count, rows = _json_table(text)
        elif fmt == "text":
            columns, count, rows, last = _whitespace_table(text)
        else:
            columns, count, rows = _csv_table(text)
    except (json.JSONDecodeError, csv.Error, RecursionError) as exc:
        raise ValueError(f"the body does not parse as a {label} table ({type(exc).__name__})") from None
    out = {"columns": columns, "rows": rows, "row_count": count}
    if last:
        out["last_row"] = last
    return out


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


RBA_META = ("Title", "Description", "Frequency", "Type", "Units", "Source", "Publication date")
RBA_DATE_FORMATS = ("%d/%m/%Y", "%d-%b-%Y")


def _rba_day(cell: str) -> str | None:
    """An RBA date cell as ISO: DD/MM/YYYY and DD-Mon-YYYY as given, Mon-YYYY (monthly tables such as F13) as the month's
    last day, like the other monthly tables."""
    import calendar
    from datetime import date, datetime

    for fmt in RBA_DATE_FORMATS:
        try:
            return datetime.strptime(cell, fmt).date().isoformat()
        except ValueError:
            continue
    try:
        month = datetime.strptime(cell, "%b-%Y")
    except ValueError:
        return None
    return date(month.year, month.month, calendar.monthrange(month.year, month.month)[1]).isoformat()


def rba_csv(raw: bytes, series: str | None = None) -> tuple[bytes, dict]:
    """A Reserve Bank of Australia statistics table as a CSV of date (ISO) and one column per series ID, with each series'
    title, description, frequency, units, source and publication date kept in provenance.

    Motivation: round 21 (AURATE, 2026-11-27) forecast the cash rate from a DBnomics copy that ended in January 2024 while
    the RBA's own table F1.1 ran to October 2026. That table was listed as a source but never downloaded: its first line is a
    one-cell title and the series codes sit nine lines down, under rows of metadata.
    Logic: the rows before "Series ID" are metadata; the rows after it whose first cell is a date are observations (dates
    are DD/MM/YYYY, DD-Mon-YYYY or Mon-YYYY). Rows with no value at all are dropped. ``series`` keeps one series. Raises ValueError
    when the table has no Series ID row, no dated rows, or not the asked series."""
    import io

    text = raw.decode("utf-8-sig").replace("\r\n", "\n").replace("\r", "\n")
    rows = list(csv.reader(io.StringIO(text)))
    head = next((k for k, r in enumerate(rows) if r and r[0].strip() == "Series ID"), None)
    if head is None:
        raise ValueError("this RBA table has no Series ID row; check the table code on rba.gov.au/statistics/tables")
    ids = [c.strip() for c in rows[head][1:]]
    meta = {r[0].strip(): r[1:] for r in rows[:head] if r and r[0].strip() in RBA_META}
    keep = [k for k, i in enumerate(ids) if i]
    if series:
        if series not in ids:
            raise ValueError(f"this RBA table has no series {series!r}; its series are: {', '.join(i for i in ids if i)}")
        keep = [ids.index(series)]
    out = []
    last: dict[str, str] = {}
    for r in rows[head + 1:]:
        if not r or not r[0].strip():
            continue
        day = _rba_day(r[0].strip())
        if day is None:
            continue
        cells = [(r[k + 1].strip() if k + 1 < len(r) else "") for k in keep]
        if not any(cells):
            continue
        for k, cell in zip(keep, cells):
            if cell:
                last[ids[k]] = day
        out.append([day, *cells])
    if not out:
        raise ValueError("this RBA table has no dated observations")
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(["date", *[ids[k] for k in keep]])
    w.writerows(out)

    def field(name: str, k: int) -> str:
        values = meta.get(name) or []
        return values[k].strip()[:300] if k < len(values) else ""

    info = {ids[k]: {"title": field("Title", k), "description": field("Description", k), "frequency": field("Frequency", k),
                     "units": field("Units", k), "source": field("Source", k), "published": field("Publication date", k),
                     "last_date": last.get(ids[k], "")} for k in keep}
    return buf.getvalue().encode("utf-8"), {"rba": {"series": info}}


def dbnomics_csv(raw: bytes) -> tuple[bytes, dict]:
    """A DBnomics series document as a CSV of period, period_start_day and value, with what the mirror says about itself
    (series name, frequency, when it indexed the provider). Raises ValueError when the document holds no series."""
    data = json.loads(raw.decode("utf-8"))
    docs = ((data.get("series") or {}).get("docs") or []) if isinstance(data, dict) else []
    if not docs or not isinstance(docs[0], dict):
        message = data.get("message") if isinstance(data, dict) else None
        raise ValueError(f"DBnomics returned no series{': ' + str(message)[:200] if message else ''}; find the exact "
                         "provider, dataset and series codes with search_open_data")
    doc = docs[0]
    periods, values = doc.get("period") or [], doc.get("value") or []
    starts = doc.get("period_start_day") or [""] * len(periods)
    if not periods or len(periods) != len(values):
        raise ValueError("DBnomics returned a series without matching periods and values")
    lines = ["period,period_start_day,value"]
    for period, start, value in zip(periods, starts, values):
        cell = "" if value in (None, "NA") else str(value)
        lines.append(f"{period},{start},{cell}")
    info = {"series_name": str(doc.get("series_name", ""))[:300], "frequency": doc.get("@frequency"),
            "mirror_indexed_at": doc.get("indexed_at"), "last_period": periods[-1]}
    return ("\n".join(lines) + "\n").encode("utf-8"), {"dbnomics": info}


def download_to_dataset(source_key: str, params: dict, datasets: DatasetStore, *, name: str | None = None) -> dict:
    """Download one source and store it with its provenance. Returns the stored meta plus a 'preview' of columns, the
    first five rows and row_count. If the same source and parameters are already stored, returns that dataset without
    downloading again."""
    url = sources.build_url(source_key, params)          # validates every parameter first
    canonical = sources.normalise(source_key, params)
    digest = params_digest(canonical)
    source = sources.get(source_key)
    fmt = _format_of(source_key, canonical)
    existing = _find_existing(datasets, source_key, digest)
    if existing is not None:
        body = datasets.path(existing["id"]).read_bytes()
        return dict(existing, preview=preview(body, fmt))
    fetched = fetch.download(url)
    downloaded = fetched["bytes"]
    table_bytes = downloaded
    extra_provenance: dict = {}
    if source.body_kind in ("zip", "xlsx"):
        table_bytes = containers.to_csv(downloaded, source.body_kind, canonical.get("series_id"))
        fmt = "csv"
    elif source.body_kind == "dbnomics":
        table_bytes, extra_provenance = dbnomics_csv(downloaded)
        fmt = "csv"
    elif source.body_kind == "rba":
        table_bytes, extra_provenance = rba_csv(downloaded, canonical.get("series"))
        fmt = "csv"
    table = preview(table_bytes, fmt)     # nothing is stored unless the body parses as the table it claims to be
    # The stored file is the table alone, so a reader of the file sees the header first. The location block that came
    # before it is kept as text in provenance, and the hash of the raw download is kept too.
    preamble, body = "", table_bytes
    if fmt == "csv":
        preamble, table_text = split_csv(table_bytes.decode("utf-8-sig"))
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
        "raw_sha256": hashlib.sha256(downloaded).hexdigest(),
        "raw_bytes": len(downloaded),
        "preamble": preamble,
        "licence_key": sources.get(source_key).licence_key,
        **extra_provenance,
    }
    suffix = {"json": "json", "csv": "csv", "text": "txt"}[fmt]
    meta = datasets.add(name or f"{source_key}.{suffix}", body, provenance=provenance)
    return dict(meta, preview=table)
