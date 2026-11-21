"""Helpers the agent's analytics code can import inside the sandbox: ``import prophet_lib as pl``.

Motivation vs Logic
Motivation: real files are messy (odd encodings, preambles, semicolons and decimal commas, day-first dates, sentinel
values, wide layouts, bad rows). Every mission would otherwise re-write the same fragile loaders, and a silent
mis-parse produces a confident forecast of the wrong thing.
Logic: loaders never guess silently. They return what they did (``notes``) and what they could not read (``problems``),
and they leave interpretation (which column is the target, whether -999 means missing) to the caller. Nothing here
encodes a scenario: it reads, measures and reports.

Runs inside the sandbox only (pandas and numpy are available there). Inputs are in ``in/``; outputs go to ``out/``.
"""
from __future__ import annotations

import csv
import io
import json
import re
from pathlib import Path
from typing import Any

import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", message="Could not infer format")
warnings.filterwarnings("ignore", message="Parsing dates in")
warnings.filterwarnings("ignore", message="Could not infer format")

MAX_PROFILE_COLUMNS = 40
_NA_TOKENS = {"", "na", "n/a", "nan", "null", "none", "nil", "-", "--", "?", "missing", "#n/a", "#value!"}
_ENCODINGS = ("utf-8-sig", "cp1252", "latin-1")
_UTF16_BOMS = (b"\xff\xfe", b"\xfe\xff")


# --------------------------------------------------------------------------------------------------- reading --

def read_text(path: str | Path) -> tuple[str, str]:
    """Decode a file, trying common encodings. Returns (text, encoding used).

    UTF-16 is accepted only with a byte-order mark. Without one, almost any even-length byte string decodes as UTF-16
    (cp1252 text with accents became CJK characters and lost every row), so it must not be tried blindly."""
    raw = Path(path).read_bytes()
    if raw.startswith(_UTF16_BOMS):
        try:
            return raw.decode("utf-16"), "utf-16"
        except UnicodeDecodeError:
            pass
    for enc in _ENCODINGS:
        try:
            return raw.decode(enc), enc
        except (UnicodeDecodeError, UnicodeError):
            continue
    return raw.decode("utf-8", "replace"), "utf-8 (with replacements)"


def _is_binary(path: str | Path) -> bool:
    """True when the start of the file holds control bytes that no text or table file has (an image, a damaged
    workbook, a random file named .xlsx). UTF-16 files with a byte-order mark are text."""
    with Path(path).open("rb") as f:
        head = f.read(8192)
    if head.startswith(_UTF16_BOMS) or not head:
        return False
    controls = sum(1 for b in head if b < 0x20 and b not in (0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x1A))
    return b"\x00" in head or controls > 0.02 * len(head)


def _sniff_delimiter(lines: list[str]) -> str:
    sample = "\n".join(lines[:40])
    best, best_score = ",", -1.0
    for d in (",", ";", "\t", "|"):
        counts = [len(next(csv.reader([ln], delimiter=d))) for ln in lines[:40] if ln.strip()]
        if not counts:
            continue
        modal = max(set(counts), key=counts.count)
        score = (modal > 1) * (counts.count(modal) / len(counts)) * (1 + min(modal, 20) / 20)
        if score > best_score:
            best, best_score = d, score
    if best_score <= 0 and re.search(r"\s{2,}", sample):
        return r"\s+"
    return best


class TableReadError(Exception):
    """The file cannot be read as a table. The message is written for the person who uploaded it."""


MAX_ROWS = 500_000
MAX_DECOMPRESSED = 200 * 1024 * 1024


def sniff_kind(path: str | Path) -> str:
    """What the file really is, from its first bytes, not its extension."""
    p = Path(path)
    with p.open("rb") as f:
        head = f.read(8)
    if head[:2] == b"\x1f\x8b":
        return "gzip"
    if head[:3] == b"BZh":
        return "bzip2"
    if head[:4] == b"PAR1":
        return "parquet"
    if head[:6] == b"ARROW1":
        return "feather"
    if head[:8] == b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1":
        return "ole"                      # legacy .xls, or an encrypted workbook
    if head[:4] == b"PK\x03\x04":
        import zipfile
        try:
            names = zipfile.ZipFile(p).namelist()
        except zipfile.BadZipFile:
            return "corrupt zip"
        if "xl/workbook.xml" in names:
            return "xlsx"
        if "xl/workbook.bin" in names:
            return "xlsb"
        if "mimetype" in names and "content.xml" in names:
            return "ods"
        if any(n.startswith("Index/") for n in names):
            return "numbers"
        return "zip"
    return "text"


def _cell(v: Any) -> str:
    if v is None or (isinstance(v, float) and v != v):
        return ""
    if isinstance(v, float) and v.is_integer() and abs(v) < 1e15:
        return str(int(v))
    if isinstance(v, (pd.Timestamp,)) or hasattr(v, "isoformat"):
        return v.isoformat(sep=" ") if hasattr(v, "hour") else v.isoformat()
    return str(v).strip()


def list_sheets(path: str | Path) -> list[dict]:
    """Sheets of a workbook with their visibility, or [] for a file that is not a workbook."""
    kind = sniff_kind(path)
    if kind == "xlsx":
        import openpyxl
        wb = openpyxl.load_workbook(path, read_only=True, data_only=True, keep_links=False)
        try:
            return [{"name": ws.title, "state": ws.sheet_state} for ws in wb.worksheets]
        finally:
            wb.close()
    if kind in ("xlsb", "ods", "ole"):
        from python_calamine import CalamineWorkbook
        wb = CalamineWorkbook.from_path(str(path))
        return [{"name": n, "state": "visible"} for n in wb.sheet_names]
    return []


def _excel_rows(path: Path, kind: str, sheet: str | int | None, report: dict) -> list[list[str]]:
    sheets = list_sheets(path)
    visible = [s for s in sheets if s["state"] == "visible"] or sheets
    hidden = [s["name"] for s in sheets if s["state"] != "visible"]
    if hidden:
        report["hidden_sheets_not_read"] = hidden
    report["sheets"] = [s["name"] for s in visible]
    if sheet is None or sheet == 0:
        pick = visible[0]["name"]
    elif isinstance(sheet, int):
        pick = visible[min(sheet, len(visible) - 1)]["name"]
    else:
        names = [s["name"] for s in sheets]
        if sheet not in names:
            raise TableReadError(f"there is no sheet named {sheet!r}; the sheets are {names}")
        pick = sheet
    report["sheet"] = pick
    rows: list[list[str]] = []
    if kind == "xlsx":
        import openpyxl
        wb = openpyxl.load_workbook(path, read_only=True, data_only=True, keep_links=False)
        try:
            ws = wb[pick]
            ws.reset_dimensions()
            for i, r in enumerate(ws.iter_rows(values_only=True)):
                if i >= MAX_ROWS:
                    report["problems"].append(f"stopped after {MAX_ROWS:,} rows; the sheet is longer")
                    break
                rows.append([_cell(c) for c in r])
        finally:
            wb.close()
    else:
        from python_calamine import CalamineWorkbook
        wb = CalamineWorkbook.from_path(str(path))
        data = wb.get_sheet_by_name(pick).to_python()
        if len(data) > MAX_ROWS:
            report["problems"].append(f"stopped after {MAX_ROWS:,} rows; the sheet is longer")
            data = data[:MAX_ROWS]
        rows = [[_cell(c) for c in r] for r in data]
    return rows


def _blocks(rows: list[list[str]]) -> list[tuple[int, int]]:
    """Row ranges (start, end, 0-based, inclusive) of tables separated by fully blank rows. Only blocks of 3+ rows count."""
    out, start = [], None
    for i, r in enumerate(rows + [[]]):
        blank = not any(c.strip() for c in r)
        if not blank and start is None:
            start = i
        if blank and start is not None:
            if i - start >= 3:
                out.append((start, i - 1))
            start = None
    return out


_WORKBOOK_EXTENSIONS = {".xlsx": "xlsx", ".xlsm": "xlsx", ".xlsb": "xlsb", ".ods": "ods", ".xls": "ole"}


def read_table(path: str | Path, sheet: str | int | None = None, block: int | None = None) -> tuple[pd.DataFrame, dict]:
    """Read a table of any common shape as text. Returns (frame of strings, report).

    Spreadsheets: ``sheet`` is a name or position (default: the first visible sheet); hidden sheets are listed and not
    read. When a sheet holds several tables separated by blank rows, ``block`` picks one (default the first) and the
    others are listed. Compressed files are opened; archives with several members read the first supported one and list
    the rest. The report lists what was done and what could not be read; nothing is repaired silently.
    """
    p = Path(path)
    report: dict[str, Any] = {"file": p.name, "problems": []}
    kind = sniff_kind(p)
    report["detected_type"] = kind
    if kind == "numbers":
        raise TableReadError("Apple Numbers files cannot be read; export from Numbers to .xlsx or .csv and upload that")
    if kind == "ole" and p.suffix.lower() in (".xlsx", ".xlsm"):
        raise TableReadError("this workbook is password-protected or in an old format; save an unprotected .xlsx copy")
    if kind == "corrupt zip":
        raise TableReadError("the file is damaged (its archive index cannot be read); no data from it was used")
    workbook_kind = _WORKBOOK_EXTENSIONS.get(p.suffix.lower())
    if workbook_kind and kind != workbook_kind:
        # A workbook name on a file that is not a workbook (renamed CSV, HTML export, truncated download). Reading it as
        # text would give a table of noise, so it is refused and the person is told what the file really is.
        raise TableReadError(f"{p.name} has a {p.suffix} name but its content is not an Excel or OpenDocument workbook "
                             f"(detected: {kind}); no data from it was used. Save it again as .xlsx or export .csv")
    if kind in ("gzip", "bzip2"):
        import bz2
        import gzip
        opener = gzip.open if kind == "gzip" else bz2.open
        with opener(p, "rb") as f:
            raw = f.read(MAX_DECOMPRESSED + 1)
        if len(raw) > MAX_DECOMPRESSED:
            raise TableReadError("the compressed file expands to more than 200 MB; split it and upload the parts")
        tmp = Path("home") / f"{p.stem}.unz"
        tmp.parent.mkdir(exist_ok=True)
        tmp.write_bytes(raw)
        df, rep = read_table(tmp, sheet, block)
        rep["file"], rep["decompressed_from"] = p.name, kind
        return df, rep
    if kind == "zip":
        import zipfile
        z = zipfile.ZipFile(p)
        infos = [i for i in z.infolist() if not i.is_dir() and not i.filename.startswith(("__MACOSX", ".")) and ".." not in i.filename]
        if sum(i.file_size for i in infos) > MAX_DECOMPRESSED:
            raise TableReadError("the archive expands to more than 200 MB; upload the files separately")
        usable = [i for i in infos if i.filename.lower().rsplit(".", 1)[-1] in ("csv", "tsv", "txt", "xlsx", "xls", "json", "jsonl", "ods", "xlsb", "parquet")]
        if not usable:
            raise TableReadError("the archive holds no spreadsheet or data file")
        tmp = Path("home") / "zipmember"
        tmp.mkdir(parents=True, exist_ok=True)
        member = tmp / Path(usable[0].filename).name
        member.write_bytes(z.read(usable[0]))
        df, rep = read_table(member, sheet, block)
        rep["file"], rep["archive_member"] = p.name, usable[0].filename
        if len(usable) > 1:
            rep["other_members_not_read"] = [i.filename for i in usable[1:20]]
        return df, rep
    if kind == "feather":
        raise TableReadError("Feather files need a reader that is not installed; export to .csv or .parquet")
    if kind == "parquet":
        import duckdb
        df = duckdb.sql(f"SELECT * FROM read_parquet('{str(p).replace(chr(39), '')}') LIMIT {MAX_ROWS}").df()
        report["format"] = "parquet"
        return df.astype(str).replace({"nan": "", "None": "", "NaT": ""}), report
    if kind in ("xlsx", "xlsb", "ods", "ole"):
        try:
            rows = _excel_rows(p, kind, sheet, report)
        except TableReadError:
            raise
        except Exception as exc:
            raise TableReadError(f"the workbook could not be read in full ({type(exc).__name__}); no data from it was used")
        report["format"] = "excel" if kind != "ods" else "opendocument"
        return _rows_to_frame(rows, report, block), report
    if _is_binary(p):
        raise TableReadError(f"{p.name} is not a table or text file (it holds binary data, possibly a damaged or wrong-type "
                             "file); no data from it was used")
    text, enc = read_text(p)
    report["encoding"] = enc
    stripped = text.lstrip()
    if p.suffix.lower() in (".json", ".jsonl", ".ndjson") or stripped[:1] in "[{":
        try:
            obj = json.loads(text)
            df = pd.json_normalize(obj if isinstance(obj, list) else (obj.get("data", obj) if isinstance(obj, dict) else obj))
            report.update(format="json")
            return df.astype(str).replace("nan", ""), report
        except (ValueError, TypeError, AttributeError):
            try:
                df = pd.read_json(io.StringIO(text), lines=True)
                report.update(format="json lines")
                return df.astype(str), report
            except ValueError:
                report["problems"].append("looked like JSON but did not parse; read as text instead")
    lines = text.splitlines()
    delim = _sniff_delimiter([ln for ln in lines if ln.strip()])
    report.update(format="delimited text", delimiter=("whitespace" if delim == r"\s+" else repr(delim)))
    if delim == r"\s+":
        rows = [re.split(r"\s+", ln.strip()) for ln in lines if ln.strip()]
    else:
        # Parse the whole text, not the lines: a quoted field may hold a line break, and splitting first merged it.
        rows = [r for r in csv.reader(io.StringIO(text, newline=""), delimiter=delim) if r and any(c.strip() for c in r)]
    return _rows_to_frame(rows, report, None, pad=False), report


def _rows_to_frame(rows: list[list[str]], report: dict, block: int | None, pad: bool = True) -> pd.DataFrame:
    raw = rows
    blocks = _blocks(raw)
    sel = range(len(raw))
    if len(blocks) > 1:
        report["tables_on_sheet"] = [{"rows": f"{a + 1}-{b + 1}", "first_cells": [c for c in raw[a][:4]]} for a, b in blocks]
        # The first table is the default, as documented; the others are listed in the report, never merged.
        pick = block if block is not None else 0
        a, b = blocks[min(pick, len(blocks) - 1)]
        report["table_read"] = min(pick, len(blocks) - 1)
        sel = range(a, b + 1)
    idx = [i for i in sel if any(c.strip() for c in raw[i])]
    rows = [raw[i] for i in idx]
    if not rows:
        return pd.DataFrame()
    if pad:
        # Spreadsheet rows are padded to the sheet width; trim columns that are empty everywhere.
        width = max(len(r) for r in rows)
        rows = [r + [""] * (width - len(r)) for r in rows]
        keep = [j for j in range(width) if any(r[j].strip() for r in rows)]
        if len(keep) < width:
            rows = [[r[j] for j in keep] for r in rows]
        nwidth = len(rows[0])
    else:
        # Delimited text: a row with the wrong number of fields is reported, never padded into shape.
        lens = [len(r) for r in rows]
        nwidth = max(set(lens), key=lens.count)
    counts = [sum(1 for c in r if c.strip()) for r in rows]
    # A table's header row is the first row that has about as many filled cells as the widest rows below it.
    modal = max(set(counts[:200]), key=counts[:200].count)
    start = next((i for i, c in enumerate(counts) if c >= (max(2, -(-modal * 3 // 5)) if modal >= 2 else 1)), 0)
    if idx[start]:
        report["skipped_preamble_lines"] = idx[start]
    first = [c.strip() for c in rows[start]]
    headerless = _looks_headerless(first, rows[start + 1:start + 41], nwidth)
    if headerless:
        header = [f"column_{i + 1}" for i in range(nwidth)]
        report["header"] = "none found; columns are named column_1, column_2, ..."
        data_rows, first_line = rows[start:], start + 1
    else:
        header, data_rows, first_line = first, rows[start + 1:], start + 2
        merged = _two_row_header(first, rows[start + 1:start + 42], nwidth)
        if merged:
            header, data_rows, first_line = merged, rows[start + 2:], start + 3
            report["header_rows"] = 2
            report["header_note"] = "a two-row header (merged group labels over column names) was combined into one name per column"
    body, bad = [], []
    for i, r in enumerate(data_rows, start=first_line):
        (body if len(r) == nwidth else bad).append(r if len(r) == nwidth else (i, len(r)))
    if bad:
        report["problems"].append({"wrong_field_count_rows": len(bad), "first_rows_and_counts": bad[:10], "expected_fields": nwidth})
    df = pd.DataFrame(body, columns=_unique(header))
    report["header_row"] = None if headerless else idx[start] + 1
    totals = [i for i, r in enumerate(body) if r and re.match(r"^\s*(grand\s+|sub[\s-]*)?total\b", r[0] or "", re.I)]
    if totals:
        report["rows_that_look_like_totals"] = len(totals)
    return df


def _two_row_header(top: list[str], below: list[list[str]], width: int) -> list[str] | None:
    """Combined column names when the first row holds group labels (often merged cells) and the next row holds the names.

    Detected when the next row is text above numbers, and the first row has gaps or repeats where the second row has names.
    Returns None when the first row alone is the header."""
    if len(below) < 4 or any(len(r) != width for r in below[:4]):
        return None
    sub = [c.strip() for c in below[0]]
    named = [j for j in range(width) if sub[j]]
    if len(named) < 2 or len(named) < width // 2:
        return None
    body = below[1:]
    numeric = lambda j: pd.to_numeric(pd.Series([r[j] for r in body if len(r) == width]).str.replace(",", "", regex=False)
                                     .str.replace(r"[$€£%\s]", "", regex=True), errors="coerce").notna().mean() >= 0.8
    if not all(numeric(j) and pd.isna(pd.to_numeric(sub[j].replace(",", ""), errors="coerce")) for j in named):
        return None
    labels = [top[j] for j in named if top[j]]
    if not (any(not top[j] for j in named) or len(set(labels)) < len(labels)):
        return None
    out, group = [], ""
    for j in range(width):
        group = top[j] or group              # a merged group label covers the columns to its right until the next label
        out.append(f"{group} {sub[j]}".strip() if sub[j] else top[j])
    return out


def _looks_headerless(first: list[str], below: list[list[str]], width: int) -> bool:
    """True when the first row is data: it parses as numbers or dates in the columns where the rows below do."""
    below = [r for r in below if len(r) == width]
    if len(below) < 3:
        return False
    numeric_below = [c for c in range(width) if pd.to_numeric(pd.Series([r[c] for r in below]).str.replace(",", ".", regex=False), errors="coerce").notna().mean() >= 0.9]
    if not numeric_below:
        return False
    as_number = [pd.notna(pd.to_numeric(first[c].replace(",", "."), errors="coerce")) for c in numeric_below]
    return sum(as_number) >= max(1, len(numeric_below) // 2) and not any(re.search(r"[A-Za-z]{3,}", first[c]) for c in numeric_below)


def _unique(names: list[str]) -> list[str]:
    seen: dict[str, int] = {}
    out = []
    for i, n in enumerate(names):
        n = n or f"column_{i + 1}"
        seen[n] = seen.get(n, 0) + 1
        out.append(n if seen[n] == 1 else f"{n}_{seen[n]}")
    return out


# --------------------------------------------------------------------------------------------------- parsing --

_UNIT_WORDS = {"aud", "usd", "eur", "gbp", "nzd", "cad", "jpy", "chf", "cny", "inr", "kg", "g", "mg", "t", "mt", "lb", "lbs", "oz", "m", "cm", "mm",
               "km", "l", "ml", "ea", "pc", "pcs", "units", "unit", "hrs", "hr", "h", "pts", "x"}


_MAGNITUDE_SINGLE = {"K": 1e3, "k": 1e3, "M": 1e6, "B": 1e9}      # case matters: lower-case "m" is metres
_MAGNITUDE_WORDS = {"thousand": 1e3, "thousands": 1e3, "mn": 1e6, "mio": 1e6, "million": 1e6, "millions": 1e6,
                    "bn": 1e9, "billion": 1e9, "billions": 1e9}


def _magnitude_of(suffix: str) -> float | None:
    """The scale a suffix stands for ("K", "M", "B", "million", ...), or None when it is not a scale."""
    if suffix in _MAGNITUDE_SINGLE:
        return _MAGNITUDE_SINGLE[suffix]
    return _MAGNITUDE_WORDS.get(suffix.lower()) if len(suffix) > 1 else None


def to_number(s: pd.Series, *, na: set[str] | None = None) -> tuple[pd.Series, dict]:
    """Parse text to numbers. Handles decimal commas, thousands separators, currency, percent and (negative).

    Returns (float series with NaN for unparsed or missing, info) where info counts what was converted so a mis-read is
    visible instead of silent.
    """
    na = _NA_TOKENS | (na or set())
    t = s.astype(str).str.strip()
    is_na = t.str.lower().isin(na)
    looks_like_date_or_clock = t.str.contains(r"\d[/:]\d|^\d{4}-\d{1,2}(?:-\d{1,2})?(?:[ T]|$)")
    neg = t.str.match(r"^\(.*\)$")
    t = t.str.replace(r"^\((.*)\)$", r"\1", regex=True).str.replace(r"[\s\u00a0$€£¥₹]", "", regex=True)
    # Letters survive only as a currency code or a unit beside a number ("AUD100", "12kg"). Any other text, such as an
    # id like "INV-1000", is not a number and must not lose its letters and become -1000.
    parts = t.str.extract(r"^([A-Za-z]{0,6})([-+]?[\d.,]+(?:[eE][-+]?\d+)?%?)([A-Za-z]{0,12})$")
    suffix = parts[2].fillna("")
    magnitude = suffix.map(_magnitude_of)
    fine = (parts[1].notna() & parts[0].fillna("").str.lower().isin(_UNIT_WORDS | {""})
            & (suffix.str.lower().isin(_UNIT_WORDS | {""}) | magnitude.notna()))
    t = parts[1].where(fine, "x").fillna("x")
    # "1.2M" is 1.2 million, not 1.2 metres. Only the scale is applied here; the count is reported in the info.
    pct = t.str.endswith("%")
    t = t.str.rstrip("%")
    # Both marks present: the last one is the decimal point ("1.234,56" is European, "1,234.56" is not). A lone comma
    # followed by 1-2 digits at the end is a decimal ("12,5"); otherwise a comma is a thousands mark.
    last_comma, last_dot = t.str.rfind(","), t.str.rfind(".")
    both = (last_comma >= 0) & (last_dot >= 0)
    european = both & (last_comma > last_dot)
    t = t.where(~european, t.str.replace(".", "", regex=False).str.replace(",", ".", regex=False))
    only_comma = t.str.contains(",") & ~t.str.contains(r"\.")
    dec = only_comma & t.str.match(r"^-?\d+,\d{1,2}$")
    t = t.where(~dec, t.str.replace(",", ".", regex=False))
    t = t.str.replace(",", "", regex=False)
    num = pd.to_numeric(t.mask(is_na | looks_like_date_or_clock), errors="coerce")
    num = num.where(~neg, -num.abs())
    num = num.where(~pct, num / 100.0)
    scaled = magnitude.notna() & num.notna()
    num = num * magnitude.where(scaled, 1.0).astype(float)
    info = {"missing_tokens": int(is_na.sum()), "unparsed": int((num.isna() & ~is_na & ~looks_like_date_or_clock).sum()),
            "decimal_comma_values": int(dec.sum() + european.sum()), "percent_values": int(pct.sum()),
            "parenthesised_negatives": int(neg.sum()), "scaled_by_suffix": int(scaled.sum())}
    return num, info


_ISO_START = r"^\d{4}[-/.]\d{1,2}[-/.]\d{1,2}(?:$|[ T])"
_DMY = r"^(\d{1,2})[/.\-](\d{1,2})[/.\-](\d{2}|\d{4})$"


_TIME_OFFSET = r"(?<=\d{2}:\d{2})(:\d{2}(\.\d+)?)?(Z|[+-]\d{2}:?\d{2})$"


def _wall_clock(x: pd.Series) -> pd.Series:
    """The date and time as written. A trailing offset (+11:00, Z) is dropped rather than converted, so 2024-07-05T00:00
    +11:00 stays 5 July, and a column mixing offsets still loads (converting to UTC would make it 4 July)."""
    return pd.to_datetime(x.str.replace(_TIME_OFFSET, "", regex=True), errors="coerce", format="mixed")


def _parse_dates(t: pd.Series, dayfirst: bool) -> pd.Series:
    """Each value is parsed on its own, by its shape:
    - year first (2024-01-07): always year-month-day, so it is never read day-first (7 January, not 1 July);
    - numeric day-month-year (05/01/2024): read in the order asked, and a value that is not valid in that order is NaT,
      so the two readings can be compared (pandas' lenient parser would quietly swap them);
    - anything else (5 Jan 2024): pandas' mixed-format parser, which keeps every format that fits."""
    iso = t.str.match(_ISO_START).fillna(False).astype(bool)
    dmy = t.str.match(_DMY).fillna(False).astype(bool) & ~iso
    other_mask = ~(iso | dmy)
    year_first = _wall_clock(t.where(iso))
    parts = t.where(dmy).str.extract(_DMY)
    a, b = pd.to_numeric(parts[0]), pd.to_numeric(parts[1])
    y = pd.to_numeric(parts[2])
    y = y.where(y >= 100, y + 2000)
    day, month = (a, b) if dayfirst else (b, a)
    numeric = pd.to_datetime(pd.DataFrame({"year": y, "month": month, "day": day}), errors="coerce")
    other = pd.to_datetime(t.where(other_mask), errors="coerce", format="mixed", dayfirst=dayfirst)
    return year_first.where(iso, numeric.where(dmy, other))


def to_datetime(s: pd.Series, dayfirst: bool | None = None) -> tuple[pd.Series, dict]:
    """Parse text to timestamps and say how. Reports when day-first and month-first both fit (ambiguous), so the
    caller can ask rather than guess. ``dayfirst=True`` or ``False`` states the order the person gave; it settles an
    ambiguous column and is recorded in the info. Returns (timestamps, info): unpack both, or use ``as_dates``."""
    t = s.astype(str).str.strip()
    t = t.mask(t.str.lower().isin(_NA_TOKENS))
    info: dict[str, Any] = {"ambiguous_day_month": False}
    head = t.dropna().head(60)
    if len(head) and max(_parse_dates(head, df).notna().mean() for df in (False, True)) < 0.8 \
            and pd.to_numeric(head, errors="coerce").notna().mean() < 0.8:
        info["format"] = "not a date"
        return pd.Series(pd.NaT, index=s.index), info
    num = pd.to_numeric(t, errors="coerce")
    if num.notna().mean() > 0.95:
        v = num.dropna()
        if v.between(1e9, 4e9).all():
            info["format"] = "epoch seconds"
            return pd.to_datetime(num, unit="s", errors="coerce"), info
        if v.between(1e12, 4e12).all():
            info["format"] = "epoch milliseconds"
            return pd.to_datetime(num, unit="ms", errors="coerce"), info
        if v.between(20000, 60000).all():
            info["format"] = "excel serial day"
            return pd.to_datetime("1899-12-30") + pd.to_timedelta(num, unit="D"), info
        if v.between(1800, 2200).all() and (v % 1 == 0).all():
            info["format"] = "year"
            return pd.to_datetime(num.astype("Int64").astype(str), format="%Y", errors="coerce"), info
    filled = t.dropna()
    if len(filled) and filled.str.match(r"^\d{4}[-/.]\d{1,2}[-/.]\d{1,2}(?:$|[ T])").mean() >= 0.95:
        info["format"] = "year first (unambiguous)"      # 2023-03-01 is always year-month-day, whatever order a person writes
        return _parse_dates(t, False), info
    if dayfirst is not None:
        info["format"] = "day first (stated)" if dayfirst else "month first (stated)"
        return _parse_dates(t, dayfirst), info
    day = _parse_dates(t, True)
    mon = _parse_dates(t, False)
    both_ok = day.notna() & mon.notna()
    if both_ok.any() and (day[both_ok] != mon[both_ok]).any():
        n_diff = int((day[both_ok] != mon[both_ok]).sum())
        info["ambiguous_day_month"] = day.notna().sum() == mon.notna().sum() and n_diff > 0
        info["rows_where_it_matters"] = n_diff
        if day.notna().sum() > mon.notna().sum():
            info["format"] = "day first"
            return day, info
        if mon.notna().sum() > day.notna().sum():
            info["format"] = "month first"
            return mon, info
        info["format"] = "month first (assumed; ambiguous)"
        return mon, info
    info["format"] = "unambiguous"
    return (mon if mon.notna().sum() >= day.notna().sum() else day), info


def tidy_text(s: pd.Series) -> pd.Series:
    """Names and codes that differ only in case, spacing or stray punctuation become one spelling (the most common one).

    "Sable Logistics", "sable logistics " and "SABLE  LOGISTICS" are one customer. Blanks stay missing. The series carries
    ``s.attrs["report"]``: unique values before and after, and examples of what was merged, so the change is reported."""
    raw = s.astype("string")
    key = raw.str.replace(r"[\s\u00a0]+", " ", regex=True).str.strip().str.casefold().str.replace(r"[.,;]+$", "", regex=True)
    key = key.mask(key == "")
    spelling = raw.str.replace(r"[\s\u00a0]+", " ", regex=True).str.strip().groupby(key).agg(lambda v: v.value_counts().index[0])
    out = key.map(spelling)
    merged = [[x for x in raw[key == k].dropna().unique()][:4] for k in key.dropna().unique() if raw[key == k].nunique() > 1]
    out = out.astype(object).where(out.notna(), None)
    res = pd.Series(out, index=s.index, name=s.name)
    res.attrs["report"] = {"unique_before": int(raw.nunique()), "unique_after": int(key.nunique()), "examples_merged": merged[:5]}
    return res


def as_number(s: pd.Series) -> pd.Series:
    """Just the numbers: ``to_number`` without the info. NaN where a cell was blank or not a number."""
    return to_number(s)[0]


def as_dates(s: pd.Series, dayfirst: bool | None = None) -> pd.Series:
    """Just the timestamps: ``to_datetime`` without the info. Pass ``dayfirst`` when the person stated the order."""
    return to_datetime(s, dayfirst)[0]


CONVERT_SHARE = 0.9          # a column becomes numbers or dates only when this share of its filled cells parses


def load(path: str | Path, sheet: str | int | None = None, block: int | None = None,
         dayfirst: bool | None = None) -> pd.DataFrame:
    """Read a table and return a typed DataFrame: numeric columns are numbers, date columns are timestamps, the rest stay
    text. The recommended way to open a file. What was done is in ``df.attrs["report"]`` (reading notes, which columns
    were converted and how many cells failed, ambiguous dates, columns kept as text). Blank cells are NaN, never zero.
    Identifier-like columns (leading zeros such as 00123) are kept as text."""
    df, rep = read_table(path, sheet, block)
    converted: dict[str, dict] = {}
    as_text: list[str] = []
    for col in df.columns:
        s = df[col]
        filled = s.astype(str).str.strip()
        filled = filled[~filled.str.lower().isin(_NA_TOKENS) & (filled != "")]
        if filled.empty:
            as_text.append(str(col))
            continue
        if filled.str.match(r"^[+-]?0\d+$").any():
            as_text.append(str(col))
            continue
        num, ninfo = to_number(s)
        if num.notna().sum() / len(filled) >= CONVERT_SHARE:
            df[col] = num
            converted[str(col)] = {"to": "number", **{k: v for k, v in ninfo.items() if v}}
            continue
        ts, dinfo = to_datetime(s, dayfirst)
        if dinfo.get("format") != "not a date" and ts.notna().sum() / len(filled) >= CONVERT_SHARE:
            df[col] = ts
            converted[str(col)] = {"to": "date", "format": dinfo.get("format"),
                                   "ambiguous_day_month": bool(dinfo.get("ambiguous_day_month")),
                                   "failed": int(len(filled) - ts.notna().sum())}
            continue
        as_text.append(str(col))
    df.attrs["report"] = {"read": rep, "converted": converted, "kept_as_text": as_text,
                          "ambiguous_dates": [c for c, v in converted.items() if v.get("ambiguous_day_month")]}
    return df


# ------------------------------------------------------------------------------------------------ profiling --

def _regularity(ts: pd.Series) -> dict:
    ts = ts.dropna().sort_values()
    if len(ts) < 3:
        return {"points": int(len(ts))}
    d = ts.diff().dropna()
    n_dup = int((d == pd.Timedelta(0)).sum())
    days = d.dt.total_seconds().median() / 86400
    cal = None
    if 27 <= days <= 32:
        cal, expected = "monthly", (ts.iloc[-1].year - ts.iloc[0].year) * 12 + ts.iloc[-1].month - ts.iloc[0].month + 1
    elif 88 <= days <= 93:
        cal, expected = "quarterly", ((ts.iloc[-1].year - ts.iloc[0].year) * 12 + ts.iloc[-1].month - ts.iloc[0].month) // 3 + 1
    elif 360 <= days <= 370:
        cal, expected = "yearly", ts.iloc[-1].year - ts.iloc[0].year + 1
    if cal:
        step_txt, off = cal, 0.0
    else:
        # The typical step ignores same-timestamp rows (zero gaps); a file with many rows per day otherwise reads as 0.
        nonzero = d[d > pd.Timedelta(0)]
        step = nonzero.mode().iloc[0] if len(nonzero) else pd.Timedelta(0)
        step_txt, off = str(step), float((d != step).mean())
        expected = int((ts.iloc[-1] - ts.iloc[0]) / step) + 1 if step > pd.Timedelta(0) else len(ts)
    return {"first": str(ts.iloc[0]), "last": str(ts.iloc[-1]), "points": int(len(ts)),
            "typical_step": step_txt, "duplicate_timestamps": n_dup,
            "missing_periods": max(0, int(expected) - int(ts.nunique())), "share_off_step": round(off, 3),
            "is_sorted": bool(ts.index.is_monotonic_increasing)}


def profile(path: str | Path, sheet: str | int | None = None, block: int | None = None) -> dict:
    """Describe a data file so a decision can be made about it. Facts only: no repair is applied."""
    try:
        df, rep = read_table(path, sheet, block)
    except TableReadError as exc:
        return {"read": {"file": Path(path).name, "problems": [str(exc)]}, "rows": 0, "columns": [], "issues": [str(exc)],
                "layout_hints": []}
    out: dict[str, Any] = {"read": rep, "rows": int(len(df)), "columns": [], "issues": [], "layout_hints": []}
    if rep.get("hidden_sheets_not_read"):
        out["issues"].append(f"hidden sheets were not read: {rep['hidden_sheets_not_read']}")
    if rep.get("tables_on_sheet"):
        out["issues"].append(f"the sheet holds {len(rep['tables_on_sheet'])} separate tables (rows {[t['rows'] for t in rep['tables_on_sheet']]}); "
                             f"table {rep['table_read']} was read; ask which one is meant if it matters")
    if rep.get("rows_that_look_like_totals"):
        out["issues"].append(f"{rep['rows_that_look_like_totals']} rows look like totals or subtotals; keeping them would double-count")
    if df.empty:
        out["issues"].append("no table could be read from this file")
        return out
    numeric_cols, time_cols, id_cols = [], [], []
    for name in df.columns:
        col = df[name]
        info: dict[str, Any] = {"name": str(name), "unique": int(col.nunique(dropna=True)), "sample": [str(v)[:30] for v in col.head(3)]}
        num, ninfo = to_number(col)
        share = float(num.notna().mean())
        if share >= 0.9:
            v = num.dropna()
            info.update(kind="number", missing=int(num.isna().sum()), min=_r(v.min()), max=_r(v.max()), mean=_r(v.mean()),
                        median=_r(v.median()), zero_share=round(float((v == 0).mean()), 3), negatives=int((v < 0).sum()))
            top = v.round(6).value_counts()
            if len(top) and top.iloc[0] >= max(3, 0.002 * len(v)) and top.index[0] in (-1, -9, -99, -999, -9999, 9999, 99999, 999999, -99999, -999999):
                info["sentinel_candidate"] = {"value": float(top.index[0]), "count": int(top.iloc[0])}
                out["issues"].append(f"column {name!r}: value {top.index[0]:g} repeats {int(top.iloc[0])} times and looks like a missing-data code")
            if ninfo["decimal_comma_values"]:
                info["decimal_comma_values"] = ninfo["decimal_comma_values"]
            if ninfo["unparsed"]:
                info["unparsed_values"] = ninfo["unparsed"]
            if len(v) > 2 and (v % 1 == 0).all() and v.between(1800, 2200).all() and v.is_monotonic_increasing and v.is_unique:
                info["looks_like_year"] = True
                time_cols.append(str(name))
            else:
                numeric_cols.append(str(name))
        else:
            ts, tinfo = to_datetime(col)
            if ts.notna().mean() >= 0.9 and col.nunique() > 1:
                info.update(kind="time", **_regularity(ts), parsed_as=tinfo.get("format"))
                if tinfo.get("ambiguous_day_month"):
                    out["issues"].append(f"column {name!r}: dates such as {col.iloc[0]!r} read differently day-first and month-first "
                                         f"({tinfo.get('rows_where_it_matters')} rows differ); ask which applies")
                    info["ambiguous_day_month"] = True
                time_cols.append(str(name))
            else:
                info.update(kind="text", missing=int((col.astype(str).str.strip().str.lower().isin(_NA_TOKENS)).sum()))
                present = col[~col.astype(str).str.strip().str.lower().isin(_NA_TOKENS | {""})].astype(str)
                folded = present.str.replace(r"[\s\u00a0]+", " ", regex=True).str.strip().str.casefold().str.rstrip(".,;")
                if info["unique"] >= 3 and folded.nunique() < present.nunique():
                    info["spelling_variants"] = {"unique": int(present.nunique()), "after_tidy": int(folded.nunique())}
                    out["issues"].append(
                        f"column {name!r}: {present.nunique()} different values but {folded.nunique()} once case and spacing are ignored "
                        "(for example the same name typed two ways); use pl.tidy_text before counting, grouping or joining on it")
                if info["unique"] <= max(50, 0.2 * len(df)) or "spelling_variants" in info:
                    id_cols.append(str(name))
        out["columns"].append(info)
    if len(out["columns"]) > MAX_PROFILE_COLUMNS:
        out["columns_omitted"] = len(out["columns"]) - MAX_PROFILE_COLUMNS
        out["columns"] = out["columns"][:MAX_PROFILE_COLUMNS]
    if rep.get("problems"):
        out["issues"].append("some rows could not be read as part of the table (see read.problems)")
    if time_cols and len(numeric_cols) > 2 and not id_cols:
        out["layout_hints"].append(f"wide: one time column and {len(numeric_cols)} numeric columns; each numeric column may be its own series")
    if time_cols and id_cols and numeric_cols:
        out["layout_hints"].append(f"long: a time column, grouping column(s) {id_cols[:3]} and numeric column(s); each group may be its own series")
    if not time_cols and numeric_cols:
        out["layout_hints"].append("no time column found; rows may already be in time order, or the time information is in the header")
    return out


def profile_workbook(path: str | Path, max_sheets: int = 12) -> dict:
    """Profile every visible sheet of a workbook (or the one table of any other file)."""
    sheets = list_sheets(path) if sniff_kind(path) in ("xlsx", "xlsb", "ods", "ole") else []
    if not sheets:
        return {"sheets": [{"name": None, **profile(path)}]}
    visible = [s["name"] for s in sheets if s["state"] == "visible"] or [s["name"] for s in sheets]
    out = {"sheets": [{"name": n, **profile(path, n)} for n in visible[:max_sheets]]}
    if len(visible) > max_sheets:
        out["sheets_not_profiled"] = visible[max_sheets:]
    hidden = [s["name"] for s in sheets if s["state"] != "visible"]
    if hidden:
        out["hidden_sheets"] = hidden
    return out


def _r(x: Any) -> float:
    return float(np.round(float(x), 6))


# ----------------------------------------------------------------------------------------- building a series --

def build_series(df: pd.DataFrame, value: str, time: str | None = None, *, freq: str | None = None,
                 agg: str = "sum", fill: str = "keep", sentinel: list[float] | None = None) -> pd.Series:
    """One numeric series, oldest first, from a frame of strings.

    ``freq`` is a pandas offset alias (for example "D", "W", "MS"); when given, rows are aggregated with ``agg`` and the
    calendar is made complete. ``fill`` says what to do with the gaps this creates: "keep" (leave NaN), "zero", or
    "interpolate". ``sentinel`` lists codes to treat as missing. Choosing these is the caller's decision, and it should
    be reported in the answer.
    """
    num, _ = to_number(df[value])
    if sentinel:
        num = num.mask(num.isin(sentinel))
    if time is None:
        return num.reset_index(drop=True)
    ts, _ = to_datetime(df[time])
    s = pd.Series(num.values, index=ts).dropna(axis=0, how="all")
    s = s[s.index.notna()].sort_index()
    if freq:
        grouped = s.groupby(pd.Grouper(freq=freq))
        # A period with no recorded value is missing, not zero: sum() of nothing is 0 in pandas unless min_count says so.
        s = grouped.sum(min_count=1) if agg == "sum" else getattr(grouped, agg)()
        if fill == "zero":
            s = s.fillna(0.0)
        elif fill == "interpolate":
            s = s.interpolate(limit_direction="both")
    return s


def save_series(series: pd.Series | list[float] | dict[str, Any], path: str = "out/series.json", *, freq: str | None = None) -> str:
    """Write series for the forecast tools: a list, a Series (its dates are kept for charts) or a dict of named ones."""
    items = series if isinstance(series, dict) else {"series": series}
    out: dict[str, Any] = {"series": {}, "dates": {}}
    if freq:
        out["freq"] = freq
    for name, s in items.items():
        if isinstance(s, pd.Series):
            vals = s.astype(float).tolist()
            if isinstance(s.index, pd.DatetimeIndex):
                out["dates"][name] = [d.strftime("%Y-%m-%d %H:%M") if (d.hour or d.minute) else d.strftime("%Y-%m-%d") for d in s.index]
        else:
            vals = [float(v) for v in s]
        if any(v != v for v in vals):
            raise ValueError(f"series {name!r} still has {sum(v != v for v in vals)} missing values; fill or drop them before saving "
                             "(the choice affects the forecast, so make it explicitly)")
        out["series"][name] = vals
    if len(out["series"]) == 1 and "series" in out["series"]:
        out["series"] = out["series"]["series"]
        out["dates"] = out["dates"].get("series", [])
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(out), encoding="utf-8")
    return path


def find_season(values: list[float] | np.ndarray, candidates: tuple[int, ...] | None = None) -> dict:
    """Autocorrelation at likely seasonal lags, after removing a linear trend. Facts for the caller to weigh."""
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 12:
        return {"note": "too short to test for seasonality", "points": n}
    t = np.arange(n)
    x = x - np.polyval(np.polyfit(t, x, 1), t)
    sd = x.std()
    if sd == 0:
        return {"note": "series is constant after removing trend"}
    x = (x - x.mean()) / sd
    cands = candidates or tuple(l for l in (2, 3, 4, 5, 6, 7, 12, 24, 26, 48, 52, 96, 168, 365) if l < n // 2)
    acf = {int(l): round(float(np.mean(x[l:] * x[:-l])), 3) for l in cands}
    best = max(acf, key=acf.get) if acf else None
    return {"autocorrelation_by_lag": acf, "strongest_lag": best, "strength": acf.get(best) if best else None,
            "reading": "above about 0.3 suggests a repeating pattern at that lag; near 0 suggests none"}


# ------------------------------------------------------------------------------------------------- charts --
# One entry point for every chart, built to the standard a senior analyst holds: the title states the finding, units and
# period sit in the subtitle, bars start at zero, lines are labelled at their end instead of a legend, colours are
# colour-blind safe, numbers carry thousands separators, and a source line names where the data came from. Rules that
# cannot be bent raise ValueError with the fix; softer findings come back in ``problems``.

OKABE_ITO = ["#0072B2", "#D55E00", "#009E73", "#E69F00", "#56B4E9", "#CC79A7"]   # colour-blind safe, in order of use
_INK, _MUTED, _GRID, _CONTEXT = "#2b241c", "#6b6256", "#e5dccb", "#b9ae9a"
MAX_SERIES = 6
MAX_BARS = 15
CHART_KINDS = ("line", "bar", "hist", "scatter", "forecast", "small_multiples")


def _decimals(values: Any) -> int:
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if not len(v) or np.allclose(v, np.round(v)):
        return 0
    return 1 if np.nanmax(np.abs(v)) >= 10 else 2


def _fmt(v: float, unit: str, d: int = 0, compact: bool = False) -> str:
    sign = "-" if v < 0 else ""
    a = abs(v)
    if compact and a >= 1e9:
        body = f"{a / 1e9:,.1f}B"
    elif compact and a >= 1e6:
        body = f"{a / 1e6:,.1f}M"
    else:
        body = f"{a:,.{d}f}"
    if unit in ("$", "A$", "€", "£", "¥"):
        return f"{sign}{unit}{body}"
    if unit == "%":
        return f"{sign}{body}%"
    return f"{sign}{body}"


def _as_series_dict(data: Any) -> dict[str, pd.Series]:
    if isinstance(data, pd.DataFrame):
        return {str(c): data[c].astype(float) for c in data.columns}
    if isinstance(data, pd.Series):
        return {str(data.name or "value"): data.astype(float)}
    if isinstance(data, dict):
        return {str(k): (v if isinstance(v, pd.Series) else pd.Series(list(v))).astype(float) for k, v in data.items()}
    raise ValueError("chart data must be a pandas Series, a DataFrame (one column per series) or a dict of Series")


def _dodge(ys: list[float], gap: float) -> list[float]:
    """Spread label positions so that neighbours are at least ``gap`` apart, keeping their order."""
    order = sorted(range(len(ys)), key=lambda i: ys[i])
    out = list(ys)
    for a, b in zip(order, order[1:]):
        if out[b] - out[a] < gap:
            out[b] = out[a] + gap
    shift = (np.mean([ys[i] for i in order]) - np.mean([out[i] for i in order])) if order else 0
    return [o + shift for o in out]


def _frame(fig: Any, ax: Any, title: str, subtitle: str, source: str) -> None:
    fig.subplots_adjust(top=0.80 if subtitle else 0.86, bottom=0.15, left=0.09, right=0.86)
    fig.text(0.02, 0.965, title, ha="left", va="top", fontsize=15, fontweight="bold", color=_INK, wrap=True)
    if subtitle:
        fig.text(0.02, 0.895, subtitle, ha="left", va="top", fontsize=10.5, color=_MUTED)
    fig.text(0.02, 0.02, source, ha="left", va="bottom", fontsize=8.5, color=_MUTED)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(_CONTEXT)
    ax.tick_params(colors=_MUTED, labelsize=9.5, length=3)
    ax.grid(axis="y", color=_GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _date_axis(ax: Any, index: Any) -> None:
    if isinstance(index, pd.DatetimeIndex):
        import matplotlib.dates as mdates
        loc = mdates.AutoDateLocator(minticks=4, maxticks=8)
        ax.xaxis.set_major_locator(loc)
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(loc))


def _end_labels(ax: Any, items: list[tuple[str, float, str]], x_last: Any, unit: str, d: int, extra: bool = False) -> None:
    """Direct labels at the right end of each line (name and last value), spread so they never overlap."""
    lo, hi = ax.get_ylim()
    lines = max(n.count("\n") + 1 for n, _, _ in items)
    ys = _dodge([y for _, y, _ in items], (hi - lo) * 0.065 * lines)
    for (name, y, colour), ty in zip(items, ys):
        ax.annotate(name, xy=(x_last, ty), xytext=(8, 0), textcoords="offset points", va="center", ha="left", fontsize=9.5,
                    color=colour, fontweight="bold", annotation_clip=False)


def chart(kind: str, data: Any = None, *, title: str, subtitle: str = "", unit: str = "", x_label: str = "", y_label: str = "",
          source: str = "Source: the uploaded data", path: str = "out/chart.png", order: str = "sort",
          highlight: str | None = None, x: str | None = None, y: str | list[str] | None = None, trend: bool = False,
          history: pd.Series | None = None, point: Any = None, bands: list[tuple[float, Any, Any]] | None = None,
          future_index: Any = None, shared_y: bool = True) -> dict:
    """Draw one chart to ``path`` (PNG, with an SVG and the plotted numbers as CSV beside it) and report on it.

    kind: "line" (data: Series, DataFrame or dict of Series, at most 6 series; or a table with x="<column>" and y="<column>"
    or a list of columns), "bar" (a Series of category to value; sorted
    largest first unless order="keep"), "hist" (a sequence of numbers), "scatter" (a DataFrame with x= and y= column names;
    trend=True adds a fitted line), "small_multiples" (a dict or DataFrame, one panel per series, up to 12), and "forecast".
    For "forecast" pass history= (a Series with a date or number index), point= (the forecast values) and bands=[(level,
    lower, upper), ...] with the widest first, for example [(80, q10, q90), (50, q25, q75)]; future_index= is optional.

    title: a sentence that states the finding ("Weekly demand is flat, with a 20% rise expected in March"), not the chart type.
    subtitle: what is measured, the unit and the period. unit: "%", "$", "units", "kg" and so on. source: where the data came from.
    highlight: the name of the one line to emphasise; the others are drawn in grey.
    Returns {"files": [...], "problems": [...], "checks": {...}}. Print the result and fix every problem before you rely on the chart.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if kind not in CHART_KINDS:
        raise ValueError(f"unknown chart kind {kind!r}; use one of {', '.join(CHART_KINDS)}")
    if isinstance(data, pd.DataFrame) and x and kind in ("line", "bar", "small_multiples"):
        # A table with its x column named: that column becomes the axis and y= (one name or a list) the measures.
        if x not in data.columns:
            raise ValueError(f"x={x!r} is not a column; columns are {list(data.columns)}")
        cols = [y] if isinstance(y, str) else (list(y) if y else [c for c in data.columns if c != x and pd.api.types.is_numeric_dtype(data[c])])
        missing = [c for c in cols if c not in data.columns]
        if missing:
            raise ValueError(f"y column(s) {missing} not found; columns are {list(data.columns)}")
        frame = data[[x, *cols]].copy()
        frame[x] = pd.to_datetime(frame[x]) if pd.api.types.is_datetime64_any_dtype(frame[x]) else frame[x]
        data = frame.set_index(x)[cols[0] if kind == "bar" else cols]
    if len(title.split()) < 4:
        raise ValueError("the title must state the finding in a sentence of at least four words, for example "
                         "'Sales fell 12% in the second half'. A title that names the chart type is not allowed.")
    if not subtitle and kind != "small_multiples":
        raise ValueError("give a subtitle with what is measured, its unit and the period, for example 'Units per week, Jan to Jun 2024'")

    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Times", "STIXGeneral", "DejaVu Serif"],
                         "svg.fonttype": "none", "axes.unicode_minus": True})
    problems: list[str] = []
    checks: dict[str, Any] = {"colour_blind_safe_palette": True, "source_line": bool(source)}
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 5), dpi=144)
    table: pd.DataFrame

    if kind == "line":
        series = _as_series_dict(data)
        if len(series) > MAX_SERIES:
            raise ValueError(f"{len(series)} lines are too many to read (limit {MAX_SERIES}). Show the {MAX_SERIES - 1} that matter and "
                             "group the rest as 'Other', or use kind='small_multiples'.")
        if highlight and highlight not in series:
            raise ValueError(f"highlight={highlight!r} is not one of the series: {list(series)}")
        d = _decimals(np.concatenate([s.values for s in series.values()]))
        items = []
        for n, (name, s) in enumerate(series.items()):
            colour = OKABE_ITO[n % len(OKABE_ITO)] if not highlight else (OKABE_ITO[1] if name == highlight else _CONTEXT)
            if s.isna().any():
                problems.append(f"{name}: {int(s.isna().sum())} missing points are drawn as gaps, not as zero")
            ax.plot(s.index, s.values, color=colour, linewidth=2.4 if (not highlight or name == highlight) else 1.4, solid_capstyle="round")
            last = s.dropna()
            if len(last):
                items.append((name, float(last.iloc[-1]), colour))
        _frame(fig, ax, title, subtitle, source)
        _date_axis(ax, next(iter(series.values())).index)
        _end_labels(ax, items, max(s.dropna().index[-1] for s in series.values() if s.notna().any()), unit, d)
        fig.subplots_adjust(right=float(np.clip(0.97 - 0.0105 * max(len(n) for n in series) - 0.03, 0.62, 0.9)))
        table = pd.DataFrame(series)

    elif kind == "bar":
        s = _as_series_dict(data)
        if len(s) != 1:
            raise ValueError("kind='bar' takes one Series of category to value; for several measures draw one chart each")
        s = next(iter(s.values())).dropna()
        if len(s) > MAX_BARS:
            raise ValueError(f"{len(s)} bars are too many (limit {MAX_BARS}). Show the top {MAX_BARS - 3} and group the rest as 'Other'.")
        if order == "sort":
            s = s.sort_values(ascending=False)
        horizontal = len(s) > 5 or max(len(str(i)) for i in s.index) > 10
        labels = [str(i) for i in s.index]
        colours = [OKABE_ITO[0]] * len(s)
        if highlight is not None:
            colours = [OKABE_ITO[1] if str(i) == highlight else _CONTEXT for i in s.index]
        d = _decimals(s.values)
        if horizontal:
            ypos = np.arange(len(s))[::-1]
            ax.barh(ypos, s.values, color=colours, height=0.68)
            ax.set_yticks(ypos, labels)
            ax.set_xlim(min(0, s.min() * 1.15), max(0, s.max() * 1.15) or 1)
            ax.grid(axis="x", color=_GRID, linewidth=0.8)
            ax.grid(axis="y", visible=False)
            ax.axvline(0, color=_INK, linewidth=1)
            for yp, v in zip(ypos, s.values):
                ax.annotate(_fmt(v, unit, d), xy=(v, yp), xytext=(5 if v >= 0 else -5, 0), textcoords="offset points",
                            va="center", ha="left" if v >= 0 else "right", fontsize=9.5, color=_INK)
            checks["zero_baseline"] = ax.get_xlim()[0] <= 0 <= ax.get_xlim()[1]
            ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: _fmt(v, unit, d, compact=True)))
        else:
            xpos = np.arange(len(s))
            ax.bar(xpos, s.values, color=colours, width=0.62)
            ax.set_xticks(xpos, labels)
            ax.set_ylim(min(0, s.min() * 1.15), max(0, s.max() * 1.15) or 1)
            ax.axhline(0, color=_INK, linewidth=1)
            for xp, v in zip(xpos, s.values):
                ax.annotate(_fmt(v, unit, d), xy=(xp, v), xytext=(0, 4 if v >= 0 else -4), textcoords="offset points",
                            ha="center", va="bottom" if v >= 0 else "top", fontsize=9.5, color=_INK)
            checks["zero_baseline"] = ax.get_ylim()[0] <= 0 <= ax.get_ylim()[1]
            ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: _fmt(v, unit, d, compact=True)))
        _frame(fig, ax, title, subtitle, source)
        if horizontal:
            ax.spines["left"].set_visible(False)
            ax.tick_params(axis="y", length=0, labelsize=10, colors=_INK)
        fig.subplots_adjust(right=0.94, left=0.22 if horizontal else 0.09)
        if not checks["zero_baseline"]:
            raise ValueError("a bar chart must start at zero")
        table = s.rename("value").to_frame()

    elif kind == "hist":
        v = pd.Series(np.asarray(data, dtype=float)).dropna()
        if len(v) < 5:
            raise ValueError("a distribution needs at least 5 values")
        iqr = float(v.quantile(0.75) - v.quantile(0.25))
        width = 2 * iqr / (len(v) ** (1 / 3)) if iqr > 0 else 0
        bins = int(np.clip(np.ceil((v.max() - v.min()) / width) if width > 0 else 10, 5, 40))
        counts, edges, _ = ax.hist(v, bins=bins, color=OKABE_ITO[0], edgecolor="white", linewidth=0.8)
        med = float(v.median())
        ax.axvline(med, color=OKABE_ITO[1], linewidth=2)
        d = _decimals(v.values)
        ax.annotate(f"Median {_fmt(med, unit, d)}", xy=(med, ax.get_ylim()[1]), xytext=(6, -4), textcoords="offset points",
                    va="top", ha="left", fontsize=10, color=OKABE_ITO[1], fontweight="bold")
        _frame(fig, ax, title, subtitle, source)
        fig.subplots_adjust(right=0.94)
        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda t, _: _fmt(t, unit, d, compact=True)))
        checks["zero_baseline"] = True
        table = pd.DataFrame({"bin_start": edges[:-1], "bin_end": edges[1:], "count": counts})

    elif kind == "scatter":
        if not isinstance(data, pd.DataFrame) or not x or not y:
            raise ValueError("kind='scatter' needs data=<DataFrame>, x='<column>', y='<column>'")
        df = data[[x, y]].astype(float).dropna()
        ax.scatter(df[x], df[y], s=26, color=OKABE_ITO[0], alpha=0.75, edgecolor="white", linewidth=0.5)
        if trend and len(df) >= 3:
            k, b = np.polyfit(df[x], df[y], 1)
            xs = np.array([df[x].min(), df[x].max()])
            ax.plot(xs, k * xs + b, color=OKABE_ITO[1], linewidth=2)
            r = float(np.corrcoef(df[x], df[y])[0, 1])
            checks["correlation"] = round(r, 3)
            ax.annotate(f"Trend line, correlation {r:.2f}", xy=(xs[1], k * xs[1] + b), xytext=(6, 0), textcoords="offset points",
                        va="center", fontsize=9.5, color=OKABE_ITO[1], fontweight="bold", annotation_clip=False)
        _frame(fig, ax, title, subtitle, source)
        ax.grid(axis="x", color=_GRID, linewidth=0.8)
        ax.set_xlabel(x_label or x, color=_MUTED)
        ax.set_ylabel(y_label or y, color=_MUTED)
        table = df
        if len(df) > 2000:
            problems.append(f"{len(df)} points: consider a sample or a hexbin, the points overlap")

    elif kind == "small_multiples":
        series = _as_series_dict(data)
        if not 2 <= len(series) <= 12:
            raise ValueError("small_multiples draws 2 to 12 panels; group or filter to fewer series")
        plt.close(fig)
        cols = 2 if len(series) <= 4 else 3
        rows = int(np.ceil(len(series) / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(9, 1.0 + 2.1 * rows), dpi=144, sharex=True, sharey=shared_y, squeeze=False)
        for n, (name, s) in enumerate(series.items()):
            a = axes[n // cols][n % cols]
            a.plot(s.index, s.values, color=OKABE_ITO[0], linewidth=1.8)
            a.set_title(name, loc="left", fontsize=10, color=_INK, fontweight="bold")
            for sp in ("top", "right"):
                a.spines[sp].set_visible(False)
            a.grid(axis="y", color=_GRID, linewidth=0.8)
            a.tick_params(colors=_MUTED, labelsize=8.5)
            _date_axis(a, s.index)
        for n in range(len(series), rows * cols):
            axes[n // cols][n % cols].axis("off")
        fig.suptitle(title, x=0.02, ha="left", fontsize=14, fontweight="bold", color=_INK)
        if subtitle:
            fig.text(0.02, 0.905 if rows == 1 else 0.925, subtitle, fontsize=10, color=_MUTED, ha="left")
        fig.text(0.02, 0.01, source, fontsize=8.5, color=_MUTED, ha="left")
        fig.tight_layout(rect=(0, 0.03, 1, 0.90))
        checks["shared_y_axis"] = bool(shared_y)
        table = pd.DataFrame(series)

    else:  # forecast
        if history is None or point is None or not bands:
            raise ValueError("kind='forecast' needs history=<Series>, point=<forecast values> and bands=[(80, lower, upper), ...]")
        h = history.astype(float).dropna()
        pts = np.asarray(point, dtype=float)
        n = len(pts)
        if future_index is None:
            if isinstance(h.index, pd.DatetimeIndex):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    f = pd.infer_freq(h.index[-min(len(h), 40):]) or None
                step = (h.index[-1] - h.index[-2]) if f is None else None
                fi = pd.date_range(h.index[-1], periods=n + 1, freq=f)[1:] if f else pd.DatetimeIndex([h.index[-1] + step * (i + 1) for i in range(n)])
            else:
                fi = pd.RangeIndex(int(h.index[-1]) + 1, int(h.index[-1]) + 1 + n)
        else:
            fi = future_index
        if len(fi) != n:
            raise ValueError(f"future_index has {len(fi)} entries but point has {n}")
        d = _decimals(np.concatenate([h.values, pts]))
        ordered = sorted(bands, key=lambda b: -b[0])
        shades = [0.14, 0.28, 0.40][: len(ordered)] if len(ordered) <= 3 else [0.14] * len(ordered)
        for (level, lo, hi), a in zip(ordered, shades):
            lo, hi = np.asarray(lo, dtype=float), np.asarray(hi, dtype=float)
            if len(lo) != n or len(hi) != n:
                raise ValueError(f"band {level} must have {n} values for lower and upper")
            if np.any(lo > hi):
                raise ValueError(f"band {level} has a lower value above its upper value")
            ax.fill_between(fi, lo, hi, color=OKABE_ITO[1], alpha=a, linewidth=0)
        ax.plot(h.index, h.values, color=_INK, linewidth=2.0)
        bridge_x = [h.index[-1], *fi]
        ax.plot(bridge_x, [h.iloc[-1], *pts], color=OKABE_ITO[1], linewidth=2.4)
        ax.axvline(h.index[-1], color=_MUTED, linestyle=(0, (4, 3)), linewidth=1)
        top = ax.get_ylim()[1]
        ax.annotate("Forecast starts", xy=(h.index[-1], top), xytext=(-6, -4), textcoords="offset points", ha="right", va="top",
                    fontsize=9.5, color=_MUTED)
        _frame(fig, ax, title, subtitle, source)
        _date_axis(ax, h.index)
        d_ = d
        labels = [(f"Forecast  {_fmt(float(pts[-1]), unit, d_, compact=True)}", float(pts[-1]), OKABE_ITO[1])]
        for (level, lo, hi) in ordered:
            lab = f"{level:g}% range\n{_fmt(float(np.asarray(lo)[-1]), unit, d_, compact=True)} to {_fmt(float(np.asarray(hi)[-1]), unit, d_, compact=True)}"
            labels.append((lab, float(np.asarray(hi)[-1]), _MUTED))
        _end_labels(ax, labels, fi[-1], unit, d_)
        fig.subplots_adjust(right=0.76)
        checks["bands"] = [float(b[0]) for b in ordered]
        checks["forecast_origin_marked"] = True
        cols = {"history": h}
        table = pd.concat([pd.DataFrame({"history": h}), pd.DataFrame({"forecast": pts, **{f"lower_{b[0]:g}": np.asarray(b[1]) for b in ordered},
                                                                     **{f"upper_{b[0]:g}": np.asarray(b[2]) for b in ordered}}, index=fi)])

    if kind in ("line", "bar", "hist", "scatter", "forecast"):
        if x_label and kind != "scatter":
            ax.set_xlabel(x_label, color=_MUTED)
        ylab = y_label or ("" if kind == "bar" else unit)
        if ylab and kind not in ("scatter", "bar", "hist"):
            ax.set_ylabel(ylab, color=_MUTED)
        if kind == "hist":
            ax.set_ylabel("Count", color=_MUTED)

    # Programmatic checks on what was drawn: labels fully inside the figure, no text overlapping another.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    ticks = {id(t) for a_ in fig.axes for t in (*a_.get_xticklabels(), *a_.get_yticklabels())}
    texts = [t for t in fig.findobj(matplotlib.text.Text) if t.get_visible() and t.get_text().strip() and id(t) not in ticks]
    boxes = [(t.get_text(), t.get_window_extent(renderer)) for t in texts]
    fw, fh = fig.canvas.get_width_height()
    for txt, b in boxes:
        if b.x1 > fw + 1 or b.x0 < -1 or b.y0 < -1 or b.y1 > fh + 1:
            problems.append(f"text is cut off at the figure edge: {txt[:40]!r}")
    annots = [(t.get_text(), t.get_window_extent(renderer)) for t in texts if isinstance(t, matplotlib.text.Annotation)]
    for i in range(len(annots)):
        for j in range(i + 1, len(annots)):
            if annots[i][1].overlaps(annots[j][1]):
                problems.append(f"labels overlap: {annots[i][0][:30]!r} and {annots[j][0][:30]!r}")
    checks["labels_overlap"] = not any(p.startswith("labels overlap") for p in problems)
    checks["text_inside_figure"] = not any(p.startswith("text is cut off") for p in problems)

    fig.savefig(out, facecolor="white")
    svg = out.with_suffix(".svg")
    fig.savefig(svg, facecolor="white")
    plt.close(fig)
    csv_path = out.with_suffix(".csv")
    table.to_csv(csv_path)
    return {"files": [str(out), str(svg), str(csv_path)], "problems": problems, "checks": checks,
            "note": "the CSV holds exactly the numbers drawn; every figure you quote about the chart must match it"}


# ------------------------------------------------------------------------------------------ large files --
# Files too big to read whole are queried where they are. DuckDB streams CSV and Parquet from disk, spills to a temp
# folder when a step needs more memory than the cap, and returns only the small result the analysis asks for.

BIG_FILE_BYTES = 150 * 1024 * 1024
QUERY_MEMORY = "1GB"


def _scan_expr(path: Path) -> str:
    q = str(path).replace("'", "''")
    kind = sniff_kind(path)
    if kind == "parquet":
        return f"read_parquet('{q}')"
    if kind in ("csv", "text", "tsv"):
        return f"read_csv_auto('{q}', sample_size=200000, ignore_errors=true)"
    raise TableReadError(f"{path.name} is a {kind} file; query reads CSV, TSV and Parquet. Use pl.load for workbooks.")


def _connect(files: dict[str, str | Path]) -> Any:
    import duckdb
    con = duckdb.connect(":memory:")
    con.execute(f"SET memory_limit='{QUERY_MEMORY}'")
    con.execute("SET threads=2")
    tmp = Path(".duck_tmp")
    tmp.mkdir(exist_ok=True)
    con.execute(f"SET temp_directory='{tmp.resolve()}'")
    allowed = []
    for name, f in files.items():
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
            raise ValueError(f"table name {name!r} must be a plain identifier")
        p = Path(f).resolve()
        con.execute(f'CREATE VIEW "{name}" AS SELECT * FROM {_scan_expr(p)}')
        allowed.append(str(p).replace("'", "''"))
    con.execute("SET allowed_paths=[" + ",".join(f"'{a}'" for a in allowed) + "]")
    con.execute("SET enable_external_access=false")        # from here the query can read only the named files
    return con


def size_report(path: str | Path) -> dict:
    """How big a CSV or Parquet file really is and how to work with it: bytes, rows, columns with types, and a strategy."""
    p = Path(path)
    con = _connect({"t": p})
    rows = con.sql("SELECT count(*) FROM t").fetchall()[0][0]
    head = con.sql("SELECT * FROM t LIMIT 0")
    cols = [(n, str(t)) for n, t in zip(head.columns, head.types)]
    size = p.stat().st_size
    if size < BIG_FILE_BYTES and rows <= MAX_ROWS:
        strategy = "small enough to read whole: pl.load(path) and use pandas"
    else:
        strategy = ("too big to read whole: query it in place with pl.query(sql, t=path) so only a grouped or filtered result "
                    "comes back; aggregate to the period or key the question needs before charting or forecasting; never "
                    "print more than 50 rows")
    return {"file": p.name, "bytes": size, "rows": int(rows), "columns": [{"name": n, "type": t} for n, t in cols], "strategy": strategy}


def query(sql: str, max_rows: int = 20_000, **files: str | Path) -> pd.DataFrame:
    """Run one SELECT over files that stay on disk. Name each file as a table: pl.query("SELECT region, sum(units) u FROM t
    GROUP BY 1 ORDER BY u DESC", t="in/sales.csv"). Returns a DataFrame of at most ``max_rows`` rows; df.attrs["truncated"]
    says whether more existed. The query can read only the files you named and cannot write."""
    if not files:
        raise ValueError("name at least one file, for example t='in/sales.csv'")
    stmt = sql.strip().rstrip(";")
    if ";" in stmt or not re.match(r"(?is)^\s*(with|select|from)\b", stmt):
        raise ValueError("pass one SELECT (or WITH ... SELECT) statement")
    con = _connect(files)
    df = con.sql(stmt).limit(max_rows + 1).df()
    df.attrs["truncated"] = len(df) > max_rows
    return df.iloc[:max_rows]
