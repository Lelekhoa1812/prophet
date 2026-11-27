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
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

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


def _resolve_input(path: str | Path) -> Path:
    """The file a path means. Earlier outputs are mounted at in/carried/<file name>, and a model that wrote out/series.json in an
    earlier run often asks for in/carried/out/series.json (28 failed runs in one live mission). Try the path, then the same
    file name in in/carried, in, and in/derived; when none exists, say what is there."""
    p = Path(path)
    if p.exists():
        return p
    for root in ("in/carried", "in", "in/derived"):
        alt = Path(root) / p.name
        if alt.exists():
            return alt
    here = [str(f) for root in ("in", "in/carried", "in/derived") if Path(root).is_dir() for f in sorted(Path(root).iterdir()) if f.is_file()]
    raise FileNotFoundError(f"no file {str(path)!r}. Files you can open: {here or 'none (pass carry=[{run_id, file}] to bring an earlier output)'}. "
                            "Earlier outputs are at in/carried/<file name>, without the out/ folder")


def read_table(path: str | Path, sheet: str | int | None = None, block: int | None = None) -> tuple[pd.DataFrame, dict]:
    """Read a table of any common shape as text. Returns (frame of strings, report).

    Spreadsheets: ``sheet`` is a name or position (default: the first visible sheet); hidden sheets are listed and not
    read. When a sheet holds several tables separated by blank rows, ``block`` picks one (default the first) and the
    others are listed. Compressed files are opened; archives with several members read the first supported one and list
    the rest. The report lists what was done and what could not be read; nothing is repaired silently.
    """
    p = _resolve_input(path)
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


def _utc_if_offsets_differ(t: pd.Series) -> pd.Series | None:
    """Timestamps with differing offsets read as UTC instants, or None when the wall clock is enough.

    Motivation vs Logic: on the Sydney fall-back day, 02:00 +11:00 and 02:00 +10:00 are two different hours. Dropping the
    offsets (_wall_clock) gave both the same label, and build_series then summed them to 20 (repro, measured-repro in
    audit A1-11). Converting to UTC keeps them apart. A column with one offset, or stamps at midnight (dates, not
    instants), keeps the wall clock, because converting a date written with an offset can move it a day.
    """
    has = t.str.contains(r"\d{2}:\d{2}(?::\d{2}(?:\.\d+)?)?(?:Z|[+-]\d{2}:?\d{2})$", regex=True, na=False)
    if not has.any():
        return None
    offs = t[has].str.extract(r"(Z|[+-]\d{2}:?\d{2})$")[0]
    if offs.nunique() < 2:
        return None
    wall = _wall_clock(t[has]).dropna()
    if not ((wall - wall.dt.normalize()) != pd.Timedelta(0)).any():
        return None
    return pd.to_datetime(t, errors="coerce", format="mixed", utc=True).dt.tz_convert(None)


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


def order_evidence(t: pd.Series) -> tuple[int, int]:
    """How many numeric day-month-year values can only be read one way. (day_only, month_only): a first part above 12
    must be the day, a second part above 12 must be the month. A column with both kinds mixes the two orders, and no
    reading of it is safe to take silently; the person has to say which applies."""
    day_only = month_only = 0
    for v in t.dropna().astype(str):
        m = re.match(_DMY, v.strip())
        if not m:
            continue
        a, b = int(m.group(1)), int(m.group(2))
        if a > 12 >= b:
            day_only += 1
        elif b > 12 >= a:
            month_only += 1
    return day_only, month_only


def to_datetime(s: pd.Series, dayfirst: bool | None = None) -> tuple[pd.Series, dict]:
    """Parse text to timestamps and say how. Reports when day-first and month-first both fit (ambiguous), so the
    caller can ask rather than guess. ``dayfirst=True`` or ``False`` states the order the person gave; it settles an
    ambiguous column and is recorded in the info. Returns (timestamps, info): unpack both, or use ``as_dates``."""
    t = s.astype(str).str.strip()
    t = t.mask(t.str.lower().isin(_NA_TOKENS))
    info: dict[str, Any] = {"ambiguous_day_month": False}
    utc = _utc_if_offsets_differ(t)
    if utc is not None:
        info["format"] = "ISO with differing offsets, read as UTC instants"
        info["clock"] = "UTC"
        return utc, info
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
    day_only, month_only = order_evidence(t)
    if day_only and month_only:
        # The column mixes the two orders. Say so, with the counts, whichever reading is used.
        info["order_conflict"] = {"day_first_only_rows": day_only, "month_first_only_rows": month_only}
    if dayfirst is not None:
        info["format"] = "day first (stated)" if dayfirst else "month first (stated)"
        parsed = _parse_dates(t, dayfirst)
        contradicted = month_only if dayfirst else day_only
        if contradicted:
            # Rows that only fit the other order come out as NaT here. They are not guessed, and the count is reported.
            info["rows_contradicting_stated_order"] = contradicted
        return parsed, info
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
         dayfirst: bool | None = None, run_id: str | None = None) -> pd.DataFrame:
    """Read a table and return a typed DataFrame: numeric columns are numbers, date columns are timestamps, the rest stay
    text. The recommended way to open a file. What was done is in ``df.attrs["report"]`` (reading notes, which columns
    were converted and how many cells failed, ambiguous dates, columns kept as text). Blank cells are NaN, never zero.
    Identifier-like columns (leading zeros such as 00123) are kept as text. ``run_id`` is accepted and ignored: an earlier run's
    file is brought in with ``carry`` on the call and opened at in/carried/<file name>."""
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
    nums: dict[str, pd.Series] = {}
    times: dict[str, pd.Series] = {}
    for name in df.columns:
        col = df[name]
        info: dict[str, Any] = {"name": str(name), "unique": int(col.nunique(dropna=True)), "sample": [str(v)[:30] for v in col.head(3)]}
        blank = col.astype(str).str.strip().str.lower().isin(_NA_TOKENS)
        filled = max(int((~blank).sum()), 1)
        if blank.all():
            info.update(kind="empty", missing=int(len(col)))
            out["issues"].append(f"column {name!r} is empty in every row; it carries no information and is ignored")
            out["columns"].append(info)
            continue
        num, ninfo = to_number(col)
        # The share is of the cells that hold something, as in pl.load: a numeric column that is 28% empty is still numeric.
        share = float(num.notna().sum()) / filled
        if share >= CONVERT_SHARE and num.notna().any():
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
                nums[str(name)] = num
        else:
            ts, tinfo = to_datetime(col)
            # A column whose day and month orders are mixed is still a date column: it is read, with the conflict reported,
            # rather than falling back to text and hiding the problem.
            mixed = bool(tinfo.get("order_conflict")) and ts.notna().mean() >= 0.5
            if ((float(ts.notna().sum()) / filled >= CONVERT_SHARE and ts.notna().any()) or mixed) and col.nunique() > 1:
                info.update(kind="time", **_regularity(ts), parsed_as=tinfo.get("format"))
                times[str(name)] = ts
                if mixed:
                    oc = tinfo["order_conflict"]
                    out["issues"].append(
                        f"column {name!r}: {oc['day_first_only_rows']} dates only fit day-first and {oc['month_first_only_rows']} only "
                        f"fit month-first, so the column mixes both orders and the reading cannot be settled from the data. Ask which "
                        f"order applies before forecasting, and say how many rows were not read ({int(ts.isna().sum())} unread here)")
                    info["ambiguous_day_month"] = True
                    info["order_conflict"] = oc
                elif tinfo.get("ambiguous_day_month"):
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
    if times and nums and not id_cols:
        _outlier_facts(out, nums, next(iter(times.values())))
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


def _outlier_facts(out: dict, nums: dict[str, pd.Series], ts: pd.Series) -> None:
    """Values far from their neighbours in a single time-ordered series, reported as a fact. They may be entry errors or
    real events; the caller decides. Skipped when so many points qualify that 'unusual' stops meaning anything."""
    stamps = ts.to_numpy(dtype="datetime64[ns]")
    order = np.argsort(stamps, kind="stable")
    dated = ~np.isnat(stamps[order])           # a row with no readable date (a total, a footer) is not part of the series
    by_name = {c["name"]: c for c in out["columns"]}
    for name, num in list(nums.items())[:5]:
        x = num.to_numpy(dtype=float)[order]
        keep = np.isfinite(x) & dated
        if keep.sum() < 30:
            continue
        rows = np.flatnonzero(keep)
        res = detect_anomalies(x[keep])
        n_found = len(res["indices"])
        if not n_found or n_found > 0.05 * keep.sum():
            continue
        worst = np.argsort(res["scores"])[::-1][:5]
        picks = [{"row": int(order[rows[res["indices"][i]]]) + 1, "date": str(ts.iloc[order[rows[res["indices"][i]]]])[:10],
                  "value": res["values"][i]} for i in worst]
        found = {"count": n_found, "largest": picks}
        if keep.all() and (order == np.arange(len(order))).all():
            found["ignore_ranges_if_confirmed"] = res["ignore_ranges"][:20]      # positions equal the table's row order
        by_name[name]["outlier_candidates"] = found
        out["issues"].append(
            f"column {name!r}: {n_found} value(s) sit far from their neighbours (largest: {picks[0]['value']:g} on {picks[0]['date']}). "
            "They may be entry errors or real events (a promotion, a holiday): compare with known events before ignoring or keeping them")


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

# ------------------------------------------------------------------------------------------ panels of entities --
# A long table often holds several entities (stations, products) that share dates. Adding them up silently gives a total
# that no one asked for. So a table whose dates repeat across a text column that names entities must say which entity
# it means: pass entity= and key=, pass agg= explicitly, or use pl.panel for all of them.

_ENTITY_MAX_LEVELS = 100      # more distinct labels than this is not a short list of entities


_MEASURE_NAME = re.compile(r"qty|quant|unit|sales|amount|price|value|count|cost|revenue|total|weight|volume|demand|rate|temp|"
                           r"score|hours?|minutes?|seconds?", re.IGNORECASE)


def _whole_number_label(s: pd.Series, col: str) -> bool:
    """An integer-valued column that is not named as a measure: a numeric ID such as store_id = 12. Audit item 13: the
    guard skipped every numeric column, so entities labelled 1, 2, 3 summed silently."""
    if pd.api.types.is_bool_dtype(s) or _MEASURE_NAME.search(str(col)):
        return False
    vals = pd.to_numeric(s, errors="coerce").dropna()
    return len(vals) > 0 and bool(np.all(np.mod(vals.to_numpy(dtype=float), 1) == 0))


def _entity_columns(df: pd.DataFrame, time: str, value: str, ts: pd.Series) -> list[str]:
    """Columns that look like entities: 2 to 100 distinct labels, and at least one date that has several of them. Text
    columns qualify; numeric columns qualify when they are whole numbers that are not named as a measure (numeric IDs)."""
    dates = pd.Series(np.asarray(ts), index=df.index, name="_date")
    found: list[str] = []
    for col in df.columns:
        if col in (time, value):
            continue
        s = df[col]
        if pd.api.types.is_datetime64_any_dtype(s):
            continue
        if pd.api.types.is_numeric_dtype(s) and not _whole_number_label(s, col):
            continue
        if pd.api.types.is_bool_dtype(s):
            continue
        if not 2 <= s.nunique(dropna=True) <= _ENTITY_MAX_LEVELS:
            continue
        per_date = pd.DataFrame({"_date": dates, "_label": s}).dropna().groupby("_date")["_label"].nunique()
        if (per_date > 1).any():
            found.append(str(col))
    return found


def _panel_guard(df: pd.DataFrame, time: str | None, value: str, agg: str | None, entity: str | None,
                 key: Any) -> tuple[pd.DataFrame, str]:
    """Pick the rows to use and the aggregation. Returns (frame, agg); agg is "sum" when the caller gave none and no
    entity column was found. Raises ValueError when entities share dates and the caller did not say which one it means."""
    if entity is not None:
        if entity not in df.columns:
            raise ValueError(f"no column {entity!r}; the columns are {list(df.columns)[:30]}")
        if key is None:
            raise ValueError(f"entity={entity!r} needs key= (one value of that column). For every entity at once use "
                             f"pl.panel(df, {entity!r}, time, value)")
        rows = df[df[entity].astype(str).str.strip() == str(key).strip()]
        if rows.empty:
            labels = sorted(df[entity].dropna().astype(str).unique())[:20]
            raise ValueError(f"no rows with {entity} == {key!r}; the values are {labels}")
        return rows, agg or "sum"
    if agg is None and time is not None:
        ts, _ = to_datetime(df[time])
        if pd.Series(np.asarray(ts)).dropna().duplicated().any():
            found = _entity_columns(df, time, value, ts)
            if found:
                col = found[0]
                raise ValueError(f"the column {col!r} names several entities that share dates, and adding them would mix them "
                                 f"together. Pass entity={col!r} and key=<one value>, or pl.panel(df, {col!r}, {time!r}, "
                                 f"{value!r}) for all of them. Pass agg='sum' only if the rows are parts of one total.")
    return df, agg or "sum"


_WEEKDAYS = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")
_CLOSED_SHARE = 0.9  # share of one weekday's periods with no value that makes it look closed, not missing (proposal; A1-03)
_CLOSED_MIN_PERIODS = 4  # periods a weekday needs before its absence is judged (proposal)


# pandas 3 removed the old offset aliases, and its error for "A" suggests "Y", which is removed too (VN Unemployment notebook,
# 2026-11-27: four cells failed on "A", "Y", "A-DEC" and "Y-DEC"). Each old alias maps to the one pandas accepts now.
LEGACY_FREQ = {"A": "YE", "Y": "YE", "AS": "YS", "BA": "BYE", "BY": "BYE", "Q": "QE", "BQ": "BQE", "M": "ME", "BM": "BME",
               "SM": "SME", "H": "h", "BH": "bh", "T": "min", "S": "s", "L": "ms", "U": "us", "N": "ns"}
SMALL_STEPS = ("D", "B", "C", "h", "bh", "min", "s", "ms", "us", "ns")


def freq_alias(freq: Any) -> Any:
    """The offset alias pandas accepts for ``freq``: 'A' and 'Y' become 'YE', 'A-DEC' 'YE-DEC', 'Q' 'QE', 'M' 'ME', 'H' 'h',
    'T' 'min', and so on. Anything else (a current alias, an offset object, None) is returned unchanged."""
    if not isinstance(freq, str):
        return freq
    text = freq.strip()
    m = re.fullmatch(r"(\d*)([A-Za-z]+)(-[A-Za-z]{3})?", text)
    if not m:
        return text
    n, base, anchor = m.group(1), m.group(2), m.group(3) or ""
    return f"{n}{LEGACY_FREQ.get(base, base)}{anchor}"


def _daily_or_finer(freq: Any) -> bool:
    """True when one step is a day or less: only then can a weekday be a closed day."""
    try:
        off = pd.tseries.frequencies.to_offset(freq)
    except (ValueError, TypeError):
        return False
    if off.name in SMALL_STEPS:
        return True
    try:
        return int(off.nanos) <= 86_400 * 10**9
    except ValueError:          # a calendar offset (week anchor, month, quarter, year) has no fixed length
        return False


def _is_business(freq: str) -> bool:
    try:
        off = pd.tseries.frequencies.to_offset(freq)
    except (ValueError, TypeError):
        return False
    return isinstance(off, (pd.offsets.BusinessDay, pd.offsets.CustomBusinessDay))


def _grid_freq(index: pd.DatetimeIndex, freq: str | None) -> str | None:
    """The grid to group on. A business-day grid cannot hold weekend rows: pandas folds them into a neighbouring weekday,
    so the values land on the wrong dates. Refused with the fix named (freq='D' keeps every observed day); a business
    grid is fine when the rows really are weekdays only.

    Motivation vs Logic: live retail rep4 grouped Friday, Saturday and Sunday into one Friday value (repro: 100, 10, 20
    gave 130) and then re-dated 504 values by position, so the forecast ended 2011-11-04 against data to 2011-12-08
    (A1-01, A4-01).
    """
    freq = freq_alias(freq)
    if not freq or not _is_business(freq):
        return freq
    n_sat, n_sun = int((index.day_of_week == 5).sum()), int((index.day_of_week == 6).sum())
    if not (n_sat or n_sun):
        return freq
    parts = [f"{n} on {name}" for name, n in (("Saturday", n_sat), ("Sunday", n_sun)) if n]
    raise ValueError(f"{' and '.join(parts)} rows fall on a weekend. A business-day grid (freq={freq!r}) folds them into a "
                     "weekday's total, so the values sit on the wrong dates. Use freq='D' to keep every observed day: closed "
                     "days stay empty and can be dropped with .dropna(), and the dates are kept.")


def _refuse_repeated_stamps(index: pd.DatetimeIndex, agg_given: bool) -> None:
    """Sub-daily rows that share a timestamp are refused unless the caller names the combination.

    Motivation vs Logic: at a clock change a local hour can occur twice. Summing it doubled the hour silently (Sydney
    fall-back repro gave 20 for one hour, measured-repro, A1-11). Daily duplicates keep the sum default that prepare_series
    documents, so existing daily callers are unchanged.
    """
    if agg_given or not index.has_duplicates:
        return
    rep = index[index.duplicated(keep=False)]
    if not np.any((rep.hour != 0) | (rep.minute != 0)):
        return
    raise ValueError(f"{int(index.duplicated().sum())} sub-daily rows share a timestamp (for example {rep[0]}). At a clock change "
                     "a local hour can occur twice, and summing it would double that hour. Pass agg='sum' only if the rows are "
                     "parts of one total, or agg='mean', 'first' or 'last' to choose. A column whose offsets differ (+11:00 and "
                     "+10:00) is read as UTC instants and needs no agg.")


def _closed_mask(index: pd.DatetimeIndex, closed: Any) -> np.ndarray:
    """Periods on closed days: weekday names ('Saturday') and dates ('2024-12-25', an exchange holiday)."""
    mask = np.zeros(len(index), dtype=bool)
    if not closed:
        return mask
    names: list[str] = []
    dates: list[pd.Timestamp] = []
    for c in closed:
        text = str(c).strip()
        if text.capitalize() in _WEEKDAYS:
            names.append(text.capitalize())
            continue
        try:
            dates.append(pd.Timestamp(text).normalize())
        except (ValueError, TypeError) as err:
            raise ValueError(f"closed= takes weekday names ('Saturday') or dates ('2024-12-25'); {c!r} is neither") from err
    if names:
        mask |= np.asarray(pd.Index(index.day_name()).isin(names))
    if dates:
        mask |= np.asarray(pd.Index(index.normalize()).isin(dates))
    return mask


def _closed_candidates(index: pd.DatetimeIndex, missing: np.ndarray) -> dict[str, float]:
    """Weekdays with no value on most of their periods: a closure pattern, not scattered gaps (A1-03)."""
    days = np.asarray(index.day_name())
    out: dict[str, float] = {}
    for w in _WEEKDAYS:
        on = days == w
        if on.sum() >= _CLOSED_MIN_PERIODS and missing[on].mean() >= _CLOSED_SHARE:
            out[w] = float(missing[on].mean())
    return out


PARTIAL_EDGE = 0.5      # an edge period with fewer rows than this share of the usual count is partial


def _drop_partial_edges(s: pd.Series, counts: pd.Series) -> pd.Series:
    """The aggregated series without a first or last period that holds only part of its rows.

    Motivation: panel round 9 (NEM). The month file of 5-minute demand ended at 00:00 on 27 November, so the last "daily
    mean" was that one reading; the 14-day forecast repeated it and the answer called it 150 days of daily averages.
    Logic: when most periods hold several rows, an edge period with fewer than PARTIAL_EDGE of the median count is dropped,
    and a note is printed so the answer can say so. Interior periods are left alone (a gap is the caller's decision).
    """
    if len(counts) < 3:
        return s
    usual = float(counts[counts > 0].median()) if (counts > 0).any() else 0.0
    if usual < 2:
        return s
    dropped = []
    for at in (counts.index[0], counts.index[-1]):
        if 0 < counts[at] < PARTIAL_EDGE * usual and at in s.index:
            dropped.append(f"{at} ({int(counts[at])} of about {int(usual)} rows)")
            s = s.drop(at)
    if dropped:
        print(f"note: build_series left out partial edge periods: {'; '.join(dropped)}")
    return s


def build_series(df: pd.DataFrame, value: str, time: str | None = None, *, freq: str | None = None,
                 agg: str | None = None, fill: str = "keep", sentinel: list[float] | None = None,
                 entity: str | None = None, key: Any = None) -> pd.Series:
    """One numeric series, oldest first, from a frame of strings.

    ``freq`` is a pandas offset alias (for example "D", "W", "MS"); when given, rows are aggregated with ``agg`` and the
    calendar is made complete. ``fill`` says what to do with the gaps this creates: "keep" (leave NaN), "zero", or
    "interpolate". ``sentinel`` lists codes to treat as missing. ``entity`` and ``key`` select one entity (for example
    entity="station", key="Beijing"). When entities share dates and neither is given, this raises ValueError, because a
    sum would add them together; ``agg`` given explicitly also lifts that check. Choosing these is the caller's decision,
    and it should be reported in the answer.

    Dates are never re-dated by position: each value keeps the date it was recorded on, and the series ends on the last
    observed date. A business-day ``freq`` with weekend rows is refused (see _grid_freq): use ``freq='D'`` to keep every
    observed day. Sub-daily rows that share a timestamp are refused unless ``agg`` is named.
    """
    agg_given = agg is not None
    df, agg = _panel_guard(df, time, value, agg, entity, key)
    num, _ = to_number(df[value])
    if sentinel:
        num = num.mask(num.isin(sentinel))
    if time is None:
        return num.reset_index(drop=True)
    ts, _ = to_datetime(df[time])
    s = pd.Series(num.values, index=ts).dropna(axis=0, how="all")
    s = s[s.index.notna()].sort_index()
    _refuse_repeated_stamps(s.index, agg_given)
    if freq:
        freq = _grid_freq(s.index, freq)
        grouped = s.groupby(pd.Grouper(freq=freq))
        counts = grouped.count()
        # A period with no recorded value is missing, not zero: sum() of nothing is 0 in pandas unless min_count says so.
        s = grouped.sum(min_count=1) if agg == "sum" else getattr(grouped, agg)()
        if agg not in ("last", "first"):       # a month's last close is still that month's close
            s = _drop_partial_edges(s, counts)
        if fill == "zero":
            s = s.fillna(0.0)
        elif fill == "interpolate":
            _refuse_sparse_fill(s, freq)
            s = s.interpolate(limit_direction="both")
    return s


MIN_OBSERVED_SHARE = 0.5   # below this, interpolation would write most of the series rather than fill its gaps


def _refuse_sparse_fill(s: pd.Series, freq: Any) -> None:
    """Interpolation fills gaps in a series; it does not make one. VN Unemployment (2026-11-27): five annual values
    (1997, 1998, 2009, 2019, 2020) became 24 by interpolation and were forecast, so 19 of 24 points were a straight line."""
    total = int(s.size)
    seen = int(s.notna().sum())
    if total and seen / total < MIN_OBSERVED_SHARE:
        raise ValueError(f"only {seen} of {total} periods at freq={freq!r} have a value ({seen / total:.0%}). Interpolating "
                         "would write most of the series, not fill its gaps. Use the grain the source publishes, find a "
                         "source with more observations (search_open_data), or report that the data are too sparse.")


TIME_DST_POLICIES = ("keep_as_is", "sum_repeated_hour", "drop_repeated")


def time_policy(df: pd.DataFrame, time: str, value: str, *, timezone: str | None = None, dst: str = "keep_as_is",
                grain: str | None = None) -> tuple[pd.DataFrame, dict]:
    """Apply a time rule to a loaded frame before the scan: the timezone and what to do with a repeated clock hour.

    Motivation vs Logic (A7-10): the scan reports a daylight-saving repeated hour as a major ``time.dst_shift`` and the
    contract blocks until a decision is recorded, yet the person had no way to state the rule. Logic: an aware column is
    converted to ``timezone`` and its offset dropped, so the stamps are wall-clock in that zone. With a timezone, a
    stamp is in the repeated hour only when its wall time is ambiguous in that zone. ``sum_repeated_hour`` adds the
    values of the repeated stamps into the first, and ``drop_repeated`` keeps the first and removes the rest. Other
    duplicates are left alone. A missing hour (spring forward) is not filled, so the scan still reports it.

    Returns the new frame and a report (counts, the policy used and the ``grain`` to pass to the scan as ``freq``).
    Unknown policies, zones or grains raise ValueError.
    """
    if dst not in TIME_DST_POLICIES:
        raise ValueError(f"dst must be one of {TIME_DST_POLICIES}, not {dst!r}")
    zone = None
    if timezone is not None:
        try:
            zone = ZoneInfo(timezone)
        except (ZoneInfoNotFoundError, ValueError) as exc:
            raise ValueError(f"unknown timezone {timezone!r}") from exc
    if grain is not None:
        try:
            pd.tseries.frequencies.to_offset(grain)
        except (ValueError, TypeError) as exc:
            raise ValueError(f"unknown grain {grain!r}") from exc
    out = df.copy()
    ts = out[time] if pd.api.types.is_datetime64_any_dtype(out[time]) else pd.to_datetime(out[time], errors="coerce")
    if getattr(ts.dt, "tz", None) is not None:
        if zone is not None:
            ts = ts.dt.tz_convert(zone)
        ts = ts.dt.tz_localize(None)
    present = ts.notna()
    if zone is not None:
        # NaT here means ambiguous (the repeated hour); a nonexistent wall time is shifted forward, so it is not counted.
        ambiguous = ts.dt.tz_localize(zone, ambiguous="NaT", nonexistent="shift_forward").isna() & present
    else:
        ambiguous = pd.Series(False, index=ts.index)   # without a zone no stamp can be called a repeated hour
    repeated = present & ts.duplicated(keep=False) & ambiguous
    later = repeated & ts.duplicated(keep="first")
    first = repeated & ~later
    report: dict[str, Any] = {"timezone": timezone, "dst": dst, "grain": grain,
                              "ambiguous_wall_times": int(ts[ambiguous].nunique()) if zone is not None else None,
                              "repeated_stamps": int(ts[repeated].nunique()), "rows_removed": 0,
                              "not_filled": "a missing hour (spring forward) is not filled; the scan still reports it"}
    out[time] = ts
    if dst == "keep_as_is" or not repeated.any():
        return out, report
    if dst == "sum_repeated_hour":
        values = pd.to_numeric(out[value], errors="coerce")
        sums = values[repeated].groupby(ts[repeated].to_numpy()).sum(min_count=1)
        out[value] = values
        out.loc[first, value] = sums.reindex(ts[first]).to_numpy()
    report["rows_removed"] = int(later.sum())
    return out[~later].reset_index(drop=True), report


def save_series(series: pd.Series | list[float] | dict[str, Any], path: str = "out/series.json", *, freq: str | None = None,
                covariates: dict[str, Any] | None = None, name: str | None = None,
                provenance: dict[str, Any] | None = None) -> str:
    """Write series for the forecast tools: a list, a Series (its dates are kept for charts) or a dict of named ones.

    ``name`` labels one series (a list or a Series); load_series returns it with that name. It cannot be combined with a
    dict, whose keys are the names. ``covariates`` maps a name to a list of numbers kept beside the series. A driver known
    only up to now (price paid) must be as long as the history; one known ahead (a planned promotion, holidays) must be as
    long as the history plus the horizon you will forecast. The forecast tools then take them by name:
    future_covariates=['promo'].

    The clock is written with the series: ``step`` (a pandas alias, inferred from the dates when ``freq`` is not given)
    and ``calendar`` ("calendar", "business" or "trading"). The kind is inferred from the step; prepare_series sets
    "trading" in ``s.attrs`` when closed days were given, and that survives dropna(). Files without these keys still load.

    Motivation vs Logic: the forecast record did not carry the step, so the horizon check could not convert "14 days"
    into the series' own unit (A4-13). Recording it here lets later tools and checks read it from the file.
    """
    if name is not None and isinstance(series, dict):
        raise ValueError("name= labels one series; for a dict of series, the dict keys are the names")
    items = series if isinstance(series, dict) else {(str(name) if name is not None else "series"): series}
    out: dict[str, Any] = {"series": {}, "dates": {}}
    if covariates:
        clean: dict[str, list[float]] = {}
        for cname, cv in covariates.items():
            arr = np.asarray(cv, dtype=float).reshape(-1)
            if not np.isfinite(arr).all():
                raise ValueError(f"covariate {cname!r} has missing or non-finite values; fill them before saving")
            clean[str(cname)] = arr.tolist()
        out["covariates"] = clean
    if provenance is not None:
        # origin "retrieved" (numbers from a page web research read, with its source id and URL), "user" or "synthetic". The
        # forecast tools check it when they read the file: a synthetic series cannot name a source.
        out["provenance"] = dict(provenance)
    freq = freq_alias(freq)
    if freq:
        out["freq"] = freq
    step_val = freq
    dated = next((x.index for x in items.values() if isinstance(x, pd.Series) and isinstance(x.index, pd.DatetimeIndex)), None)
    if step_val is None and dated is not None and len(dated) >= 3:
        step_val = _infer_freq(dated.unique().sort_values())
    attr_kind = next((x.attrs.get("calendar") for x in items.values() if isinstance(x, pd.Series) and x.attrs.get("calendar")), None)
    kind = attr_kind or (("business" if _is_business(step_val) else "calendar") if step_val else None)
    if step_val:
        out["step"] = step_val
    if kind:
        out["calendar"] = kind
    for name, s in items.items():
        if isinstance(s, pd.Series):
            vals = s.astype(float).tolist()
            if isinstance(s.index, pd.DatetimeIndex):
                out["dates"][name] = [d.strftime("%Y-%m-%d %H:%M") if (d.hour or d.minute) else d.strftime("%Y-%m-%d") for d in s.index]
        else:
            vals = [float(v) for v in s]
        if not vals:
            # Panel round 14 (AUUNEMP): an empty frame was saved twice as a series of length 0; the backtest then refused it
            # and the planner went looking for the data in the wrong runs.
            raise ValueError(f"series {name!r} is empty: the frame it came from has no rows. Check the filter or the column "
                             "names (print df.shape and df.head()) before saving")
        if any(v != v for v in vals):
            # Panel round 12 (SPX): FRED marks market holidays as empty rows; build_series(freq='D') then added every weekend
            # as well (785 gaps), and the planner retried the same save seven times. A trading series keeps its observed days.
            hint = ""
            if isinstance(s, pd.Series) and isinstance(s.index, pd.DatetimeIndex):
                gaps = s.index[s.isna().to_numpy()]
                weekend = int((gaps.dayofweek >= 5).sum())
                if gaps.size and weekend >= 0.5 * gaps.size:
                    hint = (f" {weekend} of the gaps fall on weekends, so a daily grid added days the market never trades: for "
                            "a trading-day series call pl.build_series(df, value=..., time=...) without freq, which keeps only "
                            "the observed days, or save s.dropna().")
                elif gaps.size:
                    hint = (" If these are days with no trading (holidays), save s.dropna(); if they are real gaps, fill them "
                            "(build_series(..., fill='interpolate')) and say so.")
            raise ValueError(f"series {name!r} still has {sum(v != v for v in vals)} missing values; fill or drop them before saving "
                             "(the choice affects the forecast, so make it explicitly). Closed days from prepare_series(closed=...) "
                             "are dropped with s.dropna(), which keeps every remaining date." + hint)
        out["series"][name] = vals
    only = next(iter(items)) if len(items) == 1 else None
    if only is not None and (only == "series" or name is not None):
        out["series"] = out["series"][only]
        out["dates"] = out["dates"].get(only, [])
        if name is not None:
            out["name"] = str(name)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(out), encoding="utf-8")
    return path


def load_series(path: str | Path = "in/carried/series.json", name: str | None = None, run_id: str | None = None,
                carry: Any = None) -> pd.Series | dict[str, pd.Series]:
    """Read a file written by ``save_series`` back as a pandas Series with its dates as the index (a dict of Series when it holds
    several and no ``name`` is given). Earlier outputs are at in/carried/<file name> (bring them with ``carry``).

    ``run_id`` and ``carry`` are not options of this function. They were accepted and ignored (29 of 33 calls in the six live
    runs passed run_id=), so a call that named a run could read the mounted copy. They now raise and say what to do."""
    if run_id is not None:
        raise TypeError("load_series reads the file this run can see, and run_id= does not pick another run's copy. To use an "
                        "earlier output, pass carry=[{'run_id': <its id>, 'file': 'out/series.json'}] in the sandbox call, then "
                        "load it with load_series('in/carried/series.json')")
    if carry is not None:
        raise TypeError("carry= belongs to the sandbox call, not to load_series. Pass carry=[{'run_id': ..., 'file': "
                        "'out/series.json'}] when you run the code, then load_series('in/carried/series.json')")
    obj = json.loads(_resolve_input(path).read_text(encoding="utf-8"))
    series, dates = obj.get("series"), obj.get("dates")
    if isinstance(series, list):
        series, dates = {"series": series}, {"series": dates}
    if not isinstance(series, dict):
        raise ValueError("not a file written by save_series: it has no 'series'")
    def build(k: str) -> pd.Series:
        d = (dates or {}).get(k) or []
        idx = pd.DatetimeIndex(pd.to_datetime(d)) if len(d) == len(series[k]) else None
        return pd.Series(series[k], index=idx, name=k, dtype=float)
    single = list(series) == ["series"]
    if single and "name" in obj:                 # a file saved with save_series(..., name=...)
        label = str(obj["name"])
        if name is not None and name not in (label, "series"):
            raise KeyError(f"no series {name!r}; the file holds {[label]}")
        return build("series").rename(label)
    if name is not None:
        if name not in series:
            raise KeyError(f"no series {name!r}; the file holds {sorted(series)}")
        return build(name)
    built = {k: build(k) for k in series}
    return built["series"] if single else built


def save_panel(panel: Any, path: str = "out/series.json", *, freq: str | None = None) -> str:
    """Write every entity of a ``pl.panel`` result in the multi-series format the forecast tools read (one file, one entry
    per entity, with dates). Returns the path written. Read it back with pl.load_series(path, name=<entity>)."""
    if isinstance(panel, tuple):                 # the (series dict, report table) pair, as returned
        panel = panel[0]
    if not isinstance(panel, dict) or not panel:
        raise ValueError("pass the first item of pl.panel(...), the dict of entity series")
    return save_series({str(k): v for k, v in panel.items()}, path, freq=freq)


# ------------------------------------------------------------------------------------ repeating cycles --
PERIOD_MIN_POINTS = 24
PERIOD_MAX_LAG = 800          # longest period looked for; an annual cycle of daily data is 365
PERIOD_PEAK_MIN = 0.25        # autocorrelation a lag needs to count as a repeating cycle


def _detrend(x: np.ndarray) -> np.ndarray:
    t = np.arange(len(x), dtype=float)
    if len(x) < 3:
        return x - x.mean()
    return x - np.polyval(np.polyfit(t, x, 1), t)


def _acf(x: np.ndarray, max_lag: int) -> np.ndarray:
    """Autocorrelation at lags 0..max_lag by FFT, normalised so lag 0 is 1. Zero for a constant series."""
    x = x - x.mean()
    var = float(x @ x)
    if var < 1e-12:
        return np.zeros(max_lag + 1)
    n = len(x)
    size = 1 << (2 * n - 1).bit_length()
    f = np.fft.rfft(x, size)
    ac = np.fft.irfft(f * np.conj(f), size)[: max_lag + 1]
    # Unbiased at long lags: fewer pairs contribute, so scale back up (capped so the tail does not explode).
    scale = n / np.maximum(n - np.arange(max_lag + 1), n * 0.25)
    return (ac / var) * scale


def dominant_period(history: np.ndarray) -> tuple[int | None, float]:
    """The shortest lag at which the detrended history clearly repeats, and how strongly. (None, strength) if none does."""
    x = np.asarray(history, dtype=float)
    if len(x) < PERIOD_MIN_POINTS:
        return None, 0.0
    d = _detrend(x[-20000:])
    max_lag = min(PERIOD_MAX_LAG, len(d) // 3)
    if max_lag < 3:
        return None, 0.0
    ac = _acf(d, max_lag)
    peaks = [k for k in range(2, max_lag) if ac[k] >= PERIOD_PEAK_MIN and ac[k] >= ac[k - 1] and ac[k] > ac[k + 1]]
    if not peaks:
        return None, float(ac[2:max_lag].max()) if max_lag > 2 else 0.0
    best = max(peaks, key=lambda k: ac[k])
    # Prefer the shortest period that explains the cycle: a multiple of it has nearly the same autocorrelation.
    for k in sorted(peaks):
        if k < best and ac[k] >= 0.8 * ac[best] and abs(best / k - round(best / k)) < 0.06:
            return k, float(ac[k])
    return best, float(ac[best])


# ------------------------------------------------------------------------------------ preparing a series --
# What a senior analyst does before any forecast: find the step the data really has, make the calendar complete, decide
# what each gap is, and cut off a tail that was filled rather than observed. Every choice comes back in the report.

_STEP_ALIASES = ((1, "min"), (60, "h"), (1440, "D"))        # minutes per step for the sub-daily and daily cases


def _infer_freq(idx: pd.DatetimeIndex) -> str | None:
    """A pandas alias for the step of a sorted, de-duplicated index; None when the spacing is irregular."""
    if len(idx) < 3:
        return None
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        found = pd.infer_freq(idx)
    if found:
        return found
    # Real files have holes, so infer_freq fails. Use the typical step and check it against the weekday pattern.
    step = pd.Series(idx).diff().dropna().median()
    minutes = step / pd.Timedelta(minutes=1)
    if minutes < 1440:
        for m, alias in _STEP_ALIASES[:2]:
            if abs(minutes - m) < 0.5 * m:
                return alias
        return None
    days = minutes / 1440
    if abs(days - 1) < 0.01:
        weekend = int(((idx.day_of_week >= 5)).sum())
        return "B" if weekend == 0 and len(idx) > 20 else "D"
    if abs(days - 7) < 0.01:
        return "W-" + ["MON", "TUE", "WED", "THU", "FRI", "SAT", "SUN"][int(idx.day_of_week.value_counts().idxmax())]
    if 27 <= days <= 32:
        return "MS" if int(idx.day.value_counts().idxmax()) == 1 else "ME"
    if 88 <= days <= 93:
        return "QS"
    if 360 <= days <= 370:
        return "YS"
    return None


def prepare_series(data: Any, value: str | None = None, time: str | None = None, *, freq: str | None = None,
                   agg: str | None = None, max_short_gap: int | None = None, trim_tail: bool = True,
                   entity: str | None = None, key: Any = None, closed: list[str] | None = None) -> tuple[pd.Series, dict]:
    """A forecast-ready series and a report of every choice made. ``data`` is a frame of strings (give ``value`` and
    ``time``) or a Series with a date index. ``entity`` and ``key`` pick one entity from a long table (for example
    entity="station", key="Beijing"). For several entities at once use pl.panel. When entities share dates and neither
    is given, this raises ValueError, because a sum would mix them.

    The step is inferred from the dates unless ``freq`` is given; rows that share a period are combined with ``agg``.
    Missing periods stay missing until the end: a gap is never turned into zero. Leading and trailing gaps are cut off
    (a tail that is not observed is not data; ``trim_tail=False`` keeps it, and the report says what was dropped). A gap
    inside the series is filled by straight-line interpolation, and one longer than ``max_short_gap`` periods is also
    listed in ``ignore_ranges`` (positions in the returned series) so the forecast can be told not to read it. The report
    also holds duplicates, the observed share, the run of identical values at the end, and the strength of any repeating
    cycle. It decides nothing about meaning: whether a zero or a gap is real is the caller's judgement.

    Closed days (A1-03): ``closed`` names weekdays ("Saturday") or dates (exchange holidays, "2024-12-25") on which the
    source does not trade. Those periods stay empty and are never interpolated, and they add nothing to the total
    (``filled_sum_share`` is the share of the total that came from interpolation). Drop them with ``.dropna()`` before
    save_series to forecast on the trading days; the dates of the rest are kept. When one weekday is empty on most of its
    periods and ``closed`` is not given, this raises and names the choice, because that is a closure, not missing data.
    ``closed=[]`` says the gaps are data to interpolate.

    A business-day ``freq`` with weekend rows is refused: use ``freq='D'`` to keep every observed day (see build_series).
    The report's ``calendar`` is "trading" when closed days were given, "business" for a business-day step, else "calendar";
    the same kind is set on ``s.attrs`` so save_series records it.
    """
    notes: list[str] = []
    agg_given = agg is not None
    if isinstance(data, pd.Series):
        agg = agg or "sum"                      # a Series has no entity column, so the sum default applies
        raw = data.astype(float)
        raw = raw[raw.index.notna()].sort_index()
    else:
        if value is None or time is None:
            raise ValueError("give value= and time= column names when passing a table")
        data, agg = _panel_guard(data, time, value, agg, entity, key)
        num, _ = to_number(data[value])
        ts, _ = to_datetime(data[time])
        raw = pd.Series(num.values, index=ts)       # rows with no value stay: an empty edge is a fact the report states
        raw = raw[raw.index.notna()].sort_index()
    if not isinstance(raw.index, pd.DatetimeIndex):
        raise ValueError("the series has no date index; pass dates or build it with time=")
    n_raw = int(raw.size)
    _refuse_repeated_stamps(raw.dropna().index, agg_given)
    dupes = int(raw.index.duplicated().sum())
    if dupes:
        notes.append(f"{dupes} rows share a date with another row and are combined with {agg}")
    uniq = raw.index.unique()
    freq = freq or _infer_freq(uniq)
    if freq is None:
        raise ValueError("the dates are not evenly spaced enough to infer a step; pass freq= (for example 'D', 'W', 'MS') "
                         "or aggregate first")
    freq = _grid_freq(raw.dropna().index, freq)
    grouped = raw.groupby(pd.Grouper(freq=freq))
    s = grouped.sum(min_count=1) if agg == "sum" else getattr(grouped, agg)()
    s = s.astype(float)
    n_periods = int(s.size)
    observed = s.notna()
    first, last = (int(np.argmax(observed.values)), int(len(s) - np.argmax(observed.values[::-1]) - 1)) if observed.any() else (0, -1)
    lead = first
    tail = int(len(s) - 1 - last) if observed.any() else 0
    if lead:
        notes.append(f"{lead} empty periods at the start were cut off")
    if tail:
        if trim_tail:
            notes.append(f"{tail} empty periods at the end were cut off; the last observed period is {s.index[last].date()}")
        else:
            notes.append(f"{tail} empty periods at the end were kept")
    end = last + 1 if trim_tail else len(s)
    s = s.iloc[first:end]
    if s.size and s.notna().mean() < MIN_OBSERVED_SHARE:
        _refuse_sparse_fill(s, freq)
    if closed is None and _daily_or_finer(freq):
        cands = _closed_candidates(s.index, s.isna().values)
        if cands:
            names = ", ".join(cands)
            raise ValueError(f"{names} has no value on {_CLOSED_SHARE:.0%} or more of its periods, which looks like a closed day "
                             f"rather than missing data. Pass closed={list(cands)!r} to keep those days empty (they are not "
                             "interpolated; drop them with s.dropna() before save_series to forecast on the trading days), or "
                             "closed=[] if the gaps are data to interpolate.")
    closed_mask = _closed_mask(s.index, closed)
    n_closed = int(closed_mask.sum())
    missing = s.isna().values & ~closed_mask            # a closed day is not a gap: nothing is interpolated into it
    gaps: list[list[int]] = []
    i = 0
    while i < len(s):
        if missing[i]:
            j = i
            while j < len(s) and missing[j]:
                j += 1
            gaps.append([i, j])
            i = j
        else:
            i += 1
    n = len(s)
    open_n = n - n_closed
    limit = max_short_gap if max_short_gap is not None else max(3, min(30, n // 50))
    long_gaps = [g for g in gaps if g[1] - g[0] > limit]
    if gaps:
        # Interpolate across the open periods only, then put the closed days back as empty. limit_area="inside": a gap at
        # either end has no observation on one side, so it is not filled. With trim_tail=False the open tail stays NaN and
        # save_series refuses it, rather than repeating the last value (audit: [1,2,3,NaN] -> 3,3).
        s = s[~closed_mask].interpolate(limit_direction="both", limit_area="inside").reindex(s.index)
        notes.append(f"{int(missing.sum())} gaps in {len(gaps)} runs were filled by straight-line interpolation (never zero)")
        if not trim_tail and np.isnan(s.iloc[-1]):
            notes.append("the trailing gap is not observed and is left open: it is not filled, and it cannot be forecast from")
    if n_closed:
        notes.append(f"{n_closed} periods on closed days ({', '.join(str(c) for c in closed)}) are left empty: a closed day is "
                     "not missing data, so nothing is interpolated into it")
    if long_gaps:
        notes.append(f"{len(long_gaps)} runs are longer than {limit} periods; they are filled but listed in ignore_ranges so the "
                     "forecast does not learn from them")
    vals = s.values
    total = float(np.nansum(vals)) if n else 0.0
    filled_sum = float(np.nansum(vals[missing])) if n and missing.any() else 0.0
    filled_share = round(filled_sum / total, 4) if total else 0.0
    run = 0
    while run < len(vals) and vals[-1 - run] == vals[-1]:
        run += 1
    floor_tail = bool(len(vals)) and vals[-1] == float(np.nanmin(vals))
    tail_const = run if run >= 7 and not floor_tail else 0
    if tail_const:
        notes.append(f"the last {tail_const} values are identical; check whether they were observed or carried forward")
    zeros = int((vals == 0).sum()) if len(vals) else 0
    report = {
        "freq": freq, "rows_read": n_raw, "duplicate_dates": dupes, "periods": int(n), "periods_before_trim": n_periods,
        "first": str(s.index[0].date()) if n else None, "last": str(s.index[-1].date()) if n else None,
        "observed_share": round(float(1 - missing.sum() / open_n), 4) if open_n > 0 else 0.0,
        "closed_periods": n_closed, "closed_days": [str(c) for c in (closed or [])], "filled_sum_share": filled_share,
        "calendar": "trading" if closed else ("business" if freq == "B" else "calendar"),
        "gap_runs": len(gaps), "filled_periods": int(missing.sum()), "short_gap_limit": int(limit),
        "long_gaps": [{"start": str(s.index[a].date()), "end": str(s.index[b - 1].date()), "periods": b - a} for a, b in long_gaps],
        "ignore_ranges": [list(g) for g in long_gaps], "dropped_leading": lead, "dropped_tail": tail if trim_tail else 0,
        "constant_tail": tail_const, "zero_periods": zeros, "notes": notes,
    }
    observed_vals = vals[~np.isnan(vals)]                  # closed days are empty; the cycle is found on the values that exist
    period, strength = dominant_period(observed_vals) if len(observed_vals) else (None, 0.0)
    report["period"] = period                              # periods per repeating cycle, found from the data; None when nothing repeats
    report["cycle_strength"] = round(strength, 3) if period else None
    report["cycles_in_history"] = round(len(observed_vals) / period, 1) if period else None
    if period and len(observed_vals) / period < 3:
        notes.append(f"the history holds only {len(observed_vals) / period:.1f} cycles of length {period}; a model cannot learn a cycle from fewer than three")
    s.attrs["calendar"] = report["calendar"]          # save_series records the kind; dropna() keeps the attrs
    return s, report


def panel(df: pd.DataFrame, entity: str, time: str, value: str, freq: str | None = None) -> tuple[dict, pd.DataFrame]:
    """Split a long table by ``entity`` and prepare one series per entity with prepare_series.

    Returns (series, report). ``series`` maps each entity label (as text) to its forecast-ready series. ``report`` has one
    row per entity: rows read, first and last date, periods, gaps filled, zero share, missing share, and the step used.
    Rows with no entity label are left out and the report table says so in ``attrs['dropped_rows']``. Save the whole
    panel for the forecast tools with pl.save_panel(series, 'out/series.json').
    """
    for col in (entity, time, value):
        if col not in df.columns:
            raise ValueError(f"no column {col!r}; the columns are {list(df.columns)[:30]}")
    labelled = df[df[entity].notna()]
    series: dict[str, pd.Series] = {}
    rows: list[dict] = []
    for label, sub in labelled.groupby(entity, sort=True):
        s, rep = prepare_series(sub, value, time, freq=freq)
        series[str(label)] = s
        periods = rep["periods"]
        rows.append({
            "entity": str(label), "rows": rep["rows_read"], "first": rep["first"], "last": rep["last"],
            "periods": periods, "gaps_filled": rep["filled_periods"],
            "zero_share": round(rep["zero_periods"] / periods, 4) if periods else 0.0,
            "missing_share": round(1 - rep["observed_share"], 4), "freq": rep["freq"],
        })
    report = pd.DataFrame(rows, columns=["entity", "rows", "first", "last", "periods", "gaps_filled", "zero_share",
                                         "missing_share", "freq"])
    report.attrs["dropped_rows"] = int(len(df) - len(labelled))
    return series, report


# -------------------------------------------------------------------------- events, holidays, anomalies --
# Known-ahead drivers and suspicious points, built here so the agent states each choice in code and can chart it. These
# report facts and build arrays; whether a spike is an error to ignore or an event to explain is the caller's decision.

def _starts(index: Any, horizon: int, freq: str | None) -> pd.DatetimeIndex:
    """Period start dates for the history plus ``horizon`` further periods."""
    freq = freq_alias(freq)
    idx = pd.DatetimeIndex(pd.to_datetime(index))
    if len(idx) < 3:
        raise ValueError("need at least 3 dates to know the period length")
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")        # pandas is changing what infer_freq returns; the string is what we use
        f = freq or pd.infer_freq(idx)
    if f is None:
        step = pd.Series(idx).diff().dropna().median()
        return idx.append(pd.DatetimeIndex([idx[-1] + step * (i + 1) for i in range(horizon)]))
    off = pd.tseries.frequencies.to_offset(f)
    return idx.append(pd.date_range(idx[-1] + off, periods=horizon, freq=off)) if horizon else idx


def event_covariate(index: Any, events: list[tuple[Any, Any]], horizon: int = 0, freq: str | None = None) -> np.ndarray:
    """A 0 to 1 driver from date ranges, as long as the history plus ``horizon`` periods. ``events`` is a list of
    (first_day, last_day), inclusive. Each period gets the share of its days inside an event, so a one-day holiday is 1.0
    in a daily series and about 0.14 in a weekly one. Pass future event dates too: a planned promotion is known ahead."""
    starts = _starts(index, horizon, freq)
    ends = starts[1:].append(pd.DatetimeIndex([starts[-1] + (starts[-1] - starts[-2])]))
    span = (ends - starts).total_seconds().to_numpy()
    out = np.zeros(len(starts))
    for a, b in events:
        lo, hi = pd.Timestamp(a), pd.Timestamp(b) + pd.Timedelta(days=1)
        overlap = (np.minimum(ends.to_numpy(), hi.to_datetime64()) - np.maximum(starts.to_numpy(), lo.to_datetime64()))
        out += np.clip(overlap / np.timedelta64(1, "s"), 0, None) / span
    return np.clip(out, 0, 1)


def list_holidays(country: str, subdiv: str | None = None, start: Any = None, end: Any = None) -> list[dict]:
    """Public holidays (date and name) for a country code such as 'AU' or 'US', optionally a state, between two dates.
    With no start or end, the default is the whole current calendar year (audit item 23: the old default raised)."""
    import holidays as hol
    today = pd.Timestamp.today()
    lo = pd.Timestamp(start) if start is not None else pd.Timestamp(year=today.year, month=1, day=1)
    hi = pd.Timestamp(end) if end is not None else pd.Timestamp(year=today.year, month=12, day=31)
    cal = hol.country_holidays(country, subdiv=subdiv, years=range(lo.year, hi.year + 1))
    return [{"date": d.isoformat(), "name": nm} for d, nm in sorted(cal.items()) if lo.date() <= d <= hi.date()]


def holiday_covariate(index: Any, country: str, subdiv: str | None = None, horizon: int = 0, freq: str | None = None) -> np.ndarray:
    """Public-holiday driver for the history plus ``horizon`` periods (see event_covariate). Holiday calendars change by
    country and state, and shop-level effects (a market closed, a school break) are not in them."""
    starts = _starts(index, horizon, freq)
    step = starts[-1] - starts[-2]
    days = list_holidays(country, subdiv, starts[0], starts[-1] + step)
    return event_covariate(index, [(d["date"], d["date"]) for d in days], horizon, freq)


def detect_anomalies(values: list[float] | np.ndarray, season: int | None = None, threshold: float = 6.0,
                     dates: Any = None) -> dict:
    """Points far from what their neighbours or the previous cycle imply, scored with a robust spread (median absolute
    deviation) so the outliers themselves do not hide each other. Returns index ranges in the form the forecast tools take
    as ignore_ranges. A promotion or a holiday also looks like a spike: compare these against known events before ignoring
    anything, and ignore only what has no explanation.

    Every path returns the same keys. ``candidates`` is a list of {index, date, value, score, rule}; ``date`` is filled
    when ``dates`` (one per value) is given, else None. ``values`` and ``scores`` run alongside ``indices``."""
    x = np.asarray(values, dtype=float)
    n = len(x)
    date_text = None
    if dates is not None and len(dates) == n:
        date_text = [str(d) for d in dates]
    if n < 12:
        return {"method": "too short to judge", "note": "too short to judge", "threshold": threshold, "robust_spread": None,
                "points": n, "indices": [], "scores": [], "values": [], "candidates": [], "ignore_ranges": []}
    m = int(season) if season and season > 1 and season * 2 <= n else 0
    if m:
        # Full windows only: a truncated window at the edge does not span a whole cycle and would bias the level there.
        level = pd.Series(x).rolling(m, center=True, min_periods=m).median().ffill().bfill().to_numpy()
        detrended = x - level
        phase = np.arange(n) % m
        profile = np.array([np.nanmedian(detrended[phase == r]) for r in range(m)])
        resid = detrended - profile[phase]
        method = f"distance from the level and the usual shape of a cycle (cycle of {m})"
        rule = "robust_z_cycle"
    else:
        k = 7 if n >= 28 else 3
        med = pd.Series(x).rolling(k, center=True, min_periods=1).median().to_numpy()
        resid = x - med
        method = f"distance from the rolling median (window {k})"
        rule = "robust_z_rolling_median"
    ok = np.isfinite(resid)
    centre = np.median(resid[ok])
    mad = np.median(np.abs(resid[ok] - centre))
    scale = 1.4826 * mad if mad > 1e-12 else (np.std(resid[ok]) or 1.0)
    score = np.where(ok, np.abs(resid - centre) / scale, 0.0)
    idx = np.flatnonzero(score > threshold)
    ranges: list[list[int]] = []
    for i in idx.tolist():
        if ranges and i == ranges[-1][1]:
            ranges[-1][1] = i + 1
        else:
            ranges.append([i, i + 1])
    candidates = [{"index": int(i), "date": date_text[i] if date_text else None, "value": round(float(x[i]), 6),
                   "score": round(float(score[i]), 2), "rule": rule} for i in idx.tolist()]
    return {"method": method, "threshold": threshold, "robust_spread": round(float(scale), 6), "points": n,
            "indices": [int(i) for i in idx], "scores": [round(float(score[i]), 2) for i in idx],
            "values": [round(float(x[i]), 6) for i in idx], "candidates": candidates, "ignore_ranges": ranges}


# A lag counts as a repeating cycle when its autocorrelation is at least this high (the same bar as dominant_period).
SEASON_PEAK_MIN = 0.3
# A shorter lag is preferred over the best peak when it is within this much autocorrelation of it: a multiple of a true
# period has nearly the same autocorrelation, so the shortest such lag is the period.
SEASON_TOLERANCE = 0.1


def find_season(values: list[float] | np.ndarray, candidates: tuple[int, ...] | None = None) -> dict:
    """Autocorrelation at likely seasonal lags, after removing a linear trend. Facts for the caller to weigh.

    Missing values stay in place: each lag uses only the pairs where both points are observed, so the time axis is never
    shifted. The strongest lag is the shortest candidate whose autocorrelation is within SEASON_TOLERANCE of the best
    peak, so a multiple of the true period (14 for 7) is not reported. ``strongest_lag`` is None when no lag reaches
    SEASON_PEAK_MIN."""
    x = np.asarray(values, dtype=float).reshape(-1)
    observed = np.isfinite(x)
    n = len(x)
    if int(observed.sum()) < 12:
        return {"note": "too short to test for seasonality", "points": int(observed.sum())}
    t = np.arange(n, dtype=float)
    x = x - np.polyval(np.polyfit(t[observed], x[observed], 1), t)
    x[~observed] = np.nan
    sd = float(np.nanstd(x))
    if not sd > 1e-12:
        return {"note": "series is constant after removing trend"}
    x = (x - np.nanmean(x)) / sd
    cands = candidates or tuple(l for l in (2, 3, 4, 5, 6, 7, 12, 24, 26, 48, 52, 96, 168, 365) if l < n // 2)
    acf: dict[int, float] = {}
    for lag in sorted({int(l) for l in cands if 1 <= int(l) < n}):
        a, b = x[lag:], x[:-lag]
        both = np.isfinite(a) & np.isfinite(b)
        if both.sum() < max(10, n // 4):      # too few pairs at this lag to judge it
            continue
        acf[lag] = round(float(np.mean(a[both] * b[both])), 3)
    if not acf:
        return {"note": "too few paired points at every candidate lag", "points": int(observed.sum())}
    best_lag = max(acf, key=acf.get)
    best = acf[best_lag]
    chosen = best_lag
    if best >= SEASON_PEAK_MIN:
        for lag in sorted(acf):
            if lag < best_lag and acf[lag] >= best - SEASON_TOLERANCE:
                chosen = lag
                break
    found = best >= SEASON_PEAK_MIN
    return {"autocorrelation_by_lag": acf, "strongest_lag": chosen if found else None,
            "strength": acf[chosen] if found else None, "best_peak_lag": best_lag,
            "reading": f"a lag at or above {SEASON_PEAK_MIN} suggests a repeating pattern; none reached it here means no "
                       "usable season"}


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
          future_index: Any = None, shared_y: bool = True, group: str | None = None) -> dict:
    """Draw one chart to ``path`` (PNG, with an SVG and the plotted numbers as CSV beside it) and report on it.

    kind: "line" (data: Series, DataFrame or dict of Series, at most 6 series; or a table with x="<column>" and y="<column>"
    or a list of columns; a long table with one row per x and group value needs group="<column>" to get one line per group),
    "bar" (a Series of category to value; sorted
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
        cols = [y] if isinstance(y, str) else (list(y) if y else [c for c in data.columns if c not in (x, group) and pd.api.types.is_numeric_dtype(data[c])])
        missing = [c for c in cols if c not in data.columns]
        if missing:
            raise ValueError(f"y column(s) {missing} not found; columns are {list(data.columns)}")
        if group is not None:
            if group not in data.columns:
                raise ValueError(f"group={group!r} is not a column; columns are {list(data.columns)}")
            if len(cols) != 1:
                raise ValueError("group= draws one measure, one line per group: pass a single y= column")
            wide = data[[x, group, cols[0]]].copy()
            if wide.duplicated([x, group]).any():
                raise ValueError(f"some {x!r} and {group!r} pairs appear more than once; sum or aggregate the rows first, "
                                 "for example df.groupby([x, group])[y].sum()")
            wide[x] = pd.to_datetime(wide[x]) if pd.api.types.is_datetime64_any_dtype(wide[x]) else wide[x]
            data = wide.pivot(index=x, columns=group, values=cols[0])
            data.columns = [str(c) for c in data.columns]
        else:
            frame = data[[x, *cols]].copy()
            if frame[x].duplicated().any():
                raise ValueError(f"{x!r} repeats, so several rows would be drawn over each other as one line. Pass group='<column>' "
                                 "to draw one line per group, or aggregate the rows first.")
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


# ------------------------------------------------------------------------------------------- figures the answer quotes --
def result(name: Any, value: Any = None, unit: str = "") -> Any:
    """Print a figure the final answer will quote, as ``RESULT name: value [unit]``, and return it.

    Live retail ended partial because the answer quoted a 14-day total, a share, a ratio and rounded weekday means that no cell
    had printed: the model worked them out in its head, and the grounding check (rightly) refused them. Compute the figure
    here, print it with this helper, and quote it as printed. ``pl.result({"total_14d": t, "share": s})`` prints one line each.
    The value is printed to 4 decimals, so a figure written to fewer decimals (or in thousands, "27.9k") still matches it.
    """
    if isinstance(name, dict):
        return {k: result(k, v, unit) for k, v in name.items()}
    v = value
    if hasattr(v, "item") and getattr(v, "ndim", 1) == 0:
        v = v.item()
    if isinstance(v, bool) or not isinstance(v, (int, float)) or v != v:
        print(f"RESULT {name}: {v}" + (f" {unit}" if unit else ""))
        return value
    shown = int(v) if float(v).is_integer() else round(float(v), 4)
    print(f"RESULT {name}: {shown}" + (f" {unit}" if unit else ""))
    return value


# ----------------------------------------------------------------------------- polars names on pandas objects --
# The alias ``pl`` makes models write polars code. Four polars names are added to pandas objects, only where pandas has no
# method of that name, so pandas behaviour is never replaced:
#   DataFrame.null_count() -> one-row frame of missing counts per column; Series.null_count() -> int
#   Series/DataFrame.is_null() and is_not_null() -> isna() and notna()
#   Series/DataFrame.sort(by, descending=False) -> sort_values(by, ascending=not descending)
# Anything else polars-only still raises AttributeError, as pandas does.

def _install_polars_names() -> None:
    def df_null_count(self: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame([self.isna().sum().to_dict()])

    def series_null_count(self: pd.Series) -> int:
        return int(self.isna().sum())

    def df_sort(self: pd.DataFrame, by: Any = None, descending: bool = False, **kw: Any) -> pd.DataFrame:
        if by is None:
            raise ValueError("sort needs by=<column name or list of names>")
        return self.sort_values(by=by, ascending=not descending, **kw)

    def series_sort(self: pd.Series, descending: bool = False, **kw: Any) -> pd.Series:
        return self.sort_values(ascending=not descending, **kw)

    additions = {
        pd.DataFrame: {"null_count": df_null_count, "is_null": lambda self: self.isna(),
                       "is_not_null": lambda self: self.notna(), "sort": df_sort},
        pd.Series: {"null_count": series_null_count, "is_null": lambda self: self.isna(),
                    "is_not_null": lambda self: self.notna(), "sort": series_sort},
    }
    # Not polars, but the same kind of gap: Series.abs() exists, TimedeltaIndex.abs() does not, and "nearest date" code writes
    # (dates - target).abs().argmin() (live: 6 failed cells in one mission).
    additions[pd.TimedeltaIndex] = {"abs": lambda self: abs(self)}
    for cls, methods in additions.items():
        for name, fn in methods.items():
            if not hasattr(cls, name):
                fn.__name__ = name
                fn.__doc__ = f"polars name {name!r} on a pandas object (see prophet_lib, polars names)"
                setattr(cls, name, fn)


_install_polars_names()


try:                                       # inside the sandbox the helpers sit side by side; in tests they are a package
    from prophet_pair import wrap_pairs as _wrap_pairs
except ImportError:                        # pragma: no cover
    from prophet_agent.sandbox.prophet_pair import wrap_pairs as _wrap_pairs
_wrap_pairs(globals())


def __getattr__(name: str):
    """``pl.<name>`` for a statistics helper that lives in prophet_stats (``pl.hampel``, ``pl.changepoints``): models mix the
    two prefixes (live: two cells failed on pl.hampel). Anything else raises the usual AttributeError, naming both modules."""
    if name.startswith("__"):
        raise AttributeError(name)
    import prophet_stats
    if hasattr(prophet_stats, name) and not name.startswith("_"):
        return getattr(prophet_stats, name)
    if name in _AGENT_TOOLS:
        # live retail: six cells called pl.backtest_series
        raise AttributeError(f"{name} is one of the agent's tools, not a function in the analysis sandbox: save the series "
                             f"with pl.save_series, then call {name} as its own step with source={{run_id, file}}")
    raise AttributeError(f"module 'prophet_lib' has no attribute {name!r} (and neither has prophet_stats)")


_AGENT_TOOLS = {"backtest_series", "timesfm_forecast", "forecast_report", "calibrate_intervals", "control_lab", "panel_backtest",
                "build_drivers", "detect_anomalies", "inspect_data", "web_research", "web_fetch"}
