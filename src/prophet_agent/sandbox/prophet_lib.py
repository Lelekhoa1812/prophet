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

MAX_PROFILE_COLUMNS = 40
_NA_TOKENS = {"", "na", "n/a", "nan", "null", "none", "nil", "-", "--", "?", "missing", "#n/a", "#value!"}
_ENCODINGS = ("utf-8-sig", "utf-16", "cp1252", "latin-1")


# --------------------------------------------------------------------------------------------------- reading --

def read_text(path: str | Path) -> tuple[str, str]:
    """Decode a file, trying common encodings. Returns (text, encoding used)."""
    raw = Path(path).read_bytes()
    for enc in _ENCODINGS:
        try:
            return raw.decode(enc), enc
        except (UnicodeDecodeError, UnicodeError):
            continue
    return raw.decode("utf-8", "replace"), "utf-8 (with replacements)"


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
        rows = [r for r in csv.reader(lines, delimiter=delim) if r and any(c.strip() for c in r)]
    return _rows_to_frame(rows, report, None, pad=False), report


def _rows_to_frame(rows: list[list[str]], report: dict, block: int | None, pad: bool = True) -> pd.DataFrame:
    raw = rows
    blocks = _blocks(raw)
    sel = range(len(raw))
    if len(blocks) > 1:
        report["tables_on_sheet"] = [{"rows": f"{a + 1}-{b + 1}", "first_cells": [c for c in raw[a][:4]]} for a, b in blocks]
        pick = block if block is not None else max(range(len(blocks)), key=lambda k: blocks[k][1] - blocks[k][0])
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
    t = t.str.replace(r"^\((.*)\)$", r"\1", regex=True).str.replace(r"[^\d,.\-+eE%]", "", regex=True)
    pct = t.str.endswith("%")
    t = t.str.rstrip("%")
    # Decimal comma: "1.234,56" or "12,5". A lone comma followed by 1-2 digits at the end is a decimal; otherwise a thousands mark.
    both = t.str.contains(",") & t.str.contains(r"\.")
    t = t.where(~both, t.str.replace(".", "", regex=False).str.replace(",", ".", regex=False))
    only_comma = t.str.contains(",") & ~t.str.contains(r"\.")
    dec = only_comma & t.str.match(r"^-?\d+,\d{1,2}$")
    t = t.where(~dec, t.str.replace(",", ".", regex=False))
    t = t.str.replace(",", "", regex=False)
    num = pd.to_numeric(t.mask(is_na | looks_like_date_or_clock), errors="coerce")
    num = num.where(~neg, -num.abs())
    num = num.where(~pct, num / 100.0)
    info = {"missing_tokens": int(is_na.sum()), "unparsed": int((num.isna() & ~is_na & ~looks_like_date_or_clock).sum()),
            "decimal_comma_values": int(dec.sum() + both.sum()), "percent_values": int(pct.sum()),
            "parenthesised_negatives": int(neg.sum())}
    return num, info


def to_datetime(s: pd.Series) -> tuple[pd.Series, dict]:
    """Parse text to timestamps and say how. Reports when day-first and month-first both fit (ambiguous), so the
    caller can ask rather than guess."""
    t = s.astype(str).str.strip()
    t = t.mask(t.str.lower().isin(_NA_TOKENS))
    info: dict[str, Any] = {"ambiguous_day_month": False}
    head = t.dropna().head(60)
    if len(head) and max(pd.to_datetime(head, errors="coerce", dayfirst=df).notna().mean() for df in (False, True)) < 0.8 \
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
    day = pd.to_datetime(t, errors="coerce", dayfirst=True)
    mon = pd.to_datetime(t, errors="coerce", dayfirst=False)
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
        step = d.mode().iloc[0]
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
                if info["unique"] <= max(50, 0.2 * len(df)):
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
        s = getattr(s.groupby(pd.Grouper(freq=freq)), agg)()
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
