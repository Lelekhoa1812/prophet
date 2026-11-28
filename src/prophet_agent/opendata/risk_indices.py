"""Policy-uncertainty and geopolitical-risk indices, read without a key as dated monthly series.

Motivation vs Logic
Motivation (Mission 10 batch 2, owner review 2026-11-28): the forecasts must account for political, geopolitical and policy
risk. Two public families of indices measure it from newspaper text: the Caldara-Iacoviello geopolitical risk indices (GPR:
global, its threats and acts parts, and one per country) and the Baker-Bloom-Davis economic policy uncertainty indices (EPU:
the United States, Australia, global). Checked 2026-11-28: the GPR monthly workbook runs 1985-01 to 2026-10 (a legacy .xls,
115 columns including GPRC_AUS, GPRC_VNM, GPRC_USA and GPRC_CHN; the site updates it weekly); the Australia EPU workbook runs
1998-01 to 2026-10 and the global one 1997-01 to 2026-07 (.xlsx); the US EPU is on FRED as USEPUINDXM from 1985-01.
Logic: each index has a fixed address and parser. A value is dated at the last day of its month, when it becomes known, so
a forecast origin reads only months that had ended by its cutoff (forecast.regimes ranks it against its own history up to
then). The workbooks are read with python-calamine, which reads .xls and .xlsx. DOMAIN_INDEX names, before anything is
measured, the one index whose regime conditions each domain's band, so a search over indices cannot manufacture a regime
effect; the dashboard shows the others beside it.
"""
from __future__ import annotations

import calendar
import csv
import datetime as dt
import io
from collections.abc import Callable
from typing import Any

GPR_MONTHLY = "https://www.matteoiacoviello.com/gpr_files/data_gpr_export.xls"
EPU_AUSTRALIA = "https://www.policyuncertainty.com/media/Australia_Policy_Uncertainty_Data.xlsx"
EPU_GLOBAL = "https://www.policyuncertainty.com/media/Global_Policy_Uncertainty_Data.xlsx"
EPU_US = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=USEPUINDXM"

# key: (label, url, parser, column)
INDICES: dict[str, tuple[str, str, str, str]] = {
    "gpr": ("geopolitical risk, global (Caldara-Iacoviello GPR)", GPR_MONTHLY, "gpr", "GPR"),
    "gpr_threats": ("geopolitical threats (GPR threats)", GPR_MONTHLY, "gpr", "GPRT"),
    "gpr_acts": ("geopolitical acts (GPR acts)", GPR_MONTHLY, "gpr", "GPRA"),
    "gpr_aus": ("geopolitical risk, Australia (GPR country index)", GPR_MONTHLY, "gpr", "GPRC_AUS"),
    "gpr_vnm": ("geopolitical risk, Vietnam (GPR country index)", GPR_MONTHLY, "gpr", "GPRC_VNM"),
    "gpr_usa": ("geopolitical risk, United States (GPR country index)", GPR_MONTHLY, "gpr", "GPRC_USA"),
    "gpr_chn": ("geopolitical risk, China (GPR country index)", GPR_MONTHLY, "gpr", "GPRC_CHN"),
    "epu_us": ("economic policy uncertainty, United States (Baker-Bloom-Davis EPU)", EPU_US, "fred", "USEPUINDXM"),
    "epu_au": ("economic policy uncertainty, Australia (Baker-Bloom-Davis EPU)", EPU_AUSTRALIA, "epu", "2"),
    "epu_global": ("economic policy uncertainty, global (GDP-weighted EPU)", EPU_GLOBAL, "epu", "2"),
}

# The index whose regime conditions each domain's band, fixed before measuring (see the module note). Policy rates and labour
# markets respond to domestic policy uncertainty; gold is the asset bought against geopolitical risk; US equities price US
# policy uncertainty.
DOMAIN_INDEX: dict[tuple[str, str | None], str] = {
    ("policy_rate", "Australia"): "epu_au",
    ("labour_market", "Australia"): "epu_au",
    ("labour_market", "Vietnam"): "epu_global",
    ("precious_metal", None): "gpr",
    ("equity_index", None): "epu_us",
}


def domain_index(domain: str | None, place: str | None) -> str | None:
    return DOMAIN_INDEX.get((domain or "", place)) or DOMAIN_INDEX.get((domain or "", None))


def dashboard_keys(domain: str | None, place: str | None) -> list[str]:
    """The indices shown for a target: the global GPR and its parts, global EPU, and the place's own indices."""
    keys = ["gpr", "gpr_threats", "gpr_acts", "epu_global"]
    extra = {"Australia": ["epu_au", "gpr_aus"], "Vietnam": ["gpr_vnm", "gpr_chn"],
             "United States": ["epu_us", "gpr_usa"]}.get(place or "", [])
    if domain in ("precious_metal", "equity_index") and not extra:
        extra = ["epu_us", "gpr_usa"]
    own = domain_index(domain, place)
    return list(dict.fromkeys(([own] if own else []) + keys + extra))


def _month_end(year: int, month: int) -> str:
    return dt.date(year, month, calendar.monthrange(year, month)[1]).isoformat()


def _float(v: Any) -> float | None:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if f == f else None


def _sheet_rows(body: bytes) -> list[list[Any]]:
    import python_calamine as pc
    wb = pc.CalamineWorkbook.from_filelike(io.BytesIO(body))
    return wb.get_sheet_by_index(0).to_python()


def parse_gpr(body: bytes, column: str) -> list[tuple[str, float]]:
    """(month-end date, value) for one column of the GPR monthly workbook (first column "month", ISO dates)."""
    rows = _sheet_rows(body)
    header = [str(c).strip() for c in rows[0]]
    if "month" not in header or column not in header:
        raise ValueError(f"the GPR workbook has no {column!r} column; the layout may have changed")
    m, c = header.index("month"), header.index(column)
    out = []
    for r in rows[1:]:
        day = r[m] if isinstance(r[m], (dt.date, dt.datetime)) else None
        if day is None:
            try:
                day = dt.date.fromisoformat(str(r[m])[:10])
            except ValueError:
                continue
        v = _float(r[c]) if c < len(r) else None
        if v is not None:
            out.append((_month_end(day.year, day.month), v))
    if not out:
        raise ValueError(f"the GPR workbook holds no values for {column}")
    return out


def parse_epu(body: bytes, column: int = 2) -> list[tuple[str, float]]:
    """(month-end date, value) from a policyuncertainty.com workbook laid out as year, month, index (notes rows skipped)."""
    out = []
    for r in _sheet_rows(body)[1:]:
        y, mth = _float(r[0]) if r else None, _float(r[1]) if len(r) > 1 else None
        v = _float(r[column]) if len(r) > column else None
        if y is None or mth is None or v is None or not (1 <= int(mth) <= 12) or y < 1900:
            continue
        out.append((_month_end(int(y), int(mth)), v))
    if not out:
        raise ValueError("the EPU workbook holds no year, month and value rows; the layout may have changed")
    return out


def parse_fred_monthly(body: bytes, sid: str) -> list[tuple[str, float]]:
    out = []
    for r in csv.DictReader(io.StringIO(body.decode("utf-8-sig", errors="replace"))):
        day, v = r.get("observation_date") or r.get("DATE"), _float(r.get(sid))
        if day and v is not None:
            d = dt.date.fromisoformat(day[:10])
            out.append((_month_end(d.year, d.month), v))
    if not out:
        raise ValueError(f"FRED returned no values for {sid}")
    return out


def fetch_index(key: str, fetch: Callable[[str], bytes], cache: dict[str, bytes] | None = None) -> dict[str, Any]:
    """{key, label, url, rows, last} for one index; ``cache`` keeps a workbook shared by several indices."""
    if key not in INDICES:
        raise ValueError(f"index must be one of {sorted(INDICES)}")
    label, url, parser, column = INDICES[key]
    cache = {} if cache is None else cache
    if url not in cache:
        cache[url] = fetch(url)
    body = cache[url]
    rows = (parse_gpr(body, column) if parser == "gpr" else parse_epu(body, int(column)) if parser == "epu"
            else parse_fred_monthly(body, column))
    return {"key": key, "label": label, "url": url, "rows": rows, "last": rows[-1][0]}
