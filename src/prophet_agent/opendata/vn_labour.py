"""Vietnam's quarterly unemployment rate (working age) from the national statistics office's quarterly labour releases.

Motivation vs Logic
Motivation (Mission 10, M10-4): every VNUNEMP mission forecast from ILOSTAT, whose Vietnam series ends in 2024-Q4 and is a
different concept (age 15 and over), while the statistics office (nso.gov.vn) publishes the working-age rate every quarter,
about a week after the quarter ends. Its releases are web pages with a workbook attached, not an API, and the address
changes every quarter.
Logic: release pages are found through the site's own search API (two queries) and a list of known release pages, so a
search that misses one still leaves the history. Each page links a workbook whose first sheet ("LĐ Quy") has a header row of
period labels ("Quý III năm 2026") and a row "Tỷ lệ thất nghiệp trong độ tuổi lao động (%)"; the value under each quarter
label is read (half-year and nine-month columns are skipped). A page without a workbook gives the rate its text states for
its own quarter. When releases disagree on a quarter, the newest release wins, since a later release carries revisions.
Every value keeps the address it came from. The host sometimes fails to resolve; each fetch is retried.
"""
from __future__ import annotations

import html
import io
import re
import time
from collections.abc import Callable
from typing import Any
from urllib.parse import quote

ROMAN = {"i": 1, "ii": 2, "iii": 3, "iv": 4}
HOST = "https://www.nso.gov.vn"
SEARCH = HOST + "/wp-json/wp/v2/posts?search={q}&per_page=100&page={p}&_fields=link,date"
QUERIES = ("lao dong viec lam quy", "thong cao bao chi tinh hinh lao dong viec lam")
RELEASE = re.compile(r"https://www\.nso\.gov\.vn/[a-z-]+/(\d{4})/(\d{2})/thong-cao-bao-chi-[a-z-]*?lao-dong-viec-lam-"
                     r"quy-(i{1,3}|iv)\b[a-z0-9-]*?-nam-(\d{4})/?")
WORKBOOK = re.compile(r"https://www\.nso\.gov\.vn/wp-content/uploads/\d{4}/\d{2}/[A-Za-z0-9_.\-]+\.xlsx")
QUARTER_LABEL = re.compile(r"^\s*Quý\s+(I{1,3}|IV)\s+năm\s+(\d{4})\s*$", re.I)
RATE_ROW = re.compile(r"^\s*Tỷ lệ thất nghiệp trong độ tuổi lao động", re.I)
TEXT_RATE = re.compile(r"[Tt]ỷ lệ thất nghiệp (?:của lực lượng lao động )?trong độ tuổi(?: lao động)?[^.%]{0,120}?"
                       r"(\d+,\d+)\s*%")
# Release pages found on 2026-11-28; the December 2023 one carries every quarter from 2020-Q1 to 2023-Q4 in its workbook.
KNOWN_RELEASES = (
    HOST + "/du-lieu-va-so-lieu-thong-ke/2023/01/thong-cao-bao-chi-tinh-hinh-lao-dong-viec-lam-quy-iv-va-nam-2022/",
    HOST + "/du-lieu-va-so-lieu-thong-ke/2023/12/thong-cao-bao-chi-ve-tinh-hinh-dan-so-lao-dong-viec-lam-quy-iv-va-nam-2023/",
    HOST + "/du-lieu-va-so-lieu-thong-ke/2024/03/thong-cao-bao-chi-tinh-hinh-lao-dong-viec-lam-quy-i-nam-2024/",
    HOST + "/tin-tuc-thong-ke/2025/04/thong-cao-bao-chi-ve-tinh-hinh-lao-dong-viec-lam-quy-i-nam-2025/",
    HOST + "/du-lieu-va-so-lieu-thong-ke/2025/07/thong-cao-bao-chi-ve-tinh-hinh-lao-dong-viec-lam-quy-ii-va-6-thang-dau-nam-2025/",
    HOST + "/du-lieu-va-so-lieu-thong-ke/2025/10/thong-cao-bao-chi-ve-tinh-hinh-lao-dong-viec-lam-quy-iii-va-9-thang-nam-2025/",
    HOST + "/tin-tuc-thong-ke/2026/01/thong-cao-bao-chi-ve-tinh-hinh-dan-so-lao-dong-viec-lam-quy-iv-va-nam-2025/",
    HOST + "/tin-tuc-thong-ke/2026/04/thong-cao-bao-chi-ve-tinh-hinh-lao-dong-viec-lam-quy-i-nam-2026/",
    HOST + "/tin-tuc-thong-ke/2026/07/thong-cao-bao-chi-ve-tinh-hinh-lao-dong-viec-lam-quy-ii-va-6-thang-dau-nam-2026/",
    HOST + "/tin-tuc-thong-ke/2026/10/thong-cao-bao-chi-ve-tinh-hinh-lao-dong-viec-lam-quy-iii-va-9-thang-nam-2026/",
)
RETRIES = 3
SOURCE_LABEL = "General Statistics Office of Vietnam (nso.gov.vn), quarterly labour and employment releases"


def _roman(text: str) -> int:
    return ROMAN[text.lower()]


def parse_workbook(data: bytes) -> dict[tuple[int, int], float]:
    """{(year, quarter): working-age unemployment rate} from a release workbook. Empty when the layout is not found."""
    import openpyxl
    wb = openpyxl.load_workbook(io.BytesIO(data), read_only=True, data_only=True)
    out: dict[tuple[int, int], float] = {}
    for ws in wb.worksheets:
        rows = list(ws.iter_rows(values_only=True))
        header = next((r for r in rows if sum(1 for c in r if c and QUARTER_LABEL.match(" ".join(str(c).split()))) >= 1),
                      None)
        target = next((r for r in rows if r and r[0] and RATE_ROW.match(str(r[0]))), None)
        if header is None or target is None:
            continue
        for k, cell in enumerate(header):
            m = QUARTER_LABEL.match(" ".join(str(cell).split())) if cell else None
            if not m or k >= len(target):
                continue
            try:
                v = float(target[k])
            except (TypeError, ValueError):
                continue
            if 0 < v < 50:
                out[(int(m.group(2)), _roman(m.group(1)))] = v
        if out:
            break
    return out


def release_quarter(url: str) -> tuple[int, int] | None:
    m = RELEASE.match(url)
    return (int(m.group(4)), _roman(m.group(3))) if m else None


def parse_release_text(page: str) -> float | None:
    """The working-age rate the release text states first (its own quarter), or None."""
    text = re.sub(r"<[^>]+>", " ", html.unescape(page))
    m = TEXT_RATE.search(text)
    return float(m.group(1).replace(",", ".")) if m else None


def _retrying(get: Callable[[str], bytes]) -> Callable[[str], bytes]:
    def fetch(url: str) -> bytes:
        last: Exception | None = None
        for k in range(RETRIES):
            try:
                return get(url)
            except Exception as exc:      # the host sometimes fails to resolve; a later try usually works
                last = exc
                time.sleep(min(2.0 * (k + 1), 5.0))
        raise last if last else RuntimeError("fetch failed")
    return fetch


def find_releases(get: Callable[[str], bytes]) -> tuple[list[str], list[str]]:
    """Release page addresses (the known ones plus those the site search lists), oldest first, and search failures."""
    import json
    found = set(KNOWN_RELEASES)
    failures = []
    for q in QUERIES:
        for p in (1, 2):
            url = SEARCH.format(q=quote(q), p=p)
            try:
                docs = json.loads(get(url).decode("utf-8", errors="replace"))
            except Exception as exc:
                failures.append(f"{url}: {str(exc)[:120]}")
                break
            if not isinstance(docs, list) or not docs:
                break
            found.update(d["link"] for d in docs if isinstance(d, dict) and RELEASE.match(str(d.get("link", ""))))
    ordered = sorted((u for u in found if RELEASE.match(u)), key=lambda u: RELEASE.match(u).group(1, 2))
    return ordered, failures


def quarterly_unemployment(get: Callable[[str], bytes]) -> dict[str, Any]:
    """The quarterly series, oldest first, each value with the release it came from."""
    fetch = _retrying(get)
    releases, failures = find_releases(fetch)
    values: dict[tuple[int, int], tuple[float, str, str]] = {}      # quarter -> (value, source address, release month)
    used = []
    for page_url in releases:
        stamp = "-".join(RELEASE.match(page_url).group(1, 2))
        try:
            page = fetch(page_url).decode("utf-8", errors="replace")
        except Exception as exc:
            failures.append(f"{page_url}: {str(exc)[:120]}")
            continue
        got: dict[tuple[int, int], float] = {}
        source = page_url
        for book in sorted(set(WORKBOOK.findall(page))):
            try:
                got = parse_workbook(fetch(book))
            except Exception as exc:
                failures.append(f"{book}: {str(exc)[:120]}")
                continue
            if got:
                source = book
                break
        if not got:
            own, rate = release_quarter(page_url), parse_release_text(page)
            if own and rate is not None:
                got = {own: rate}
        for q, v in got.items():
            if q not in values or values[q][2] <= stamp:
                values[q] = (v, source, stamp)
        if got:
            used.append({"release": page_url, "source": source, "quarters": len(got)})
    if not values:
        raise ValueError("no quarterly unemployment rate could be read from the statistics office's releases: "
                         + "; ".join(failures[:4]))
    keys = sorted(values)
    gaps = [f"{y}-Q{q}" for (y, q) in _all_quarters(keys[0], keys[-1]) if (y, q) not in values]
    series = [{"period": f"{y}-Q{q}", "date": f"{y}-{3 * q - 2:02d}-01", "value": round(values[(y, q)][0], 4),
               "source": values[(y, q)][1]} for (y, q) in keys]
    return {"series": series, "first": series[0]["period"], "last": series[-1]["period"], "points": len(series),
            "gaps": gaps, "releases": used, "failures": failures, "measure": "unemployment rate of the working-age labour "
            "force, per cent, quarterly (statistics office definition; not the ILO 15-and-over series)",
            "source_label": SOURCE_LABEL}


def _all_quarters(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    out, (y, q) = [], a
    while (y, q) <= b:
        out.append((y, q))
        y, q = (y + 1, 1) if q == 4 else (y, q + 1)
    return out
