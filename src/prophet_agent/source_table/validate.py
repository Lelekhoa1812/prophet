"""Accept or reject a candidate table. The file name is not an input.

Motivation vs Logic
Motivation: a download can parse as a table and still be a different series saved under the requested name.
Logic: an alias of the claim must appear in the identity cells, a comment, a header, or the URL path. Any other
identity token in those places is a rival, and a rival refuses the name. Silence is not a match. The measure is
taken only from a header the claim named.
"""
from __future__ import annotations

import csv
import io
import re
from urllib.parse import urlsplit

from prophet_agent.source_table.choose import choose_columns, normalise

GENERIC = frozenset({
    "date", "time", "period", "close", "price", "prices", "volume", "open", "high", "low",
    "ticker", "name", "symbol", "entity", "code", "series", "value", "values", "index",
    "year", "month", "day", "id", "unit", "daily", "weekly", "monthly", "annual", "yearly",
    "history", "data", "file", "files", "csv", "json", "xlsx", "table", "download", "export",
    "public", "datasets", "dataset", "raw", "master", "blob", "content", "charts", "chart",
    "finance", "adj", "adjusted",
})
_TICKER = re.compile(r"^[A-Za-z]{2,5}$")
_COMMENT = re.compile(r"^\s*#")


def validate_table(claim: dict, csv_bytes: bytes, *, final_url: str = "", page_text: str | None = None) -> dict:
    """The verdict for these bytes and this claim. ``bound_name`` is set only when the status is accepted."""
    parsed = _parse(csv_bytes)
    if parsed["error"]:
        return _verdict("not_a_table", claim, parsed, final_url, detail=parsed["error"])
    mapping = choose_columns(claim, parsed["headers"])
    tokens, observed = _identity(parsed, mapping, final_url)
    aliases = {normalise(item) for item in claim.get("aliases") or [] if normalise(item)}
    rivals = sorted(token for token in tokens if token not in aliases and token not in GENERIC)
    matched = sorted(token for token in tokens if token in aliases)
    if not tokens:
        status, detail = "entity_unstated", "the file does not name an entity"
    elif rivals:
        status, detail = "wrong_entity", "the file names a different entity"
    elif not matched:
        status, detail = "entity_unstated", "the file does not name the claimed entity"
    elif mapping["unmapped"]:
        status, detail = "schema_gap", "a required column is not in the header: " + ", ".join(mapping["unmapped"])
    elif page_text is not None and not _literals(parsed, mapping, page_text):
        status, detail = "unsupported_value", "a measure cell is not in the page text"
    else:
        status, detail = "accepted", "entity and columns match the claim"
    verdict = _verdict(status, claim, parsed, final_url, detail=detail, observed=observed, rivals=rivals, mapping=mapping)
    time_index = (mapping or {}).get("mapped", {}).get("time")
    if isinstance(time_index, int):
        verdict["distinct_times"] = _distinct_count(parsed.get("rows") or [], time_index)
    if status == "accepted":
        verdict["bound_name"] = claim["series_name"]
    return verdict


def _distinct_count(rows: list, index: int) -> int:
    seen: set[str] = set()
    for row in rows:
        if isinstance(row, list) and index < len(row):
            cell = str(row[index]).strip()
            if cell:
                seen.add(cell)
    return len(seen)


def _verdict(status, claim, parsed, final_url, *, detail, observed="", rivals=None, mapping=None) -> dict:
    return {
        "status": status,
        "detail": detail,
        "bound_name": None,
        "requested_entity": claim.get("entity") or "",
        "observed_entity": observed,
        "rivals": rivals or [],
        "series_name": claim.get("series_name") or "",
        "claim_id": claim.get("claim_id") or "",
        "columns": parsed.get("headers") or [],
        "row_count": len(parsed.get("rows") or []),
        "final_url": final_url,
        "mapped": (mapping or {}).get("mapped") or {},
    }


def _parse(csv_bytes: bytes) -> dict:
    if not csv_bytes:
        return {"error": "the file is empty", "headers": [], "rows": [], "comments": []}
    try:
        text = csv_bytes.decode("utf-8-sig")
    except UnicodeDecodeError:
        return {"error": "the file is not UTF-8 text", "headers": [], "rows": [], "comments": []}
    comments, lines = [], []
    for line in text.splitlines():
        if _COMMENT.match(line):
            comments.append(line.strip())
        elif line.strip():
            lines.append(line)
    if not lines:
        return {"error": "the file has no data rows", "headers": [], "rows": [], "comments": comments}
    try:
        rows = list(csv.reader(io.StringIO("\n".join(lines))))
    except csv.Error:
        return {"error": "the file is not a CSV table", "headers": [], "rows": [], "comments": comments}
    if len(rows) < 2 or len(rows[0]) < 2:
        return {"error": "the file is not a table", "headers": rows[0] if rows else [], "rows": [], "comments": comments}
    return {"error": "", "headers": rows[0], "rows": rows[1:], "comments": comments}


def _identity(parsed: dict, mapping: dict, final_url: str) -> tuple[set[str], str]:
    headers = parsed["headers"]
    entity_indexes = _entity_indexes(headers, mapping)
    name_indexes = [i for i, header in enumerate(headers) if normalise(header) == "name"]
    tokens: set[str] = set()
    observed = ""
    for row in parsed["rows"][:20]:
        for index in name_indexes:
            if index < len(row):
                token, original = _cell_token(row[index])
                if token:
                    tokens.add(token)
                    observed = observed or original
        for index in entity_indexes:
            if index in name_indexes or index >= len(row):
                continue
            token, original = _cell_token(row[index])
            if token:
                tokens.add(token)
                observed = observed or original
    for comment in parsed["comments"]:
        for token, original in _line_tokens(comment):
            tokens.add(token)
            observed = observed or original
    mapped_indexes = set((mapping.get("mapped") or {}).values())
    for index, header in enumerate(headers):
        if index in mapped_indexes:
            continue
        token = normalise(header)
        if token and token not in GENERIC and _TICKER.match(header.strip()):
            tokens.add(token)
            observed = observed or header.strip()
    path = urlsplit(final_url or "").path or (final_url or "")
    for segment in re.split(r"[^A-Za-z0-9]+", path):
        if _TICKER.match(segment) and normalise(segment) not in GENERIC:
            tokens.add(normalise(segment))
            observed = observed or segment
    return tokens, observed


def _entity_indexes(headers: list[str], mapping: dict) -> list[int]:
    indexes = []
    mapped = (mapping.get("mapped") or {}).get("entity")
    if isinstance(mapped, int):
        indexes.append(mapped)
    for index, header in enumerate(headers):
        if normalise(header) in {"ticker", "symbol", "code", "entity", "series", "name"} and index not in indexes:
            indexes.append(index)
    return indexes


def _cell_token(cell: str) -> tuple[str, str]:
    original = " ".join(str(cell).split())
    token = normalise(original)
    if not token or token in GENERIC or _is_number(original):
        return "", ""
    return token, original


def _line_tokens(line: str) -> list[tuple[str, str]]:
    found = []
    for word in re.findall(r"[A-Za-z][A-Za-z0-9]{1,}", line):
        token = normalise(word)
        if token and token not in GENERIC and (_TICKER.match(word) or word[:1].isupper()):
            found.append((token, word))
    return found


def _is_number(cell: str) -> bool:
    try:
        float(cell.replace(",", "").replace("%", ""))
    except ValueError:
        return False
    return True


def _literals(parsed: dict, mapping: dict, page_text: str) -> bool:
    index = (mapping.get("mapped") or {}).get("measure")
    if not isinstance(index, int):
        return True
    page = " ".join(page_text.split())
    for row in parsed["rows"][:50]:
        if index >= len(row):
            continue
        cell = " ".join(str(row[index]).split())
        if cell and cell not in page:
            return False
    return True
