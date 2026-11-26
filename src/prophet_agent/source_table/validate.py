"""Accept or reject a candidate table. The file name is not an input.

Motivation vs Logic
Motivation: a download can parse as a table and still be a different series saved under the requested name. A table
can also be the right series with the wrong number of rows: the EPL table that passed had 11 of 20 clubs.
Logic: an alias of the claim must appear in the identity cells, a comment, a header, or the URL path. Any other
identity token in those places is a rival, and a rival refuses the name. Silence is not a match. The measure is
taken only from a header the claim named. A league claim matches its entity in the page text or the URL instead, since
its rows are clubs, not the series. The row count is then compared with the expected count (see expect.py): a table
with fewer rows, or with a named entity missing, is short and is not accepted; one with more rows is long and is not
accepted; a league table with no count to compare with is count_unstated and is not accepted.
"""
from __future__ import annotations

import csv
import io
import re
from urllib.parse import urlsplit

from prophet_agent.source_table.choose import choose_columns, normalise
from prophet_agent.source_table.expect import expected_count

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
_RANK_HEADER = re.compile(r"^\s*#\s*(?:,|$)")    # a first cell that is only "#" is a rank column header, not a comment


def validate_table(claim: dict, csv_bytes: bytes, *, final_url: str = "", page_text: str | None = None,
                   reference_csvs: list[bytes] | None = None) -> dict:
    """The verdict for these bytes and this claim. ``bound_name`` is set only when the status is accepted.

    ``reference_csvs`` are results tables of the same competition, used only to count its teams when the claim is a
    league claim and gives no count of its own.
    """
    parsed = _parse(csv_bytes)
    if parsed["error"]:
        return _verdict("not_a_table", claim, parsed, final_url, detail=parsed["error"])
    league = claim.get("table_kind") == "league"
    mapping = choose_columns(claim, parsed["headers"])
    aliases = {normalise(item) for item in claim.get("aliases") or [] if normalise(item)}
    if league:
        matched = _league_matches(claim, page_text, final_url)
        tokens, rivals, observed = set(), [], (matched[0] if matched else "")
    else:
        tokens, observed = _identity(parsed, mapping, final_url)
        rivals = sorted(token for token in tokens if token not in aliases and token not in GENERIC)
        matched = sorted(token for token in tokens if token in aliases)
    status, detail = _entity_refusal(league, tokens, rivals, matched) or ("", "")
    if not status and mapping["unmapped"]:
        status, detail = "schema_gap", "a required column is not in the header: " + ", ".join(mapping["unmapped"])
    if not status and page_text is not None and not _literals(parsed, mapping, page_text):
        status, detail = "unsupported_value", "a measure cell is not in the page text"
    count = _count(claim, parsed, page_text, reference_csvs)
    if not status:
        status, detail = _count_refusal(league, count)
    verdict = _verdict(status or "accepted", claim, parsed, final_url,
                       detail=detail or "entity and columns match the claim",
                       observed=observed, rivals=rivals, mapping=mapping)
    verdict.update(count["fields"])
    time_index = (mapping or {}).get("mapped", {}).get("time")
    if isinstance(time_index, int):
        verdict["distinct_times"] = _distinct_count(parsed.get("rows") or [], time_index)
    if verdict["status"] == "accepted":
        verdict["bound_name"] = claim["series_name"]
    return verdict


def _entity_refusal(league: bool, tokens: set, rivals: list, matched: list) -> tuple[str, str] | None:
    """The entity refusal, or None when the entity matches.

    Motivation vs Logic
    Motivation: a table with no entity and a table with a rival entity must be refused for different reasons, so the
    next step can ask the right question.
    Logic: a league claim needs its entity named by the page; any other claim needs an alias among the identity tokens
    and no rival token.
    """
    if league:
        return None if matched else ("entity_unstated", "the page does not name the claimed entity")
    if not tokens:
        return "entity_unstated", "the file does not name an entity"
    if rivals:
        return "wrong_entity", "the file names a different entity"
    if not matched:
        return "entity_unstated", "the file does not name the claimed entity"
    return None


def _count(claim: dict, parsed: dict, page_text: str | None, reference_csvs: list[bytes] | None) -> dict:
    """Compare the row count and the named entities with the expected ones. Returns the fields and the cardinality.

    Motivation vs Logic
    Motivation: the check must say how many rows it expected and how many it found, and must name the clubs that are
    absent, so a short table is shown to be short rather than silently accepted.
    Logic: the expected count comes from expect.py. Fewer rows than expected is short; more is long; the same is complete.
    Any expected entity that no cell of the table holds is missing, and a missing entity makes the table short.
    """
    rows = parsed.get("rows") or []
    expected = expected_count(claim, page_text=page_text, reference_csvs=reference_csvs or [])
    target = expected["expected_rows"]
    if target is None:
        cardinality = "unknown"
    elif len(rows) < target:
        cardinality = "short"
    elif len(rows) > target:
        cardinality = "long"
    else:
        cardinality = "complete"
    cells = {normalise(cell) for row in rows for cell in row if normalise(cell)}
    missing = [name for name in claim.get("expected_entities") or [] if normalise(name) not in cells]
    fields = {"cardinality": cardinality, "expected_rows": target, "expected_source": expected["expected_source"],
              "expected_evidence": expected["expected_evidence"], "expected_conflict": expected["expected_conflict"],
              "missing_entities": missing}
    return {"cardinality": cardinality, "target": target, "source": expected["expected_source"], "missing": missing,
            "rows": len(rows), "conflict": expected["expected_conflict"], "fields": fields}


def _count_refusal(league: bool, count: dict) -> tuple[str, str]:
    """A refusal from the row count, or ("", "") when the count does not refuse the table.

    Motivation vs Logic
    Motivation: a short table must not be bound as complete, and a league table without a count must not be bound at all.
    Logic: short (or a missing named entity) and long refuse; an unknown count refuses only a league claim.
    """
    if count["cardinality"] == "short" or count["missing"]:
        parts = [f"{count['rows']} rows, {count['target']} expected ({count['source']})"] if count["target"] else []
        if count["missing"]:
            parts.append("missing: " + ", ".join(count["missing"]))
        return "short", "the table is short: " + "; ".join(parts)
    if count["cardinality"] == "long":
        return "long", f"{count['rows']} rows, {count['target']} expected ({count['source']})"
    if league and count["cardinality"] == "unknown":
        detail = "a league table needs an expected row count: state expected_rows, expected_entities, or a page count"
        if count["conflict"]:
            detail = "the sources disagree on the row count, so no count is used"
        return "count_unstated", detail
    return "", ""


def _league_matches(claim: dict, page_text: str | None, final_url: str) -> list[str]:
    """The claim's aliases that the page text or the URL path names. The club rows are not the entity.

    Motivation vs Logic
    Motivation: in a league table every row is a club, so no cell can name the competition.
    Logic: the alias is looked for in the page text (case-folded) and in the URL path (letters and digits only).
    """
    text = " ".join((page_text or "").split()).casefold()
    path = normalise(urlsplit(final_url or "").path)
    found = []
    for alias in claim.get("aliases") or []:
        if not alias:
            continue
        if alias.casefold() in text or (normalise(alias) and normalise(alias) in path):
            found.append(alias)
    return found


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
        if _COMMENT.match(line) and not _RANK_HEADER.match(line):
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
