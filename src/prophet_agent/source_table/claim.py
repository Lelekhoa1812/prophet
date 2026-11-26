"""A table claim: the series the question asked for, as data the later checks read back.

Motivation vs Logic
Motivation: the file name was doing the job of a decision. The decision has to be a record the next tool can re-read.
Logic: one claim names the series, the entity, the measure, and the header aliases. An empty entity is refused.
The claim id is the hash of that record, so the same claim is the same id.
"""
from __future__ import annotations

import hashlib
import json
import re

SERIES = re.compile(r"^[A-Za-z][A-Za-z0-9._-]{0,80}$")
DEFAULT_TIME = ("date", "time", "period")
DEFAULT_ENTITY = ("ticker", "symbol", "name", "entity", "code", "series")


def build_claim(raw: dict) -> dict:
    """The stored claim. Raises ValueError when the entity or the series name is missing."""
    if not isinstance(raw, dict):
        raise ValueError("the claim must be an object")
    entity = _text(raw.get("entity"))
    if not entity:
        raise ValueError("the claim needs an entity")
    series = _text(raw.get("series_name"))
    if not SERIES.match(series):
        raise ValueError("series_name must be a short name that starts with a letter")
    aliases = []
    for item in raw.get("aliases") or []:
        text = _text(item)
        if text and text not in aliases:
            aliases.append(text)
    if entity not in aliases:
        aliases.insert(0, entity)
    claim = {
        "series_name": series,
        "entity": entity,
        "aliases": aliases,
        "measure": _text(raw.get("measure")),
        "time_grain": _text(raw.get("time_grain")) or "period",
        "time_aliases": _names(raw.get("time_aliases"), DEFAULT_TIME),
        "entity_aliases": _names(raw.get("entity_aliases"), DEFAULT_ENTITY),
        "measure_aliases": _names(raw.get("measure_aliases"), (_text(raw.get("measure")),) if _text(raw.get("measure")) else ()),
        "tolerance": _tolerance(raw.get("tolerance")),
    }
    claim.update(_cardinality(raw))
    claim["claim_id"] = hashlib.sha256(json.dumps(claim, sort_keys=True).encode("utf-8")).hexdigest()
    return claim


def _cardinality(raw: dict) -> dict:
    """The optional expected-size fields. A key is stored only when it is given, so an older claim keeps its id.

    Motivation vs Logic
    Motivation: a league table of 11 clubs passed as the table of 20, because nothing said how many rows to expect.
    Logic: expected_rows is a positive whole number, expected_entities is a list of names (their count is the expected
    size when no count is given), and table_kind "league" turns on the league checks. Two counts that disagree are refused.
    """
    out: dict = {}
    rows = raw.get("expected_rows")
    if rows is not None and rows != "":
        if isinstance(rows, bool) or not isinstance(rows, int) or rows < 1:
            raise ValueError("expected_rows must be a whole number of at least 1")
        out["expected_rows"] = rows
    names: list[str] = []
    for item in raw.get("expected_entities") or []:
        text = _text(item)
        if text and text not in names:
            names.append(text)
    if names:
        out["expected_entities"] = names
    if "expected_rows" in out and names and out["expected_rows"] != len(names):
        raise ValueError("expected_rows and expected_entities disagree on the number of rows")
    kind = _text(raw.get("table_kind")).lower()
    if kind:
        if kind != "league":
            raise ValueError("table_kind must be league when it is given")
        out["table_kind"] = kind
    return out


def _text(value) -> str:
    return value.strip() if isinstance(value, str) else ""


def _names(value, default: tuple[str, ...]) -> list[str]:
    found = []
    if isinstance(value, (list, tuple)):
        for item in value:
            text = _text(item)
            if text and text not in found:
                found.append(text)
    if not found:
        for item in default:
            if item and item not in found:
                found.append(item)
    return found


def _tolerance(value) -> str:
    if value is None or value == "":
        return "exact"
    if isinstance(value, str) and value.strip().lower() == "exact":
        return "exact"
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ValueError("tolerance must be exact or a number") from None
    if number < 0:
        raise ValueError("tolerance must be exact or a number")
    return str(number)
