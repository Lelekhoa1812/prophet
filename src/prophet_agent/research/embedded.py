"""Tables embedded in HTML as JSON: Next.js __NEXT_DATA__, JSON-LD, window state assignments and inline arrays of records.
No network, and no script is run.

Motivation vs Logic
Motivation: many league tables, results pages and dataset pages ship their exact numbers inside the page's own JSON. The
markup grid is a rendering of that JSON, so the JSON is the better candidate when it is there.
Logic: each script is read as text and never executed. JSON roots are parsed with the standard library, bounded by script
size, the number of scripts and decode attempts. Any list of at least MIN_RECORDS objects whose shared columns include a
numeric one becomes a candidate, with its JSON path as the locator. Nested objects are flattened one level with dotted
names. Rows and columns are capped, and a table over MAX_ROWS is marked truncated rather than dropped.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import re

from bs4 import BeautifulSoup

MAX_CANDIDATES = 8
MAX_SCRIPTS = 200
MAX_SCRIPT_CHARS = 1_000_000
MAX_ATTEMPTS = 100          # raw decode attempts per script, so a script full of brackets cannot stall the read
MAX_ROWS = 2000
MAX_COLS = 32
MAX_DEPTH = 12
MAX_NODES = 20_000
MIN_RECORDS = 3
SHARED = 0.8                # a column must be present in at least this share of the records
_NUMBER = re.compile(r"^[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?%?$")
_IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_ASSIGN = re.compile(r"\b(?:window\.)?(__[A-Za-z0-9_$]+__)\s*=\s*(?=[\[{])")
_ARRAY_START = re.compile(r"\[\s*\{")
_DECODER = json.JSONDecoder()


def embedded_tables(html: str, *, base_url: str = "") -> dict:
    """Candidates from the JSON embedded in ``html``. Each has the same keys as an html_tables candidate, plus method
    (next_data, json_ld, script_json, inline_state or inline_json) and a locator of the form source:$.json.path."""
    if not html or not str(html).strip():
        return {"candidates": []}
    soup = BeautifulSoup(str(html), "lxml")
    found: list[tuple[str, str, dict]] = []
    for index, script in enumerate(soup.find_all("script")):
        if index >= MAX_SCRIPTS:
            break
        text = script.string or script.get_text() or ""
        if not text.strip() or len(text) > MAX_SCRIPT_CHARS:
            continue
        for method, source, root in _roots(script, text):
            for path, records in _record_lists(root):
                found.append((method, f"{source}:{path}", _table(records)))
    candidates: list[dict] = []
    seen: set[str] = set()
    for method, locator, table in found:
        if len(candidates) >= MAX_CANDIDATES:
            break
        if table["sha256"] in seen:
            continue
        seen.add(table["sha256"])
        candidates.append({"index": len(candidates), "method": method, "locator": locator,
                           "columns": table["columns"], "row_count": table["row_count"], "sha256": table["sha256"],
                           "csv": table["csv"], "preview_rows": table["rows"][:5], "final_url": base_url,
                           "truncated": table["truncated"]})
    return {"candidates": candidates}


def _roots(script, text: str) -> list[tuple[str, str, object]]:
    """The JSON values a script holds: whole-script JSON for typed scripts, and assigned or inline values for plain ones."""
    kind = (script.get("type") or "").strip().lower()
    script_id = script.get("id") or ""
    if script_id == "__NEXT_DATA__":
        return _whole(text, "next_data", "__NEXT_DATA__")
    if kind == "application/ld+json":
        return _whole(text, "json_ld", "ld+json")
    if kind == "application/json":
        return _whole(text, "script_json", f"script#{script_id}" if script_id else "script[application/json]")
    if kind not in ("", "text/javascript", "application/javascript"):
        return []
    return _inline(text)


def _whole(text: str, method: str, source: str) -> list[tuple[str, str, object]]:
    try:
        return [(method, source, json.loads(text))]
    except ValueError:
        return []


def _inline(text: str) -> list[tuple[str, str, object]]:
    """Values assigned to a name such as window.__INITIAL_STATE__ = {...}, and inline arrays of objects. A function call
    (the usual form of __NUXT__) is not JSON and is skipped."""
    roots: list[tuple[str, str, object]] = []
    for match in _ASSIGN.finditer(text):
        try:
            value, _ = _DECODER.raw_decode(text, match.end())
        except ValueError:
            continue
        roots.append(("inline_state", match.group(1), value))
    position, attempts = 0, 0
    while attempts < MAX_ATTEMPTS:
        match = _ARRAY_START.search(text, position)
        if match is None:
            break
        attempts += 1
        try:
            value, end = _DECODER.raw_decode(text, match.start())
        except ValueError:
            position = match.end()
            continue
        roots.append(("inline_json", "script", value))
        position = end
    return roots


def _record_lists(root, path: str = "$", depth: int = 0, budget: list | None = None) -> list[tuple[str, list]]:
    """Every list of objects under root that is at least MIN_RECORDS long, with its JSON path. A list that is not a table
    is searched inside, so a list of wrapper objects still yields the table within each."""
    budget = budget if budget is not None else [MAX_NODES]
    if depth > MAX_DEPTH or budget[0] <= 0:
        return []
    budget[0] -= 1
    out: list[tuple[str, list]] = []
    if isinstance(root, list):
        if len(root) >= MIN_RECORDS and all(isinstance(item, dict) for item in root) and _table(root) is not None:
            return [(path, root)]
        for index, item in enumerate(root):
            out.extend(_record_lists(item, f"{path}[{index}]", depth + 1, budget))
    elif isinstance(root, dict):
        for key, value in root.items():
            out.extend(_record_lists(value, f"{path}{_segment(str(key))}", depth + 1, budget))
    return out


def _segment(key: str) -> str:
    if _IDENT.match(key):
        return f".{key}"
    return '["' + key.replace("\\", "\\\\").replace('"', '\\"') + '"]'


def _table(records: list) -> dict | None:
    """The table a list of objects makes: shared scalar columns, at least two, one of them numeric. None otherwise.

    Motivation vs Logic: a list of records is a table only when its records agree on their fields and carry numbers. A
    menu or a list of links has no numbers and is not returned."""
    flat = [_flatten(record) for record in records if isinstance(record, dict)]
    count = len(flat)
    if count < MIN_RECORDS:
        return None
    presence: dict[str, int] = {}
    for record in flat:
        for key in record:
            presence[key] = presence.get(key, 0) + 1
    columns = [key for key, seen in presence.items() if seen >= SHARED * count][:MAX_COLS]
    if len(columns) < 2 or not any(_numeric_column(flat, column) for column in columns):
        return None
    rows = [[_cell(record.get(column)) for column in columns] for record in flat[:MAX_ROWS]]
    buf = io.StringIO()
    writer = csv.writer(buf, lineterminator="\n")
    writer.writerow(columns)
    writer.writerows(rows)
    text = buf.getvalue()
    return {"columns": columns, "rows": rows, "row_count": len(rows), "csv": text, "truncated": count > MAX_ROWS,
            "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest()}


def _flatten(record: dict, prefix: str = "", depth: int = 0) -> dict:
    out: dict = {}
    for key, value in record.items():
        name = f"{prefix}{key}"
        if isinstance(value, (bool, int, float)) or (isinstance(value, str) and value.strip()):
            out[name] = value
        elif isinstance(value, dict) and depth < 1:
            out.update(_flatten(value, f"{name}.", depth + 1))
    return out


def _numeric_column(flat: list[dict], column: str) -> bool:
    values = [record[column] for record in flat if column in record]
    if not values:
        return False
    numeric = 0
    for value in values:
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)):
            numeric += 1
        elif isinstance(value, str) and _NUMBER.match(value.strip().replace(" ", "")):
            numeric += 1
    return numeric * 5 >= len(values) * 4


def _cell(value) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return " ".join(value.split())
    return str(value)
