"""Dated figures from the text of a page: the model proposes, code verifies every row against the page.

Motivation vs Logic
Motivation: many series exist only as sentences ("the unemployment rate in the third quarter was 2.22 percent"), in press
releases, statistical reports and news. In the VN Unemployment notebook (2026-11-27) the analyst read such passages and
retyped numbers into a cell; the provenance guard refused the cell, and the run stalled. Text is a legitimate source of
figures, but only when every figure can be traced to its sentence.
Logic: (1) the page text is split into sentences, and the ones holding a number and a word of the measure are numbered and
shown to the model; (2) the model proposes rows of period, value, unit, entity, the sentence number and the exact words that
state the figure (a quote), in a strict schema; (3) code keeps a row only when the quote is in that sentence word for word,
a number in the quote equals the value, and the period parses to a date; (4) rows that state different values for the same
period, unit and entity are conflicts: none of them is kept, and they are listed. The model's reading is never trusted on its
own: what is kept is what the page says. The proposal is kept in the result, so a resumed run replays it.
"""
from __future__ import annotations

import calendar
import json
import re
from datetime import date
from typing import Any

from prophet_agent.nim.schema import validate

TEXT_MODELS: tuple[str, ...] = ()      # empty: the provider's default chain
MAX_SENTENCES = 80
MAX_SENTENCE_CHARS = 400
MAX_ROWS = 120
NUMBER = re.compile(r"(?<![\w.])[-+−]?\d{1,3}(?:[,\s]\d{3})+(?:\.\d+)?(?![\w])|(?<![\w.])[-+−]?\d+(?:\.\d+)?(?![\w])")
SENTENCE_END = re.compile(r"(?<=[.!?;])\s+(?=[A-Z0-9(\"'])|\n+")
MONTHS = {name.lower(): i for i, name in enumerate(calendar.month_name) if name}
MONTHS.update({name.lower(): i for i, name in enumerate(calendar.month_abbr) if name})
ORDINAL_Q = {"first": 1, "second": 2, "third": 3, "fourth": 4, "1st": 1, "2nd": 2, "3rd": 3, "4th": 4}

PROPOSAL_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["rows"],
    "properties": {
        "rows": {"type": "array", "maxItems": MAX_ROWS, "items": {
            "type": "object", "additionalProperties": False,
            "required": ["sentence", "quote", "period", "value", "unit"],
            "properties": {
                "sentence": {"type": "integer", "minimum": 0},
                "quote": {"type": "string", "maxLength": 300},
                "period": {"type": "string", "maxLength": 40},
                "value": {"type": "number"},
                "unit": {"type": "string", "maxLength": 40},
                "entity": {"type": "string", "maxLength": 80},
            }}},
    },
}

SYSTEM_PROMPT = (
    "You read numbered sentences from one page and list every figure that states the requested measure for a dated period. "
    "For each figure give: sentence (its number), quote (the exact words from that sentence that state the figure, copied "
    "character for character, including the number), period (the period the figure is for, written as YYYY, YYYY-MM, "
    "YYYY-Qn or YYYY-MM-DD; a period named only relative to an unstated date is left out), value (the number as written, "
    "without thousands separators; a percentage stays in percent), unit (percent, persons, thousand persons, index, "
    "USD million, and so on) and entity (the country, region or group, when the sentence names it). A forecast, target or "
    "projection is not a figure for that period: leave it out. Do not compute, convert or infer values. Reply with JSON only."
)


class TextFigureError(ValueError):
    """The page or the model's reply cannot give rows."""


def sentences(text: str, measure: str) -> list[str]:
    """The sentences of the text that hold a number and a word of the measure, in order, numbered by position in the list."""
    words = [w.lower() for w in re.findall(r"[A-Za-z]{4,}", measure or "")]
    out = []
    for raw in SENTENCE_END.split(text or ""):
        s = " ".join(raw.split())
        if not s or not NUMBER.search(s):
            continue
        low = s.lower()
        if words and not any(w[:6] in low for w in words):
            continue
        out.append(s[:MAX_SENTENCE_CHARS])
        if len(out) >= MAX_SENTENCES:
            break
    return out


def _numbers(text: str) -> list[float]:
    vals = []
    for m in NUMBER.finditer(text):
        token = m.group(0).replace("−", "-").replace(",", "").replace(" ", "")
        try:
            vals.append(float(token))
        except ValueError:
            continue
    return vals


def parse_period(text: str) -> tuple[str, str] | None:
    """(period label, ISO start date) for YYYY, YYYY-MM, YYYY-Qn, YYYY-MM-DD, 'Q3 2024', 'third quarter of 2024',
    'March 2025' and '2025 March'. None when the text names no single dated period."""
    t = " ".join(str(text or "").strip().split()).replace("–", "-")
    low = t.lower()
    if m := re.fullmatch(r"(\d{4})", t):
        return t, f"{t}-01-01"
    if m := re.fullmatch(r"(\d{4})-(\d{2})-(\d{2})", t):
        try:
            return t, date(int(m[1]), int(m[2]), int(m[3])).isoformat()
        except ValueError:
            return None
    if m := re.fullmatch(r"(\d{4})-?(\d{1,2})", t):
        if 1 <= int(m[2]) <= 12:
            return f"{m[1]}-{int(m[2]):02d}", f"{m[1]}-{int(m[2]):02d}-01"
        return None
    q = (re.fullmatch(r"(\d{4})\s*-?\s*q([1-4])", low) or re.fullmatch(r"q([1-4])\s*(\d{4})", low))
    if q:
        year, n = (q[1], q[2]) if len(q[1]) == 4 else (q[2], q[1])
        return f"{year}-Q{n}", f"{year}-{3 * (int(n) - 1) + 1:02d}-01"
    if m := re.fullmatch(r"(first|second|third|fourth|1st|2nd|3rd|4th) quarter(?: of)? (\d{4})", low):
        n = ORDINAL_Q[m[1]]
        return f"{m[2]}-Q{n}", f"{m[2]}-{3 * (n - 1) + 1:02d}-01"
    if m := (re.fullmatch(r"([a-z]{3,9})\.? (\d{4})", low) or re.fullmatch(r"(\d{4}) ([a-z]{3,9})", low)):
        name, year = (m[1], m[2]) if m[1].isalpha() else (m[2], m[1])
        if name in MONTHS:
            return f"{year}-{MONTHS[name]:02d}", f"{year}-{MONTHS[name]:02d}-01"
    return None


def ask_model(client: Any, numbered: list[str], measure: str, entity: str | None, models=TEXT_MODELS) -> tuple[dict, str]:
    lines = "\n".join(f"[{i}] {s}" for i, s in enumerate(numbered))
    ask = f"Measure: {measure}" + (f"\nEntity: {entity}" if entity else "") + f"\n\nSentences:\n{lines}"
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": ask}]
    chain = list(models) or list(getattr(client, "chain", None) or [])
    if not chain:
        from prophet_agent.nim.transport import DEFAULT_CHAIN
        chain = list(DEFAULT_CHAIN)
    reply = client.complete(chain, messages, schema=PROPOSAL_SCHEMA, schema_name="text_figures", max_tokens=4096,
                            thinking=False)
    try:
        proposal = json.loads(reply.text)
    except (ValueError, TypeError):
        raise TextFigureError("the model did not return JSON") from None
    problem = validate(proposal, PROPOSAL_SCHEMA)
    if problem:
        raise TextFigureError(f"the model's reply failed the schema: {problem}")
    return proposal, str(reply.model)


def verify(proposal: dict, numbered: list[str]) -> tuple[list[dict], list[dict]]:
    """(kept rows, rejected rows with a reason). A row is kept only when its quote is in its sentence word for word, a
    number in the quote equals its value, and its period parses."""
    kept, rejected = [], []
    for row in proposal.get("rows") or []:
        idx = row.get("sentence")
        if not isinstance(idx, int) or not 0 <= idx < len(numbered):
            rejected.append({**row, "reason": "no such sentence"})
            continue
        quote = " ".join(str(row.get("quote", "")).split())
        if not quote or quote not in numbered[idx]:
            rejected.append({**row, "reason": "the quote is not in the sentence word for word"})
            continue
        value = float(row["value"])
        if not any(abs(v - value) <= 1e-9 * max(1.0, abs(value)) for v in _numbers(quote)):
            rejected.append({**row, "reason": "the value is not a number written in the quote"})
            continue
        parsed = parse_period(row.get("period", ""))
        if parsed is None:
            rejected.append({**row, "reason": "the period is not one dated period"})
            continue
        label, start = parsed
        kept.append({"period": label, "date": start, "value": value, "unit": str(row.get("unit", "")).strip(),
                     "entity": str(row.get("entity", "") or "").strip(), "quote": quote, "sentence": idx})
    return _resolve_conflicts(kept, rejected)


def _resolve_conflicts(rows: list[dict], rejected: list[dict]) -> tuple[list[dict], list[dict]]:
    groups: dict[tuple, list[dict]] = {}
    for r in rows:
        groups.setdefault((r["period"], r["unit"].lower(), r["entity"].lower()), []).append(r)
    kept = []
    for key, items in groups.items():
        values = {round(r["value"], 9) for r in items}
        if len(values) == 1:
            kept.append(items[0])
        else:
            for r in items:
                rejected.append({**r, "reason": f"conflict: the page states {sorted(values)} for {key[0]}; none is kept"})
    kept.sort(key=lambda r: (r["entity"], r["date"]))
    return kept, rejected


def extract(text: str, measure: str, *, entity: str | None = None, client: Any = None, proposal: dict | None = None,
            model_name: str | None = None) -> dict:
    """Rows of dated figures for the measure, each traced to its sentence. ``proposal`` (from an earlier run, or a test)
    replaces the model call."""
    if not isinstance(measure, str) or not measure.strip():
        raise TextFigureError("measure must name what is measured, for example 'unemployment rate'")
    numbered = sentences(text, measure)
    if not numbered:
        raise TextFigureError(f"the page has no sentence with a number and the words of {measure!r}")
    if proposal is None:
        if client is None:
            from prophet_agent.nim.transport import shared_client
            client = shared_client()
        proposal, model_name = ask_model(client, numbered, measure, entity)
    kept, rejected = verify(proposal, numbered)
    return {"rows": kept, "rejected": rejected, "sentences_read": len(numbered), "proposal": proposal,
            "model": model_name}


def to_csv(rows: list[dict]) -> bytes:
    """The kept rows as CSV: period, date, value, unit, entity, quote."""
    import csv
    import io
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["period", "date", "value", "unit", "entity", "quote"])
    for r in rows:
        w.writerow([r["period"], r["date"], r["value"], r["unit"], r["entity"], r["quote"]])
    return buf.getvalue().encode("utf-8")
