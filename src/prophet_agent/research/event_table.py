"""Dated events from news and pages as one event table: typed, deduplicated, every source URL kept (M9-4).

Motivation vs Logic
Motivation: the same fixture or results day appears in three articles with three URLs. A forecast that counts it three times
misreads the record, and a planner that cites one URL when three agree hides the evidence. The table also has to say how each
date was found: a year written in the sentence is stronger than a year taken from a publication date, which is stronger than a
year assumed from the current calendar.
Logic: each input row is checked (an ISO date, text, an http(s) source URL, an optional end date and type) and refused with a
reason when it fails. The type comes from keyword rules when none is given. The date's origin and certainty come from events.py
(how_sure, date_found); a year assumed from the current calendar is always "low", and a publication-date year is at most
"medium". Two rows with the same date, the same entity (or none) and overlapping wording are one event: the strongest certainty
and date origin are kept, the first type that is not "other" wins, and every source URL is listed. Everything here is a source
claim, not a measurement; the table carries no value.
"""
from __future__ import annotations

import csv
import io
import re
from typing import Any

from prophet_agent.research.events import iso_day
from prophet_agent.research.passages import DATE_EXPR

EVENT_TYPES = ("earnings", "product", "policy", "fixture", "holiday", "outage", "weather", "other")
DATE_ORIGINS = ("stated in text", "publication date", "inferred")
CERTAINTY = ("high", "medium", "low")
ORIGIN_RANK = {"stated in text": 3, "publication date": 2, "inferred": 1}
CERTAINTY_RANK = {"high": 3, "medium": 2, "low": 1}
MAX_TEXT = 240
COLUMNS = ["date", "end_date", "event", "type", "entity", "date_found", "certainty", "source_count", "source_urls"]
URL = re.compile(r"https?://\S+")
YEAR = re.compile(r"\b(?:19|20)\d{2}\b")

# Ordered: the first rule that matches decides the type. Holidays and outages come first so that "market closed" is not
# read as a rate decision, and earnings come before fixtures so that "final results" is not a fixture.
_TYPE_RULES = [
    ("holiday", re.compile(r"\b(?:public holidays?|bank holidays?|holidays?|market closed|closed for)\b", re.I)),
    ("outage", re.compile(r"\b(?:outages?|disruptions?|shutdowns?|shut down|recalls?|strikes?|cyber ?attacks?|blackouts?)\b", re.I)),
    ("weather", re.compile(r"\b(?:cyclones?|hurricanes?|storms?|floods?|flooding|heat ?waves?|bushfires?|droughts?|frosts?|"
                           r"snowfall|typhoons?|weather)\b", re.I)),
    ("policy", re.compile(r"\b(?:rate decisions?|cash rate|interest rates?|policy meetings?|policy decisions?|central banks?|"
                          r"fomc|rba|ecb|opec\+?|tariffs?|regulators?|regulations?|legislation|sanctions?)\b", re.I)),
    ("earnings", re.compile(r"\b(?:earnings|results|quarterly|quarter|q[1-4]|half-year|full-year|fiscal|profits?|revenues?|"
                            r"guidance|dividends?|annual report)\b", re.I)),
    ("fixture", re.compile(r"\b(?:fixtures?|matches|match|kick-?off|round \d+|derby|finals?|semi-?finals?|vs\.?|versus|"
                           r"grand prix|games?|tournament)\b", re.I)),
    ("product", re.compile(r"\b(?:launch(?:es|ed)?|unveils?|unveiled|keynote|new models?|products?|releases?|released)\b", re.I)),
]
_STOP = {"the", "and", "for", "with", "from", "that", "this", "will", "its", "are", "was", "has", "have", "on", "at", "of",
         "to", "in", "by", "be", "is", "as", "it", "their", "his", "her", "new", "says", "said", "also", "after", "before",
         "over", "into", "per", "cent"}
_MONTHS = {"january", "february", "march", "april", "may", "june", "july", "august", "september", "october", "november",
           "december", "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "sept", "oct", "nov", "dec"}


def classify_type(text: str) -> str:
    """The event type from keywords in the sentence; "other" when no rule matches."""
    for kind, rule in _TYPE_RULES:
        if rule.search(text or ""):
            return kind
    return "other"


def date_origin(text: str) -> str:
    """How the date in the sentence was found, for a row that did not come from events.py: "stated in text" when the date
    expression carries a year, else "inferred" (the year is not written, so it cannot be checked from the text)."""
    match = DATE_EXPR.search(text or "")
    return "stated in text" if match and YEAR.search(match.group(0)) else "inferred"


def _content(text: str) -> set[str]:
    """Content words of a sentence, crudely stemmed, so that "reports" and "reported" match and dates or stop words do not."""
    out: set[str] = set()
    for word in re.findall(r"[a-z][a-z0-9+]*", (text or "").lower()):
        if len(word) < 3 or word in _STOP or word in _MONTHS:
            continue
        for suffix in ("ing", "ed", "es", "s"):
            if word.endswith(suffix) and len(word) - len(suffix) >= 4:
                word = word[: -len(suffix)]
                break
        out.add(word)
    return out


def _same_event(a: set[str], b: set[str]) -> bool:
    """Two sentences are one event when the smaller word set is inside the other, or when they share at least four fifths
    of it with at least three words ("Arsenal v Chelsea" matches "Chelsea vs Arsenal"; "Apple earnings due Thursday" does not
    match "Microsoft earnings due Thursday", which shares three of four)."""
    if not a or not b:
        return False
    small = min(len(a), len(b))
    shared = len(a & b)
    if small >= 2 and shared == small:
        return True
    return small >= 3 and shared / small >= 0.8


def _row(item: Any, entity: str | None) -> dict:
    """One normalised row; raises ValueError with the reason when the input cannot be used."""
    if not isinstance(item, dict):
        raise ValueError("not an object")
    if item.get("date") is None:
        raise ValueError("date is missing")
    day = iso_day(str(item["date"]), "date")
    text = " ".join(str(item.get("event") or "").split())[:MAX_TEXT]
    if not text:
        raise ValueError("event text is missing")
    url = item.get("source_url")
    if not isinstance(url, str) or not URL.fullmatch(url.strip()):
        raise ValueError("source_url is missing or is not an http(s) address; an event is kept only with its source")
    end = item.get("end_date", item.get("end"))
    end = None if end in (None, "") else iso_day(str(end), "end_date")
    if end is not None and end < day:
        raise ValueError("end_date is before date")
    kind = item.get("type")
    if kind in (None, ""):
        kind = classify_type(text)
    elif kind not in EVENT_TYPES:
        raise ValueError(f"type must be one of {', '.join(EVENT_TYPES)}")
    origin = item.get("date_found")
    if origin not in DATE_ORIGINS:
        origin = date_origin(text)
    certainty = item.get("how_sure", item.get("certainty"))
    if certainty not in CERTAINTY:
        certainty = "low"
    if origin == "inferred":
        certainty = "low"
    elif origin == "publication date" and certainty == "high":
        certainty = "medium"
    row_entity = item.get("entity") or entity
    return {"date": day, "end_date": end, "event": text, "type": kind, "entity": str(row_entity) if row_entity else None,
            "date_found": origin, "certainty": certainty, "source_urls": [url.strip()]}


def dedupe(rows: list[dict]) -> list[dict]:
    """Merge rows that describe the same event (same date, compatible entity, overlapping wording). The strongest date origin
    and certainty are kept, the first type that is not "other" wins, the first wording is the text, and every source URL is
    listed once, in the order seen. Output is sorted by date, then text."""
    ordered = sorted(rows, key=lambda r: (r["date"], -CERTAINTY_RANK[r["certainty"]], -ORIGIN_RANK[r["date_found"]]))
    groups: list[dict] = []
    for row in ordered:
        words = _content(row["event"])
        home = None
        for g in groups:
            if g["date"] != row["date"]:
                continue
            if g["entity"] and row["entity"] and g["entity"].lower() != row["entity"].lower():
                continue
            if _same_event(g["words"], words):
                home = g
                break
        if home is None:
            groups.append({"date": row["date"], "event": row["event"], "words": words, "type": row["type"],
                           "entity": row["entity"], "end_date": row["end_date"], "date_found": row["date_found"],
                           "certainty": row["certainty"], "source_urls": list(row["source_urls"])})
            continue
        if home["type"] == "other" and row["type"] != "other":
            home["type"] = row["type"]
        home["entity"] = home["entity"] or row["entity"]
        if row["end_date"] and (home["end_date"] is None or row["end_date"] > home["end_date"]):
            home["end_date"] = row["end_date"]
        if ORIGIN_RANK[row["date_found"]] > ORIGIN_RANK[home["date_found"]]:
            home["date_found"] = row["date_found"]
        if CERTAINTY_RANK[row["certainty"]] > CERTAINTY_RANK[home["certainty"]]:
            home["certainty"] = row["certainty"]
        for url in row["source_urls"]:
            if url not in home["source_urls"]:
                home["source_urls"].append(url)
    out = []
    for g in groups:
        out.append({"date": g["date"], "end_date": g["end_date"], "event": g["event"], "type": g["type"],
                    "entity": g["entity"], "date_found": g["date_found"], "certainty": g["certainty"],
                    "source_count": len(g["source_urls"]), "source_urls": g["source_urls"]})
    return sorted(out, key=lambda r: (r["date"], r["event"]))


def normalise_events(raw: Any, *, entity: str | None = None) -> tuple[list[dict], list[dict]]:
    """The event table rows from a list of event objects (the research call's ``events`` list, or hand-made rows with
    date, event, source_url and optionally end_date, type, entity, how_sure or certainty, date_found). Returns (rows, refused)
    where refused lists {index, reason} for each input that could not be used. A row is never invented from a missing field."""
    if not isinstance(raw, list):
        raise ValueError("dated_events must be a list of event objects")
    rows, refused = [], []
    for i, item in enumerate(raw):
        try:
            rows.append(_row(item, entity))
        except ValueError as exc:
            refused.append({"index": i, "reason": str(exc)})
    return dedupe(rows), refused


def select_events(rows: list[dict], min_certainty: str = "medium") -> list[dict]:
    """The rows at or above a certainty ("high", "medium" or "low"). Rows below it stay out of the forecast, not the table."""
    if min_certainty not in CERTAINTY:
        raise ValueError(f"min_certainty must be one of {', '.join(CERTAINTY)}")
    floor = CERTAINTY_RANK[min_certainty]
    return [r for r in rows if CERTAINTY_RANK[r["certainty"]] >= floor]


def event_table_csv(rows: list[dict]) -> bytes:
    """The table as a CSV dataset (UTF-8, one row per event, every source URL in source_urls separated by " | ")."""
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=COLUMNS, lineterminator="\n")
    writer.writeheader()
    for r in rows:
        writer.writerow({"date": r["date"], "end_date": r["end_date"] or "", "event": r["event"], "type": r["type"],
                         "entity": r["entity"] or "", "date_found": r["date_found"], "certainty": r["certainty"],
                         "source_count": r["source_count"], "source_urls": " | ".join(r["source_urls"])})
    return buf.getvalue().encode("utf-8")
