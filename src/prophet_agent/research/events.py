"""A dated event list from the passages of a research call, placed against the data's end and forecast horizon (A3-08).

Motivation vs Logic
Motivation: research that is not tied to the data decorates the answer. A dated event (an OPEC+ meeting on 29 November,
a statistics release on 10 November) only matters if the planner can see when it falls relative to the data and the
forecast window, so that it can be matched with a detected level shift or the horizon.
Logic: every sentence of a passage that carries a calendar date becomes one event, with the date read from that
sentence (the year comes from the sentence, else the source's date, else the current year). how_sure is a rule, not a
judgement: "high" for an official source (tier 1) whose year is written in the text; "medium" for a tier 1 or tier 2
source with a less certain date; "low" otherwise. position is set only when the data's last date is given. Events
are for the planner to match, and are reported as "consistent with", never as cause.
"""
from __future__ import annotations

import re
from datetime import date as _date
from datetime import datetime, timezone

from prophet_agent.research.passages import DATE_EXPR, has_date, parse_date_expr

MAX_EVENTS = 30
EVENT_CHARS = 240
ISO_DAY = re.compile(r"\d{4}-\d{2}-\d{2}")
SENTENCE = re.compile(r"(?<=[.!?])\s+|\n+")


def iso_day(value: str | None, name: str) -> str | None:
    """A YYYY-MM-DD date or None; anything else is refused with the argument's name."""
    if value is None:
        return None
    if not isinstance(value, str) or not ISO_DAY.fullmatch(value.strip()):
        raise ValueError(f"{name} must be an ISO date (YYYY-MM-DD) or left out")
    try:
        _date.fromisoformat(value.strip())
    except ValueError:
        raise ValueError(f"{name} must be an ISO date (YYYY-MM-DD) or left out") from None
    return value.strip()


def _source_year(source: dict) -> int:
    date = source.get("date") or ""
    match = re.match(r"(\d{4})", str(date))
    return int(match.group(1)) if match else datetime.now(timezone.utc).year


def _how_sure(tier: int, explicit_year: bool) -> str:
    if tier == 1 and explicit_year:
        return "high"
    if (tier <= 2 and explicit_year) or tier == 1:
        return "medium"
    return "low"


def _date_found(explicit: bool, source: dict) -> str:
    """How the date was found: the year is written in the sentence ("stated in text"), else taken from the source's
    publication date, else assumed from the current year ("inferred")."""
    if explicit:
        return "stated in text"
    return "publication date" if source.get("date") else "inferred"


def _position(event_date: str, data_end: str | None, horizon_end: str | None) -> str | None:
    if data_end is None:
        return None
    if event_date <= data_end:
        return "inside the data period"
    if horizon_end is None or event_date <= horizon_end:
        return "inside the forecast window"
    return "after the forecast window"


def dated_events(sources: list[dict], data_end: str | None = None, horizon_end: str | None = None) -> list[dict]:
    """Events from the passages of the sources, earliest first, each source's sentence once. Each event has date (ISO),
    event (the sentence, clipped), source_id, source_url, how_sure and position."""
    found: list[dict] = []
    seen: set[tuple[str, str]] = set()
    for source in sources:
        tier = int(source.get("tier") or 4)
        for passage in source.get("passages") or []:
            for sentence in SENTENCE.split(passage.get("text") or ""):
                sentence = " ".join(sentence.split())
                if not has_date(sentence):
                    continue
                match = DATE_EXPR.search(sentence)
                parsed = parse_date_expr(match.group(0), _source_year(source)) if match else None
                if parsed is None:
                    continue
                event_date, explicit = parsed
                key = (event_date, sentence[:80].lower())
                if key in seen:
                    continue
                seen.add(key)
                found.append({"date": event_date, "event": sentence[:EVENT_CHARS],
                              "source_id": source.get("id"), "source_url": source.get("final_url") or source.get("url"),
                              "how_sure": _how_sure(tier, explicit),
                              "date_found": _date_found(explicit, source),
                              "position": _position(event_date, data_end, horizon_end)})
    found.sort(key=lambda e: e["date"])
    return found[:MAX_EVENTS]
