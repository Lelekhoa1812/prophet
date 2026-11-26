"""How many rows a table should have. Reads the claim, the page text and results tables. No network.

Motivation vs Logic
Motivation: a table with 11 rows of a 20-club league is a short table, and the only way to know is to say how many
rows to expect. The count often sits in the claim, in a list of clubs, in the page ("20 clubs"), or in the teams that a
results table names. Each of those is a source that can be wrong on its own.
Logic: the claim wins when it states a count or lists entities. For a league claim only, the page text and the results
tables are read. Sources that agree give one count. Sources that disagree give no count and a conflict flag, so a
short table is never called complete on a guess. A page count that follows "top", "first" or "relegated" is a
position, not the size of the league, and is ignored.
"""
from __future__ import annotations

import csv
import io
import re

from prophet_agent.source_table.choose import normalise

MIN_COUNT = 2
MAX_COUNT = 40
_PAGE_COUNT = re.compile(r"(?<![\w.])(\d{1,2})\s+(?:clubs|teams|sides)\b", re.IGNORECASE)
_POSITION_BEFORE = re.compile(r"\b(?:top|first|last|bottom|best|worst|relegated|promoted|qualifying|qualify|other|next)\s*$",
                              re.IGNORECASE)
_HOME = frozenset({"hometeam", "home", "homeclub", "hometeamname"})
_AWAY = frozenset({"awayteam", "away", "awayclub", "awayteamname"})


def expected_count(claim: dict, *, page_text: str | None = None, reference_csvs: list[bytes] | tuple = ()) -> dict:
    """The expected row count and where it came from. ``expected_rows`` is None when no safe count is known.

    Motivation vs Logic
    Motivation: a count that the sources disagree on is worse than none, because it would refuse a correct table.
    Logic: claim, then the claim's entity list, then (league only) page text and results tables. One distinct value
    is used. Two or more distinct values give None and expected_conflict True.
    """
    stated = claim.get("expected_rows")
    if isinstance(stated, int) and not isinstance(stated, bool) and stated > 0:
        return _found(stated, "claim", "the claim states the row count")
    listed = claim.get("expected_entities") or []
    if listed:
        return _found(len(listed), "claim_entities", "the claim lists the entities")
    if claim.get("table_kind") != "league":
        return _found(None, "", "")
    found: list[tuple[int, str, str]] = []
    if page_text:
        counts = _page_counts(page_text)
        if len(counts) == 1:
            found.append((counts.pop(), "page_text", "the page text states the number of clubs or teams"))
        elif len(counts) > 1:
            return _found(None, "", "the page text states more than one number of clubs or teams", conflict=True)
    for body in reference_csvs or ():
        teams = _results_teams(body)
        if teams:
            found.append((teams, "results_table", "distinct teams in a results table"))
    values = {count for count, _, _ in found}
    if not found:
        return _found(None, "", "")
    if len(values) > 1:
        return _found(None, "", "sources disagree on the number of rows: " + ", ".join(
            f"{source} {count}" for count, source, _ in found), conflict=True)
    count, source, evidence = found[0]
    return _found(count, source, evidence)


def _found(count, source: str, evidence: str, *, conflict: bool = False) -> dict:
    return {"expected_rows": count, "expected_source": source, "expected_evidence": evidence,
            "expected_conflict": conflict}


def _page_counts(page_text: str) -> set[int]:
    """Numbers stated as a club or team count in the page text, excluding positions such as "top 4 teams"."""
    text = " ".join(page_text.split())
    counts: set[int] = set()
    for match in _PAGE_COUNT.finditer(text):
        before = text[max(0, match.start() - 24):match.start()]
        if _POSITION_BEFORE.search(before):
            continue
        value = int(match.group(1))
        if MIN_COUNT <= value <= MAX_COUNT:
            counts.add(value)
    return counts


def _results_teams(body: bytes) -> int:
    """Distinct team names in the home and away columns of a results table. Zero when the table is not one."""
    if not body:
        return 0
    try:
        text = body.decode("utf-8-sig")
    except UnicodeDecodeError:
        return 0
    lines = [line for line in text.splitlines() if line.strip() and not line.lstrip().startswith("#")]
    try:
        rows = list(csv.reader(io.StringIO("\n".join(lines))))
    except csv.Error:
        return 0
    if len(rows) < 2:
        return 0
    headers = [normalise(cell) for cell in rows[0]]
    home = [i for i, name in enumerate(headers) if name in _HOME]
    away = [i for i, name in enumerate(headers) if name in _AWAY]
    if not home or not away:
        return 0
    teams: set[str] = set()
    for row in rows[1:]:
        for index in home + away:
            if index < len(row) and normalise(row[index]):
                teams.add(normalise(row[index]))
    return len(teams) if len(teams) >= MIN_COUNT else 0
