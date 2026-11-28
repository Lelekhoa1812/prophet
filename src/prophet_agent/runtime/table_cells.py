"""Column-aware check of an answer's tables against the tables the tools printed.

Motivation vs Logic
Motivation: live EPL (round 20261128-063203) wrote "| Nottingham Forest | 18 |" under Points; the standings the sandbox
printed say 9 (17th). The row check (grounding.entity_row_mismatches) judged a row only when some tool result had a name
field for the club, and then pooled every number on every line that named it (Bournemouth's pool ran from 1 to 2027), so
almost any figure passed, and the bag-of-numbers check found 18 elsewhere.
Logic: each markdown table in the answer gives (entity, column, stated value). The evidence tables are read by column from
the tool outputs: pandas fixed-width prints (a value belongs to the header label whose right edge it shares), pipe tables,
CSV text, lists of row lists under a "columns" list, and objects with a name field. Column labels are matched through
synonym groups (Pts and Points, GD and goal difference, Pl and played...) after dropping words such as "current"; a label
like "expected final pts" is its own group and matches only an identical label. Names are matched with the club key
(sports.club_key: "Nott'm Forest" is "Nottingham Forest"). A stated value is wrong only when the evidence has values for
that entity in that column and none of them equals it at the stated precision; with no such value it is not judged here.
Only strings that name an answer entity are parsed, so a long ledger stays cheap.
"""
from __future__ import annotations

import re
from typing import Any, Iterator

_SEP = re.compile(r"^\s*\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)*\|?\s*$")
_NUM = re.compile(r"^[+\-−]?\d[\d,]*(?:\.\d+)?%?$")
_DROP = re.compile(r"\b(?:current|observed|actual|now|today|so far|as of[^|]*)\b|\([^)]*\)", re.I)
GROUPS: dict[str, set[str]] = {
    "points": {"pts", "points", "pt", "point"},
    "played": {"pl", "played", "mp", "gp", "p", "matches", "games", "matches played", "games played"},
    "won": {"w", "won", "wins"},
    "drawn": {"d", "drawn", "draws"},
    "lost": {"l", "lost", "losses"},
    "goals_for": {"gf", "f", "for", "goals for", "goals_for", "goals scored"},
    "goals_against": {"ga", "a", "against", "goals against", "goals_against", "goals conceded"},
    "goal_diff": {"gd", "goal difference", "goal_difference", "goal diff", "diff", "+/-", "gd."},
    "position": {"pos", "position", "rank", "#", "place", "pos."},
}
_PROJECTED = re.compile(r"expected|median|projected|predicted|forecast|final|p10|p90|10-90|probab|title|relegation", re.I)
_BY_LABEL = {label: group for group, labels in GROUPS.items() for label in labels}
_NAME_FIELDS = {"team", "club", "name", "entity", "country", "region", "series", "squad", "home", "player"}


def _key(name: str) -> str:
    from prophet_agent.forecast.sports import club_key
    return club_key(re.sub(r"[*_`]", "", str(name)).strip())


_TEXT_ALIAS: tuple[re.Pattern[str], dict[str, str]] | None = None


def _norm_text(text: str) -> str:
    """Text with each known long club name replaced by its key ("Nottingham Forest" -> "nottm forest") wherever it appears
    in the line, not only as a whole cell. Aliases whose key ends with the alias itself ("forest") are left out, so a key
    is never replaced inside itself."""
    global _TEXT_ALIAS
    from prophet_agent.forecast import sports
    if _TEXT_ALIAS is None:
        table = {k: v for k, v in sports._CLUB_ALIASES.items() if not v.endswith(k)}
        rx = re.compile(r"(?<![a-z])(?:" + "|".join(re.escape(k) for k in sorted(table, key=len, reverse=True))
                        + r")(?![a-z])")
        _TEXT_ALIAS = (rx, table)
    rx, table = _TEXT_ALIAS
    low = re.sub(r"[.'’]", "", str(text).lower())
    low = re.sub(r"\b(?:fc|afc|cf|sc)\b", " ", low)
    return rx.sub(lambda m: table[m.group(0)], re.sub(r"[ \t]+", " ", low))


def group_of(label: str) -> str:
    """The synonym group of a column label, or the cleaned label itself."""
    raw = re.sub(r"\s+", " ", _DROP.sub(" ", str(label).lower())).strip(" .:*_`")
    return _BY_LABEL.get(raw, raw)


def _value(text: str) -> tuple[float, int] | None:
    t = str(text).strip().strip("*").replace("−", "-").replace(",", "").rstrip("%")
    if not t or not _NUM.match(t):
        return None
    try:
        v = float(t)
    except ValueError:
        return None
    return v, (len(t.split(".")[1]) if "." in t else 0)


def answer_cells(answer: str) -> list[dict]:
    """(entity, label, group, stated text, value, decimals) for each numeric cell of each markdown table in the answer."""
    lines = (answer or "").splitlines()
    out: list[dict] = []
    header: list[str] | None = None
    for i, line in enumerate(lines):
        if not line.lstrip().startswith("|"):
            header = None
            continue
        if _SEP.match(line):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if i + 1 < len(lines) and _SEP.match(lines[i + 1]):
            header = cells
            continue
        if header is None or len(cells) != len(header):
            continue
        at = next((j for j, c in enumerate(cells) if _value(c) is None and re.search(r"[A-Za-z]{2}", c)), None)
        if at is None:
            continue
        # A rank in a table of projections ranks the projection (round 20261128-201606, EPL: "Pos" by expected points was
        # compared with today's standings and every row but the leader's failed). It is not a printed figure to check.
        projected = any(_PROJECTED.search(h) for h in header)
        for j, c in enumerate(cells):
            if j == at or (v := _value(c)) is None:
                continue
            if projected and group_of(header[j]) == "position":
                continue
            out.append({"entity": cells[at].strip("* "), "key": _key(cells[at]), "label": header[j], "group": group_of(header[j]),
                        "stated": c.strip().strip("*"), "value": v[0], "decimals": v[1]})
    return out


def _strings_and_objects(obj: Any) -> Iterator[Any]:
    stack = [obj]
    while stack:
        x = stack.pop()
        if isinstance(x, dict):
            yield x
            stack.extend(x.values())
        elif isinstance(x, list):
            yield x
            stack.extend(x)
        elif isinstance(x, str):
            yield x


def _from_fixed_width(lines: list[str], i: int, keys: set[str], found: dict) -> None:
    """A pandas-style print: the header is the nearest line above with two or more known labels; each number in the row
    belongs to the header label whose right edge it shares (within one character)."""
    row = lines[i]
    for back in range(i - 1, max(-1, i - 80), -1):
        head = lines[back]
        labels = [(m.group(), m.end()) for m in re.finditer(r"\S+", head)]
        if sum(1 for lab, _ in labels if group_of(lab) in GROUPS) < 2:
            continue
        name = _row_entity(row, keys)
        if name is None:
            return
        for m in re.finditer(r"\S+", row):
            v = _value(m.group())
            if v is None:
                continue
            for lab, end in labels:
                if abs(end - m.end()) <= 1:
                    found.setdefault((name, group_of(lab)), []).append(v[0])
                    break
        return


def _row_entity(text: str, keys: set[str]) -> str | None:
    norm = _norm_text(text)
    hits = [k for k in keys if k and re.search(r"(?<![a-z])" + re.escape(k) + r"(?![a-z])", norm)]
    return max(hits, key=len) if hits else None


def _from_delimited(lines: list[str], i: int, sep: str, keys: set[str], found: dict) -> None:
    row = [c.strip() for c in lines[i].strip().strip("|").split(sep)] if sep == "|" else [c.strip() for c in lines[i].split(sep)]
    for back in range(i - 1, max(-1, i - 80), -1):
        if sep == "|" and _SEP.match(lines[back]):
            continue
        head = ([c.strip() for c in lines[back].strip().strip("|").split(sep)] if sep == "|"
                else [c.strip() for c in lines[back].split(sep)])
        if len(head) != len(row) or sum(1 for h in head if group_of(h) in GROUPS or h.lower() in _NAME_FIELDS) < 2:
            continue
        name = next((k for c in row if (k := _row_entity(c, keys)) and _value(c) is None), None)
        if name is None:
            return
        for h, c in zip(head, row):
            if (v := _value(c)) is not None:
                found.setdefault((name, group_of(h)), []).append(v[0])
        return


def evidence_cells(history: list, keys: set[str]) -> dict[tuple[str, str], list[float]]:
    """(entity key, column group) -> the values the tools printed for it, from every table shape this module reads."""
    from prophet_agent.runtime.grounding import _tool_objects
    found: dict[tuple[str, str], list[float]] = {}
    plain = {k for k in keys if k}
    for top in _tool_objects(history):
        for x in _strings_and_objects(top):
            if isinstance(x, str):
                low = x.lower().replace("'", "").replace("’", "")
                if not any(k in low or k.split(" ")[0] in low for k in plain) and \
                        not any(k in _norm_text(x) for k in plain):
                    continue
                lines = x.splitlines()
                for i, line in enumerate(lines):
                    if _row_entity(line, plain) is None:
                        continue
                    if line.count("|") >= 2:
                        _from_delimited(lines, i, "|", plain, found)
                    elif line.count(",") >= 2:
                        _from_delimited(lines, i, ",", plain, found)
                    else:
                        _from_fixed_width(lines, i, plain, found)
            elif isinstance(x, dict):
                names = [v for k, v in x.items() if str(k).lower() in _NAME_FIELDS and isinstance(v, str)]
                ent = next((_key(n) for n in names if _key(n) in plain), None)
                if ent:
                    for k, v in x.items():
                        if isinstance(v, (int, float)) and not isinstance(v, bool):
                            found.setdefault((ent, group_of(k)), []).append(float(v))
                # {"drift": {"mase": 0.8141, ...}}: the entity is the key (backtest_series "models"). Round 20261128-201606
                # (AUUNEMP): the answer quoted those MASE figures and was judged against another backtest's only.
                for k, v in x.items():
                    if isinstance(v, dict) and isinstance(k, str) and _key(k) in plain:
                        for f, n in v.items():
                            if isinstance(n, (int, float)) and not isinstance(n, bool):
                                found.setdefault((_key(k), group_of(f)), []).append(float(n))
                cols = x.get("columns")
                rows = x.get("rows") or x.get("preview") or x.get("sample")
                if isinstance(cols, list) and isinstance(rows, list):
                    for r in rows[:2000]:
                        if not isinstance(r, list) or len(r) != len(cols):
                            continue
                        ent = next((_key(c) for c in r if isinstance(c, str) and _key(c) in plain), None)
                        if ent:
                            for h, c in zip(cols, r):
                                if (v := _value(str(c))) is not None:
                                    found.setdefault((ent, group_of(h)), []).append(v[0])
    return found


def mismatches(answer: str, history: list) -> list[dict]:
    """Answer cells whose entity and column the evidence prints, with a value no printed one matches."""
    cells = answer_cells(answer)
    if not cells:
        return []
    found = evidence_cells(history, {c["key"] for c in cells})
    out = []
    # Round 20261128-125459 (EPL): the answer's standings named West Ham, Wolves, Leicester and Southampton, none of them in
    # any tool output (the printed table has the 20 clubs of this season), and a row with no evidence was never judged.
    # When the tools printed points for ten or more entities, a points or played cell for an entity in no printed table is
    # an invented row.
    printed_keys = {k for k, _ in found}
    if sum(1 for _, g in found if g == "points") >= 10:
        for key in sorted({c["key"] for c in cells if c["group"] in ("points", "played")} - printed_keys):
            c = next(x for x in cells if x["key"] == key and x["group"] in ("points", "played"))
            out.append({**c, "printed": [], "absent": True})
    for c in cells:
        printed = found.get((c["key"], c["group"]))
        if not printed:
            continue
        tol = 0.5 * 10 ** (-c["decimals"]) + 1e-9
        if any(abs(p - c["value"]) <= tol for p in printed):
            continue
        # "74.2%" for a printed share 0.7415 (round 20261128-201606, EPL: every probability cell of a correct table failed).
        if str(c["stated"]).rstrip().endswith("%") and any(abs(100 * p - c["value"]) <= tol for p in printed):
            continue
        shown = sorted({round(p, 4) for p in printed})[:4]
        out.append({**c, "printed": shown})
    return out
