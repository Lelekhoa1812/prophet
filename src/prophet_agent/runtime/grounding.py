"""Grounding check: every figure in a written answer must appear in the evidence the run gathered.

Motivation vs Logic
Motivation: a confident number with no source is the costliest kind of error in an analysis. The planner and the
evaluator can both slip, so the check does not trust either of them: it compares the figures in the answer with the
numbers the tool results actually printed.
Logic: pull numeric tokens from the answer, and from the evidence and goal (the goal may carry a horizon or a threshold).
A figure matches when some evidence number rounds to it at the precision it was written: within half a unit of its last
written digit. Deliberately strict, so 1,205 does not match 1,204. A percentage also matches a fraction in the evidence
(0.12 for 12%). The check reports unmatched figures. It never rewrites them; the goal loop asks for a rewrite.
"""
from __future__ import annotations

import re

from .store import canonical

# A number as people write it: optional sign, digits with thousands separators, optional decimals, optional per cent.
_NUM = re.compile(r"(?<![A-Za-z_\d.])-?\d[\d,]*(?:\.\d+)?%?(?!\d|(?:st|nd|rd|th)\b)")
# A date is a label, not a figure: "2026-01-19", "Jan 2026" and "3 June 2027" are dropped whole.
_MONTH = r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*\.?"
_DATE = re.compile(rf"\b{_MONTH}\s+(?:\d{{1,2}},?\s+)?\d{{4}}\b|\b\d{{1,2}}\s+{_MONTH}\s+\d{{4}}\b|\b(\d{{4}})-\d{{1,2}}(?:-\d{{1,2}})?\b|\b\d{{1,2}}/\d{{1,2}}/(\d{{2,4}})\b|\b\d{{1,2}}:\d{{2}}(?::\d{{2}})?\b")


# A day or a range of days beside a real month name ("16-19 January", "Jan 16", "March 3-5") is a date label as well. Only true
# month names count, so "Mark 15" or "Decision 12" are not swallowed.
_REAL_MONTH = (r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|June?|July?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|"
               r"Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)")
_DAYS = r"\d{1,2}(?:(?:st|nd|rd|th)\b)?(?:\s*-\s*\d{1,2}(?:(?:st|nd|rd|th)\b)?)?"
_DAY_MONTH = re.compile(rf"\b{_DAYS}\s+(?:of\s+)?{_REAL_MONTH}\b|\b{_REAL_MONTH}\.?\s+{_DAYS}(?![\d%,.]*\d)(?!%)")

# Models write dates and minus signs with non-breaking hyphens, en dashes and the true minus sign.
_YEAR = re.compile(r"(?:19|20)\d\d")
_DASHES = {ord(c): "-" for c in "\u2010\u2011\u2012\u2013\u2212"}


def _parse(token: str) -> tuple[float, int, bool]:
    pct = token.endswith("%")
    raw = token.rstrip("%").replace(",", "").rstrip(",")
    decimals = len(raw.split(".")[1]) if "." in raw else 0
    return float(raw), decimals, pct


_ROW_INDEX = re.compile(r"^(\s*\|\s*)(\d{1,3})(\s*\|)", re.M)


def _drop_row_indexes(text: str) -> str:
    """A table whose first column counts 1, 2, 3 ... numbers its rows: that column is a label, not a figure."""
    rows = list(_ROW_INDEX.finditer(text))
    keep: list[re.Match] = []
    run: list[re.Match] = []
    for m in rows + [None]:                                  # type: ignore[list-item]
        if m is not None and (not run or (int(m.group(2)) == int(run[-1].group(2)) + 1 and text[run[-1].end():m.start()].count("\n") <= 2)):
            run.append(m)
            continue
        if len(run) >= 3 and int(run[0].group(2)) in (0, 1):
            keep.extend(run)
        run = [m] if m is not None else []
    for m in reversed(keep):
        text = text[:m.start(2)] + " " * len(m.group(2)) + text[m.end(2):]
    return text


def figures(text: str) -> list[str]:
    """The numeric tokens in a text, in order of appearance."""
    text = _drop_row_indexes((text or "").translate(_DASHES))
    text = _DAY_MONTH.sub("", _DATE.sub("", text or ""))
    return [m.group().rstrip(",") for m in _NUM.finditer(text)]


def _near(value: float, target: float, decimals: int) -> bool:
    return abs(value - target) <= 0.5 * 10 ** -decimals + 1e-12


def evidence_text(history: list, goal: str) -> str:
    """What a figure may be checked against: what the tools returned, what the person answered, and the goal they
    wrote. Not the model's own words. Its tool arguments, rationale, questions and draft reasons are excluded, because a
    model that checks its figures against its own output would verify any number it made up."""
    parts = [goal or ""]
    for h in history:
        kind = h.get("kind")
        if kind == "tool" and "result" in h:
            parts.append(canonical(h["result"]))
        elif kind in ("ask_user", "evaluator_question") and "answers" in h:
            parts.append(canonical(h["answers"]))
    return "\n".join(parts)


def unverified(answer: str, history: list, goal: str) -> list[str]:
    """Figures in the answer that no number in the evidence or the goal supports. Empty means grounded."""
    pool = [_parse(t)[0] for t in figures(evidence_text(history, goal))]
    missing: list[str] = []
    for token in figures(answer):
        if _YEAR.fullmatch(token):
            continue                  # a bare year such as 2026 names a period; a forecast period is not in the evidence
        value, decimals, pct = _parse(token)
        candidates = [value / 100.0] if pct else []
        if any(_near(v, value, decimals) for v in pool) or any(_near(v, c, decimals + 2) for v in pool for c in candidates):
            continue
        if token not in missing:
            missing.append(token)
    return missing
