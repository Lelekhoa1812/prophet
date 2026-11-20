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
_NUM = re.compile(r"(?<![A-Za-z_\d.])-?\d[\d,]*(?:\.\d+)?%?")
# A date is a label, not a figure: keep its year and drop the month and day ("2026-01-19" -> "2026").
_MONTH = r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*\.?"
_DATE = re.compile(rf"\b{_MONTH}\s+(?:\d{{1,2}},?\s+)?\d{{4}}\b|\b\d{{1,2}}\s+{_MONTH}\s+\d{{4}}\b|\b(\d{{4}})-\d{{1,2}}(?:-\d{{1,2}})?\b|\b\d{{1,2}}/\d{{1,2}}/(\d{{2,4}})\b|\b\d{{1,2}}:\d{{2}}(?::\d{{2}})?\b")


# Models write dates and minus signs with non-breaking hyphens, en dashes and the true minus sign.
_DASHES = {ord(c): "-" for c in "\u2010\u2011\u2012\u2013\u2212"}


def _parse(token: str) -> tuple[float, int, bool]:
    pct = token.endswith("%")
    raw = token.rstrip("%").replace(",", "").rstrip(",")
    decimals = len(raw.split(".")[1]) if "." in raw else 0
    return float(raw), decimals, pct


def figures(text: str) -> list[str]:
    """The numeric tokens in a text, in order of appearance."""
    text = (text or "").translate(_DASHES)
    text = _DATE.sub(lambda m: (m.group(1) or m.group(2) or "") if m.lastindex else "", text or "")
    return [m.group().rstrip(",") for m in _NUM.finditer(text)]


def _near(value: float, target: float, decimals: int) -> bool:
    return abs(value - target) <= 0.5 * 10 ** -decimals + 1e-12


def unverified(answer: str, history: list, goal: str) -> list[str]:
    """Figures in the answer that no number in the evidence or the goal supports. Empty means grounded."""
    pool = [_parse(t)[0] for t in figures(canonical(history) + "\n" + goal)]
    missing: list[str] = []
    for token in figures(answer):
        value, decimals, pct = _parse(token)
        candidates = [value / 100.0] if pct else []
        if any(_near(v, value, decimals) for v in pool) or any(_near(v, c, decimals + 2) for v in pool for c in candidates):
            continue
        if token not in missing:
            missing.append(token)
    return missing
