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

import json
import re

# A number as people write it: optional sign, digits with thousands separators, optional decimals, optional per cent.
_NUM = re.compile(r"(?<![A-Za-z_\d.])-?\d[\d,]*(?:\.\d+)?(?:[  ]?%)?(?!\d|(?:st|nd|rd|th)\b)")
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
    raw = token.rstrip("%").strip().replace(",", "").rstrip(",")
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


# Structural numbers are labels, not claims about the data. They are blanked before matching, and only these forms:
#   - a numbered list marker at the start of a line ("2. Online"),
#   - an interval level or percentile name ("80% prediction interval", "10th-90th percentile", "P10", "8 in 10"),
#   - an identifier after its noun ("Week 12", "promo 9", "store 3", "horizon 8"),
#   - a conventional significance threshold ("p < 0.05"), which states a decision rule rather than a measured value.
_LIST_MARK = re.compile(r"^(\s*)\d{1,2}([.)])\s", re.M)
_LEVEL = re.compile(
    r"\b\d{1,2}(?:\.\d+)?\s?%\s+(?:prediction |likely |confidence |credible )?(?:interval|band)\b"
    r"|\b\d{1,2}(?:st|nd|rd|th)?\s*(?:-|\u2013|to|and)\s*\d{1,2}(?:st|nd|rd|th)?\s+(?:percentile|quantile)s?\b"
    r"|\b\d{1,2}(?:st|nd|rd|th)\s+(?:percentile|quantile)s?\b"
    r"|\b[Pp]\d{1,2}\b", re.I)
_P_THRESHOLD = re.compile(r"\bp\s*[<>=\u2264\u2265]\s*(0?\.\d+)", re.I)
_CONVENTIONAL_P = {0.1, 0.05, 0.01, 0.001}
_LABEL_NOUN = (r"(?:week|day|month|quarter|promo|store|region|product|sku|row|item|customer|rank|step|run|batch|origin|"
               r"horizon|period|model|window|fold|series|id|order|invoice|sheet)")
# A label noun, singular or plural, then a number or a list or range of numbers ("week 12", "rows 60 through 66",
# "customer IDs 41-48", "days 3, 10, 17, 24"). Every number in the phrase names an identifier, not a measured value.
_IDENTIFIER = re.compile(
    rf"\b{_LABEL_NOUN}s?\s*#?\s*\d{{1,4}}(?:\s*(?:,|-|\u2013|\u2014|to|through)\s*\d{{1,4}})*\b", re.I)
# Standard quantile levels written as a band ("10\u201390%", "5-95% interval"). Only these pairs are labels. Any other
# range, such as an invented width "\u00b115\u201320%", is a claim the evidence has to support.
_LEVEL_PAIR = re.compile(r"(?<![\w.])(\d{1,2}(?:\.\d+)?)\s*(?:-|\u2013|\u2014|to)\s*(\d{1,2}(?:\.\d+)?)")
_LEVEL_PAIRS = {(10.0, 90.0), (5.0, 95.0), (2.5, 97.5), (25.0, 75.0), (20.0, 80.0)}
_BAND_WORD = re.compile(r"\s*%|\s*(?:bands?|pctl|percentiles?|quantiles?|prediction|intervals?|range)\b", re.I)


def _blank(text: str, span: tuple[int, int]) -> str:
    """Blanks a span of the text, so that its digits are no longer numeric tokens."""
    a, b = span
    return text[:a] + " " * (b - a) + text[b:]


def _drop_labels(text: str) -> str:
    text = _LIST_MARK.sub(lambda m: m.group(1) + " " * (len(m.group(0)) - len(m.group(1))), text)
    # The band pair goes first: "5-95% interval" must be read whole before "95% interval" alone is blanked.
    for m in list(_LEVEL_PAIR.finditer(text))[::-1]:
        pair = (float(m.group(1)), float(m.group(2)))
        band = _BAND_WORD.match(text, m.end())
        if pair in _LEVEL_PAIRS and band:
            text = _blank(text, (m.start(), band.end()))
    text = _LEVEL.sub(lambda m: " " * len(m.group(0)), text)
    for m in list(_P_THRESHOLD.finditer(text))[::-1]:
        if float(m.group(1)) in _CONVENTIONAL_P:
            text = _blank(text, m.span(1))
    for m in list(_IDENTIFIER.finditer(text))[::-1]:
        text = _blank(text, m.span())
    return text


def _cleaned(text: str) -> str:
    text = _drop_row_indexes((text or "").translate(_DASHES))
    # Dates are read before labels are blanked: "Week 2026-01-05" must not leave "-01" and "05" behind as figures.
    text = _DAY_MONTH.sub("", _DATE.sub("", text or ""))
    return _drop_labels(text)


def figures(text: str) -> list[str]:
    """The numeric tokens in a text, in order of appearance."""
    return [m.group().rstrip(",") for m in _NUM.finditer(_cleaned(text))]


# A writer who says "about 44,500" or "~13%" has rounded a printed figure, and says so. Such a figure is matched to the evidence
# within APPROX_TOLERANCE of its value, not to the last written digit. An unmarked figure is still matched exactly.
APPROX_TOLERANCE = 0.01
_APPROX = re.compile(r"(?:~|\u2248|\b(?:about|around|roughly|approximately|approx\.?|nearly|almost|circa|near)\b)[\s\u00b1]*(?:[\w\u2013\-\s,.]{0,14}?)$", re.I)


def _figures_in_context(text: str) -> list[tuple[str, bool]]:
    """Each numeric token with whether the words just before it mark it as an approximation."""
    cleaned = _cleaned(text)
    return [(m.group().rstrip(","), bool(_APPROX.search(cleaned[max(0, m.start() - 30):m.start()])))
            for m in _NUM.finditer(cleaned)]


def _near(value: float, target: float, decimals: int) -> bool:
    return abs(value - target) <= 0.5 * 10 ** -decimals + 1e-12


def _as_text(obj) -> str:
    """Evidence as text for matching. Unlike ``canonical`` (compact, used for idempotency keys), the separators are
    spaced: compact JSON reads ``[360,365,370,375]`` as one thousands-separated number, so no figure in it would match."""
    return json.dumps(obj, sort_keys=True, separators=(", ", ": "), ensure_ascii=False, default=str)


def _observed(history: list):
    """What the run observed: tool results, the person's answers, and a sub-agent's own evidence and report. Arguments,
    rationale and the planner's words are never yielded."""
    for h in history:
        kind = h.get("kind")
        if kind == "tool" and "result" in h:
            yield _as_text(h["result"])
        elif kind in ("ask_user", "evaluator_question") and "answers" in h:
            yield _as_text(h["answers"])
        elif kind == "subagent":
            yield from _observed(h.get("evidence") or [])
            if h.get("answer"):
                yield str(h["answer"])


def evidence_text(history: list, goal: str) -> str:
    """What a figure may be checked against: what the tools returned, what the person answered, and the goal they
    wrote. Not the model's own words. Its tool arguments, rationale, questions and draft reasons are excluded, because a
    model that checks its figures against its own output would verify any number it made up."""
    return "\n".join([goal or "", *_observed(history)])


def unverified(answer: str, history: list, goal: str) -> list[str]:
    """Figures in the answer that no number in the evidence or the goal supports. Empty means grounded."""
    pool = [_parse(t)[0] for t in figures(evidence_text(history, goal))]
    missing: list[str] = []
    for token, approximate in _figures_in_context(answer):
        if _YEAR.fullmatch(token):
            continue                  # a bare year such as 2026 names a period; a forecast period is not in the evidence
        value, decimals, pct = _parse(token)
        candidates = [value / 100.0] if pct else []
        if any(_near(v, value, decimals) for v in pool) or any(_near(v, c, decimals + 2) for v in pool for c in candidates):
            continue
        if approximate and any(abs(v - t) <= APPROX_TOLERANCE * max(abs(t), 1e-9) for v in pool
                               for t in ([value] + candidates)):
            continue
        if token not in missing:
            missing.append(token)
    return missing
