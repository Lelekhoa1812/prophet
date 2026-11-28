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

# A number as people write it: optional sign, digits with thousands separators, optional decimals, then an optional
# per cent or a scale word attached to it ("27.9k", "1.2m", "3.4bn"). The scale must touch the digits and end the word, so
# "5 min" and "12 km" are not read as five million and twelve thousand.
_NUM = re.compile(r"(?<![A-Za-z_\d.])-?\d[\d,]*(?:\.\d+)?(?:[  ]?%|(?:[kKmM]|bn|BN|Bn)(?![A-Za-z]))?(?!\d|(?:st|nd|rd|th)\b)")
_SCALES = {"k": 1e3, "m": 1e6, "bn": 1e9}
# A date is a label, not a figure: "2026-01-19", "Jan 2026" and "3 June 2027" are dropped whole.
_MONTH = r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*\.?"
_DATE = re.compile(rf"\b{_MONTH}\s+(?:\d{{1,2}},?\s+)?\d{{4}}\b|\b\d{{1,2}}\s+{_MONTH}\s+\d{{4}}\b|\b\d{{4}}-(?:0[1-9]|1[0-2])(?:-(?:0[1-9]|[12]\d|3[01]))?\b|\b\d{{1,2}}/\d{{1,2}}/(\d{{2,4}})\b|\b\d{{1,2}}:\d{{2}}(?::\d{{2}})?\b")


# A day or a range of days beside a real month name ("16-19 January", "Jan 16", "March 3-5") is a date label as well. Only true
# month names count, so "Mark 15" or "Decision 12" are not swallowed.
_REAL_MONTH = (r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|June?|July?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|"
               r"Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)")
# A day is 1 to 31: "May 40 cases" is a month beside a count, not a date (audit round 6m).
_DAY = r"(?:0?[1-9]|[12]\d|3[01])"
_DAYS = rf"{_DAY}(?:(?:st|nd|rd|th)\b)?(?:\s*-\s*{_DAY}(?:(?:st|nd|rd|th)\b)?)?"
_DAY_MONTH = re.compile(rf"\b{_DAYS}\s+(?:of\s+)?{_REAL_MONTH}\b|\b{_REAL_MONTH}\.?\s+{_DAYS}(?![\d%,.]*\d)(?!%)")

# Models write dates and minus signs with non-breaking hyphens, en dashes and the true minus sign.
_YEAR = re.compile(r"(?:19|20)\d\d")
_DASHES = {ord(c): "-" for c in "\u2010\u2011\u2012\u2013\u2212"}


def _scale_of(token: str) -> float:
    """1 for a plain number; 1,000 for "27.9k", 1,000,000 for "1.2m", 1,000,000,000 for "3.4bn"."""
    return _SCALES.get(re.sub(r"[\d.,\-%\s ]", "", token).lower(), 1.0)


def _parse(token: str) -> tuple[float, int, bool]:
    pct = token.endswith("%")
    raw = re.sub(r"[kKmM%]$|bn$|BN$|Bn$", "", token.strip()).strip().replace(",", "").rstrip(",")
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
    # Panel round 4 (EPL, NBA): "95% CI" and "95% bootstrap interval" are labels too.
    r"\b\d{1,2}(?:\.\d+)?\s?%\s+(?:(?:prediction|likely|confidence|credible|bootstrap(?:ped)?|uncertainty|forecast)\s+)*"
    r"(?:intervals?|bands?|CI|PI|CrI)\b"
    r"|\b\d{1,2}(?:st|nd|rd|th)?\s*(?:-|\u2013|to|and)\s*\d{1,2}(?:st|nd|rd|th)?\s+(?:percentile|quantile)s?\b"
    r"|\b\d{1,2}(?:st|nd|rd|th)\s+(?:percentile|quantile)s?\b"
    r"|\b[Pp]\d{1,2}\b", re.I)
# An inequality states a decision rule. "p = 0.05" is a measured p-value and is checked like any figure (audit round 6m).
_P_THRESHOLD = re.compile(r"\bp\s*[<>\u2264\u2265]\s*(0?\.\d+)", re.I)
_CONVENTIONAL_P = {0.1, 0.05, 0.01, 0.001}
_LABEL_NOUN = (r"(?:week|day|month|quarter|promo|store|region|product|sku|row|item|customer|rank|step|run|batch|origin|"
               r"horizon|period|model|window|fold|series|id|order|invoice|sheet)")
# A label noun, singular or plural, then a number or a list or range of numbers ("week 12", "rows 60 through 66",
# "customer IDs 41-48", "days 3, 10, 17, 24"). Every number in the phrase names an identifier, not a measured value.
#
# A list separator is a comma followed by a space, so a thousands group ("customers 1,200") is a measured value, not a list, and
# it is not blanked. A number that continues into a decimal or a thousands group is not an identifier either (probe: "Customers
# 1,200 bought" lost its figure and "Store 3.5" left ".5" behind before this).
# A slash joins two identifiers ("step 37/38"). It does not join a decimal: the closing lookahead still refuses "step 3.5".
_IDENTIFIER = re.compile(
    rf"\b{_LABEL_NOUN}s?\s*#?\s*\d{{1,4}}(?:\s*(?:,\s+|-|\u2013|\u2014|/|to|through)\s*\d{{1,4}})*\b(?![.,]\d)", re.I)
# Standard quantile levels written as a band ("10\u201390%", "5-95% interval"). Only these pairs are labels. Any other
# range, such as an invented width "\u00b115\u201320%", is a claim the evidence has to support.
# Round 7 (CO2): "10/90 quantiles" is the same label written with a slash.
_LEVEL_PAIR = re.compile(r"(?<![\w.])(\d{1,2}(?:\.\d+)?)\s*(?:-|\u2013|\u2014|to|/)\s*(\d{1,2}(?:\.\d+)?)")
_LEVEL_PAIRS = {(10.0, 90.0), (5.0, 95.0), (2.5, 97.5), (25.0, 75.0), (20.0, 80.0)}
# A "%" makes a pair a band label only when a band word follows it ("10-90% interval", "(10-90%) range"). A bare "growth of
# 10-90%" is a measured range and keeps its figures (audit round 6m).
# Round 7 (EPL): "10\u201390% ranges" is the plural of the same label.
_BAND_WORD = re.compile(r"\s*%(?=\)?\s*(?:bands?|pctl|percentiles?|quantiles?|prediction|intervals?|ranges?)\b)|"
                        r"\s*(?:bands?|pctl|percentiles?|quantiles?|prediction|intervals?|ranges?)\b", re.I)


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


# Labels the panel of 2026-11-26 flagged as figures: a season or fiscal year written as a range ("2026-27", "2026/27"),
# the day range after an ISO date ("2021-03-19-23"), an HTTP status ("HTTP 403"), and a citation list ("[31, 33]").
_SEASON = re.compile(r"\b(?:FY\s?)?(?:19|20)\d\d\s?[/-]\s?\d\d\b(?![-/]\d)")
_DATE_DAY_RANGE = re.compile(r"(\b\d{4}-\d{2}-\d{2})-(\d{1,2})\b")
_HTTP_STATUS = re.compile(r"\b(?:HTTP|status(?: code)?|error)\s?[1-5]\d\d\b", re.I)
# Panel NGSTORAGE (2026-11-26): "the history files return 403/404" and "failed with 403 or 404" were flagged as figures.
# After a fetch verb only the common status codes count, so "returned 250 points" stays a figure to check.
_STATUS_CODE = r"(?:200|201|204|301|302|304|400|401|403|404|405|406|408|410|418|429|451|500|502|503|504)"
_HTTP_STATUS_VERB = re.compile(
    rf"\b(?:returns?|returned|returning|failed with|fails with|gave|gives|answered with|responded with|blocked with)\s"
    rf"(?:an?\s)?{_STATUS_CODE}(?:\s?(?:/|,|or|and)\s?{_STATUS_CODE})*\b"
    rf"|\b{_STATUS_CODE}(?:\s?/\s?{_STATUS_CODE})+\b", re.I)
_CITATION = re.compile(r"\[\s?\d{1,3}(?:\s?,\s?\d{1,3})*\s?\]")


def _drop_reference_labels(text: str) -> str:
    text = _DATE_DAY_RANGE.sub(lambda m: m.group(1) + " " * (len(m.group(0)) - len(m.group(1))), text)
    for pattern in (_HTTP_STATUS, _HTTP_STATUS_VERB, _CITATION):
        text = pattern.sub(lambda m: " " * len(m.group(0)), text)
    # A season range is dropped whole only when its two years run on ("2026-27"); "2026-11" is a month and stays for _DATE.
    return _SEASON.sub(lambda m: " " * len(m.group(0)) if _consecutive_years(m.group(0)) else m.group(0), text)


def _consecutive_years(token: str) -> bool:
    digits = re.findall(r"\d+", token)
    return len(digits) == 2 and int(digits[1]) == (int(digits[0]) + 1) % 100


def _cleaned(text: str, references: bool = True) -> str:
    """The text with dates and labels blanked. ``references`` drops the reference labels of written answers (seasons,
    HTTP statuses, citation lists); evidence keeps them, since a JSON list such as [360, 365] holds real figures."""
    text = _drop_row_indexes((text or "").translate(_DASHES))
    if references:
        text = _drop_reference_labels(text)
    # Dates are read before labels are blanked: "Week 2026-01-05" must not leave "-01" and "05" behind as figures.
    text = _DAY_MONTH.sub("", _DATE.sub("", text or ""))
    return _drop_labels(text)


def figures(text: str, references: bool = True) -> list[str]:
    """The numeric tokens in a text, in order of appearance (``references`` as in _cleaned)."""
    return [m.group().rstrip(",") for m in _NUM.finditer(_cleaned(text, references))]


# A writer who says "about 44,500" or "~13%" has rounded a printed figure, and says so. Such a figure is matched to the evidence
# within APPROX_TOLERANCE of its value, not to the last written digit. An unmarked figure is still matched exactly.
APPROX_TOLERANCE = 0.01
_APPROX = re.compile(r"(?:~|\u2248|\b(?:about|around|roughly|approximately|approx\.?|nearly|almost|circa|near)\b)[\s\u00b1]*(?:[\w\u2013\-\s,.]{0,14}?)$", re.I)


def _figures_in_context(text: str) -> list[tuple[str, bool]]:
    """Each numeric token with whether the words just before it mark it as an approximation."""
    cleaned = _cleaned(text)
    return [(m.group().rstrip(","), bool(_APPROX.search(cleaned[max(0, m.start() - 30):m.start()])))
            for m in _NUM.finditer(cleaned)]


def _near(value: float, target: float, decimals: int, scale: float = 1.0) -> bool:
    return abs(value - target) <= 0.5 * 10 ** -decimals * scale + 1e-12


def _as_text(obj) -> str:
    """Evidence as text for matching. Unlike ``canonical`` (compact, used for idempotency keys), the separators are
    spaced: compact JSON reads ``[360,365,370,375]`` as one thousands-separated number, so no figure in it would match."""
    # Round 13 (NEM): a fetched page held "Demand\n8,301 MW"; dumped as JSON the newline became backslash-n, the "n" sat
    # against the figure, and 8,301 was never read from the evidence. Escaped whitespace is read as a space.
    return re.sub(r"\\[ntr]", " ", json.dumps(obj, sort_keys=True, separators=(", ", ": "), ensure_ascii=False, default=str))


def _string_literals(code: str) -> list[str]:
    """The text of every string literal in Python code, with the literal parts of f-strings (not their computed fields)."""
    import ast
    import warnings
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")      # "invalid escape sequence" in the model's code is not ours to report
            tree = ast.parse(code)
    except (SyntaxError, ValueError):
        return [m[1] for m in re.findall(r"([\"'])((?:(?!\1).){1,400})\1", code)]
    out: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            out.append(node.value)
    return out


def _typed_numbers(history: list) -> set[str]:
    """Numbers the model typed inside string literals of its own sandbox code (a ledger finding written as "seasonal strength
    0.34") that no earlier tool output held, across the run, sub-agents included.

    Motivation: panel round 11 audit. AUUNEMP's answer gave "strength 0.34" and "1.7% outliers", and CO2's five break dates;
    each was a number the model wrote into a Ledger.add text. The sandbox printed the ledger, so the number came back as
    "tool output" and passed the grounding check, while the scan had measured 0.0, 4.4% and other dates. A number the code
    only echoes from its own text is the model's claim, not a measurement. A number copied from an earlier tool output (the
    3,141 missing days prepare_series printed, written again into a ledger line) is not typed: it was seen first.
    """
    out: set[str] = set()
    seen: list[str] = []
    for h in _flat(history):
        code = (h.get("args") or {}).get("code") if h.get("tool") == "sandbox_python" else None
        if isinstance(code, str):
            before = _number_keys(" ".join(seen))
            for text in _string_literals(code):
                out.update(t for t in _figures_in_text(text)
                           if not _YEAR.fullmatch(t) and _number_key(t) not in before)
        for key in ("result", "full"):
            if key in h:
                seen.append(_as_text(h[key]))
        if h.get("kind") in ("ask_user", "evaluator_question") and "answers" in h:
            seen.append(_as_text(h["answers"]))
    return out


def _flat(history: list) -> list[dict]:
    """Tool observations and answers in order, with a sub-agent's evidence in place."""
    out: list[dict] = []
    for h in history:
        if not isinstance(h, dict):
            continue
        if h.get("kind") == "subagent":
            out.extend(_flat(h.get("evidence") or []))
        elif h.get("kind") in ("tool", "ask_user", "evaluator_question"):
            out.append(h)
    return out


def _number_key(token: str) -> str:
    try:
        return repr(float(token.replace(",", "")))
    except ValueError:
        return token


def _number_keys(text: str) -> set[str]:
    return {_number_key(t) for t in _figures_in_text(text)}


def _figures_in_text(text: str) -> list[str]:
    return re.findall(r"(?<![\w.])\d+(?:,\d{3})*(?:\.\d+)?(?![\w.]*\d)", text or "")


def _scrubbed(text: str, typed: set[str] | None) -> str:
    """Evidence text with the model's typed numbers blanked, when the result is sandbox output that may echo them."""
    if not typed:
        return text
    pattern = re.compile(r"(?<![\w.])(?:" + "|".join(re.escape(t) for t in sorted(typed, key=len, reverse=True))
                         + r")(?![\w]|\.\d)")
    return pattern.sub(" ", text)


def _observed(history: list):
    """What the run observed: tool results, the person's answers, and a sub-agent's own tool evidence. Arguments,
    rationale, the planner's words and a sub-agent's written answer are never yielded: the written answer is a claim
    made by the model, so a figure in it is checked against tool results, not accepted because the model wrote it
    (audit 2026-11-22, item 1)."""
    typed = _typed_numbers(history)
    for h in history:
        kind = h.get("kind")
        if kind == "tool" and "result" in h:
            scrub = typed if h.get("tool") == "sandbox_python" else None
            yield _scrubbed(_as_text(h["result"]), scrub)
            if "full" in h:              # what the tool really returned, before its summary cut it (round 4, live panels)
                yield _scrubbed(_as_text(h["full"]), scrub)
        elif kind in ("ask_user", "evaluator_question") and "answers" in h:
            yield _as_text(h["answers"])
        elif kind == "subagent":
            yield from _observed(h.get("evidence") or [])


def evidence_text(history: list, goal: str) -> str:
    """What a figure may be checked against: what the tools returned, what the person answered, and the goal they
    wrote. Not the model's own words. Its tool arguments, rationale, questions and draft reasons are excluded, because a
    model that checks its figures against its own output would verify any number it made up."""
    return "\n".join([goal or "", *_observed(history), *_computed(history)])


def _computed(history: list):
    """Figures code computes from stored results and appends to the answer itself: the measured range of the horizon total.
    They are the ledger's own numbers, so an answer that repeats them is grounded. Live retail_biz (2026-11-23): the
    appended range 777,152 to 1,137,370 was flagged as unverified, and the run ended partial for it."""
    from prophet_agent.runtime import contracts      # here, not at import: contracts imports this module
    try:
        rng = contracts.measured_total_range(history)
    except Exception:
        return
    if rng:
        yield " ".join(f"{rng[k]:,.0f} {rng[k]:.2f}" for k in ("q10", "q90", "centre", "mean", "mean_q10", "mean_q90"))


def unverified(answer: str, history: list, goal: str) -> list[str]:
    """Figures in the answer that no number in the evidence or the goal supports. Empty means grounded."""
    # Magnitudes: "a loss of 1,683.5" or "-\u00a31,683.5" is the printed -1683.5, whose sign the prose carries in its words.
    pool = []
    for t in figures(evidence_text(history, goal), references=False):
        v, _, _ = _parse(t)
        pool.append(abs(v) * _scale_of(t))
    missing: list[str] = []
    for token, approximate in _figures_in_context(answer):
        if _YEAR.fullmatch(token):
            continue                  # a bare year such as 2026 names a period; a forecast period is not in the evidence
        value, decimals, pct = _parse(token)
        value = abs(value)
        scale = _scale_of(token)
        candidates = [value / 100.0] if pct else []
        # "27.9k" is a rounding of a printed 27,912: it claims 27,850 to 27,950, half a unit of its last digit at its own scale.
        if any(_near(v, value * scale, decimals, scale) for v in pool) or any(_near(v, c, decimals + 2) for v in pool for c in candidates):
            continue
        if scale != 1.0 and any(_near(v, value, decimals) for v in pool):
            continue                  # "5m" that is metres, not millions, and 5 is printed
        if approximate and any(abs(v - t) <= APPROX_TOLERANCE * max(abs(t), 1e-9) for v in pool
                               for t in ([value * scale] + candidates)):
            continue
        if token not in missing:
            missing.append(token)
    for token in entity_row_mismatches(answer, history):
        if token not in missing:
            missing.append(token)
    for token in untraced_dates(answer, history, goal):
        if token not in missing:
            missing.append(token)
    from prophet_agent.runtime import table_cells
    for cell in table_cells.mismatches(answer, history):
        if cell["stated"] not in missing:
            missing.append(cell["stated"])
    return missing


_ISO_DATE = re.compile(r"\b(?:19|20)\d\d-[01]\d-[0-3]\d\b")
_FINDING = re.compile(r"\b(?:shift|break|outlier|spike|peak|trough|jump|drop|fell|rose|high|low|change|level|regime|"
                      r"anomal|surge|plunge|crash|rall|turning point)", re.I)
_AS_OF = re.compile(r"\b(?:today|as of|as at|run date|retrieved|accessed|published)\b", re.I)


def untraced_dates(answer: str, history: list, goal: str) -> list[str]:
    """ISO dates the answer gives for a finding in the data (a shift, a break, a peak) that no tool output prints.

    Motivation: live SPX (round 20261128-063203) wrote "three level shifts (2020-11-05, 2024-01-24, 2025-08-08)"; the scan
    printed one break, on 2020-11-05, and the ledger said no level shifts. The figure check skipped dates. Measured on that
    round's answers, the only other untraced date was the run's own date ("as of"), which this check leaves alone."""
    found = _ISO_DATE.findall(answer or "")
    if not found:
        return []
    text = evidence_text(history, goal)
    out: list[str] = []
    for sentence in re.split(r"(?<=[.!?;])\s+|\n+", answer):
        if not _FINDING.search(sentence) or _AS_OF.search(sentence):
            continue
        for d in _ISO_DATE.findall(sentence):
            if d not in text and d not in out:
                out.append(d)
    return out


# A writer copying a printed 557,051.21 as "557,050", or 352,602 as "352,702", has made a slip, not a claim. When an unmatched
# figure of 1,000 or more lies within the snap tolerance of exactly one printed value that has decimals, it is corrected to that
# value. Printed whole numbers are never snapped: 10,001 rows and 10,000 rows are different facts.
# The tolerance is the smaller of 0.05% of the figure and SNAP_MAX_ABS units. The relative bound alone let a written 10,000,000
# become the printed 10,004,900 (4,900 units apart, audit round 6l item 4); the absolute bound stops that at 1,000 units, which
# still covers a copying slip of a digit in the hundreds (352,602 as 352,702).
SNAP_TOLERANCE = 0.0005
SNAP_MAX_ABS = 1000.0
SNAP_MIN = 1000.0


def _token_pattern(digits: str) -> re.Pattern:
    """The digits as a whole figure: not inside a longer number, and not the integer part of a decimal."""
    return re.compile(r"(?<![\d.,])" + re.escape(digits.lstrip("-")) + r"(?!\d|[.,]\d)")


def snap(answer: str, history: list, goal: str) -> tuple[str, list[tuple[str, str]]]:
    """The answer with transcription slips in large figures corrected to the printed value, and the (old, new) pairs."""
    bad = unverified(answer, history, goal)
    if not bad:
        return answer, []
    printed = {}
    for t in figures(evidence_text(history, goal), references=False):
        v, dec, pct = _parse(t)
        if dec > 0 and not pct and _scale_of(t) == 1.0:
            printed[abs(v)] = dec
    fixed: list[tuple[str, str]] = []
    for token in bad:
        value, decimals, pct = _parse(token)
        v = abs(value)
        if pct or _scale_of(token) != 1.0 or _YEAR.fullmatch(token) or value < 0:
            continue
        if decimals == 0 and v < SNAP_MIN:
            # A whole number cut down from a printed decimal: 198 for 198.77 (Beijing, live 2026-11-23). Cut down, not rounded.
            # Corrected only when every printed value that the figure could be cut from rounds up to the same whole number.
            ups = {int(p // 1) + 1 for p in printed if int(p // 1) == int(v) and p - p // 1 >= 0.5}
            pattern = _token_pattern(token)
            if len(ups) == 1 and pattern.search(answer):
                new = str(ups.pop())
                answer = pattern.sub(new, answer)
                fixed.append((token, new))
            continue
        if v < SNAP_MIN:
            continue
        limit = min(SNAP_TOLERANCE * v, SNAP_MAX_ABS)
        near = sorted({p for p in printed if abs(p - v) <= limit})
        if len(near) != 1:
            continue
        new = f"{near[0]:,.{decimals}f}" if "," in token else f"{near[0]:.{decimals}f}"
        digits = token.lstrip("-")
        pattern = re.compile(r"(?<![\d.,])" + re.escape(digits) + r"(?!\d|[.,]\d)")
        if pattern.search(answer):
            answer = pattern.sub(new, answer)
            fixed.append((digits, new))
    return answer, fixed


def snap_note(fixed: list[tuple[str, str]]) -> str:
    return (f"\n\n_{len(fixed)} figure(s) were corrected to the value the analysis printed (a copying slip, or a printed decimal "
            "cut down to its whole part)._"
            if fixed else "")


# ---------------------------------------------------------------- narrative claims (A6-09, A4-14, A4-04)
#
# A figure check does not catch a claim with no number in it: "weekly (7-day) confirmed" or "naive is the appropriate
# benchmark". These readers pull the claims a written answer makes, as plain facts (a period, a best rule, a count, a cause,
# a method that was tested). contracts.check_answer compares them with what the stored tool output says. Readers only: they
# return text and numbers, and decide nothing on their own.

def sentences(text: str) -> list[str]:
    """The units a claim is read from: a table row whole, otherwise each sentence of each line."""
    units: list[str] = []
    for line in (text or "").split("\n"):
        if line.lstrip().startswith("|"):
            if line.strip():
                units.append(line)
            continue
        units.extend(s for s in re.split(r"(?<=[.!?;])\s+", line) if s.strip())
    return units


_NEGATION = re.compile(r"\b(?:not|no|never|without|none|nor|untested|unconfirmed|unverified)\b|n't\b", re.I)


def negated(unit: str, start: int) -> bool:
    """True when a negation word sits in the few words before ``start``: "is not the best", "no cycle"."""
    return bool(_NEGATION.search(unit[max(0, start - 25):start]))


_CYCLE_CLAIMS = (
    # No comma between the word and the number: "weekly cycle, 105 cycles in history" counts cycles, it is not a length.
    # "period ~12 months" (round 11, GISTEMP, where the measured period was 24): an approximate length is still a claim.
    # "Annual cycle measured at 24 months" (round 12, GISTEMP) states a length too.
    re.compile(r"\b(?:weekly|cycle|season|seasonal|period|periodic)\s{0,2}(?:\(|of\s+|:|(?:measured\s+|detected\s+|estimated\s+)?at\s+)?\s{0,2}"
               r"(?:~\s?|about\s+|approximately\s+|roughly\s+|around\s+)?(\d{1,3})[- ](?:day|step|period)?\b", re.I),
    # Only a space, a dash or a closing bracket between: "horizon 14 days, weekly seasonality" names a horizon, not a cycle.
    re.compile(r"\b(\d{1,3})[- ](?:day|step|period)s?\)?[\s-]{1,2}(?:weekly|cycles?|seasonal|seasonality|season|pattern|periodic|periodicity)\b", re.I),
    # "season=252" quotes an argument a tool was given, not a claim about the data (live 1969cd11), so "=" is not read.
    re.compile(r"\b(?:period|cycle|season)\s*(?::|of)\s*(\d{1,3})\b", re.I),
)


# A lag comparison, not a cycle: "the same period 260 days earlier" (live retail business user cf7f4aa6, 2026-11-23) is a
# year-over-year change test, and its 260 is a number of steps back, not the length of a repeating pattern.
_LAG_AFTER = re.compile(r"\s{0,2}(?:trading |business )?(?:days?|steps?|periods?|weeks?|points?)?\s{0,2}(?:earlier|ago|before|prior)\b",
                        re.I)
_SAME_BEFORE = re.compile(r"\bsame\s$", re.I)


def _is_lag(unit: str, m: re.Match) -> bool:
    return bool(_LAG_AFTER.match(unit, m.end())) or bool(_SAME_BEFORE.search(unit[max(0, m.start() - 12):m.start()]))


# Round 18 CO2: "Seasonal-naive (lag 365) MASE 1.18 ... confirming the 365-day season is wrong" is a baseline's score, and was
# read as a claim that the cycle is 365.
_BASELINE_SCORE = re.compile(r"seasonal[ _-]naive|\bMASE\b|\blag\s+\d", re.I)


def claimed_periods(answer: str) -> list[tuple[int, str]]:
    """The cycle lengths, in steps, that the answer states: "weekly (7-day) cycle" gives 7. Negated statements and lag
    comparisons ("the same period 260 days earlier") are left out."""
    out: list[tuple[int, str]] = []
    for unit in sentences(answer):
        if _BASELINE_SCORE.search(unit):        # a baseline's score line names the lag it was given, not a cycle of the data
            continue
        for pat in _CYCLE_CLAIMS:
            for m in pat.finditer(unit):
                n = int(m.group(1))
                if n >= 2 and not negated(unit, m.start()) and not _is_lag(unit, m):
                    out.append((n, unit.strip()))
    return out


_BASELINE_NAMES = (
    ("seasonal_naive", r"seasonal[ _-]naive"),
    # "mean of last season", "mean of the last 5 business days" (live wording) are the same rule
    ("mean_last_season", r"mean[ _]of[ _](?:the[ _])?last[ _](?:season|\d{1,3}[ _](?:business[ _])?(?:days?|weeks?|steps?|points?))|mean_last_season"),
    ("tsb_expected", r"\bTSB\b|intermittent[- ]demand rule"),
    ("drift", r"\bdrift\b"),
    ("naive", r"\bnaive\b"),
)
_BEST_WORDS = re.compile(r"\b(?:best|most accurate|appropriate benchmark|right benchmark|the benchmark)\b", re.I)
BEST_WINDOW = 30           # characters either side of "best" in which the rule it names must sit
_NO_BEATS = re.compile(r"\bno (?:other |simple )?(?:model|method|rule|baseline|forecast)s? (?:beats?|outperforms?|is better than"
                       r"|is more accurate than)\s+(.*)", re.I)


def _plain(text: str) -> str:
    """Accents dropped, one letter for one letter: "naïve" reads as "naive" (live retail_biz wrote "seasonal naïve")."""
    import unicodedata
    return "".join(c for c in unicodedata.normalize("NFKD", text or "") if not unicodedata.combining(c))


def baselines_named(text: str) -> list[str]:
    """The simple rules a text names, in order. A longer name is blanked first, so "seasonal naive" is not also "naive"."""
    found: list[tuple[int, str]] = []
    rest = _plain(text)
    for name, pat in _BASELINE_NAMES:
        for m in re.finditer(pat, rest, re.I):
            found.append((m.start(), name))
        rest = re.sub(pat, lambda m: " " * len(m.group()), rest, flags=re.I)
    return [n for _, n in sorted(found)]


# A stated forecast result. "The forecast is ready" names no result. "No forecast has been produced" is a gap, not a result.
_FORECAST_RESULT = re.compile(
    r"\b(?:the forecast|TimesFM)\b[^.\n]{0,60}?\b(?:is|equals|shows|gives|puts|projects|expects|forecasts)\b[^.\n]{0,24}?-?\d"
    r"|\b(?:is|are) forecast to\b"
    r"|\bforecast(?:ed)? (?:winner|champion)\b"
    r"|\bwill win\b",
    re.I)


def claimed_forecast_results(answer: str) -> list[str]:
    """Sentences that state a forecast result. A negated sentence is a gap, not a result."""
    out: list[str] = []
    for unit in sentences(answer):
        match = _FORECAST_RESULT.search(unit)
        if not match or negated(unit, match.start()) or _NEGATION.search(unit[match.start():match.end()]):
            continue
        out.append(unit.strip())
    return out


def claimed_best_baselines(answer: str) -> list[tuple[str, str]]:
    """The simple rule the answer calls the best or the appropriate benchmark, one (rule, sentence) per sentence that names
    exactly one rule. "No model beats naive" claims naive is best. A negated claim, or a sentence that names two rules, is
    left out: it says nothing this check can settle."""
    out: list[tuple[str, str]] = []
    for unit in sentences(_plain(answer)):
        beats = _NO_BEATS.search(unit)
        if beats:
            names = set(baselines_named(beats.group(1)))
            if len(names) == 1 and not negated(unit, beats.start()):
                out.append((names.pop(), unit.strip()))
            continue
        best = _BEST_WORDS.search(unit)
        if not best or negated(unit, best.start()):
            continue
        # The rule must sit next to the claim: in "more accurate than naive (lower MASE), but less accurate than the best
        # simple baseline" (live) the best is another rule. A rule named anywhere else in the sentence is not called best.
        names = set(baselines_named(unit))
        near = set(baselines_named(unit[max(0, best.start() - BEST_WINDOW): best.end() + BEST_WINDOW]))
        if len(names) == 1 and names == near:
            out.append((names.pop(), unit.strip()))
    return out


_NUMBER_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10}
# Round 13 (SPX): "Two unexplained regime shifts detected" against five breaks: an adjective sat between count and noun.
# Round 20 (NVDA): "already reflected in the 2025-07-09 and 2026-04-14 level shifts" was read as 14 shifts. A number that
# ends a date (after -, /, . or :) is a day, not a count.
_SHIFT_COUNT = re.compile(r"(?<![-/.:\d])\b(\d{1,2}|one|two|three|four|five|six|seven|eight|nine|ten)\s+"
                          r"(?:(?!of\b|the\b|these\b|those\b|which\b|or\b|and\b)[a-z-]+\s+){0,2}?"
                          r"(?:level[- ]shifts?|regime shifts?|regime changes?|structural breaks?|breaks?|changepoints?)\b", re.I)


def claimed_shift_counts(answer: str) -> list[tuple[int, str]]:
    """The number of level shifts the answer states, as a whole number."""
    out: list[tuple[int, str]] = []
    for unit in sentences(answer):
        for m in _SHIFT_COUNT.finditer(unit):
            word = m.group(1).lower()
            n = _NUMBER_WORDS[word] if word in _NUMBER_WORDS else int(word)
            if not negated(unit, m.start()):
                out.append((n, unit.strip()))
    return out


# Round 9 audit: seven of eight completed answers wrote "no level shifts", "no outliers" or "no missing values" while the
# saved scan had flagged them (CPI: four breaks and a 6.4% outlier share; SPX: 65 missing closes filled by interpolation).
SCAN_TOPICS = {
    "level_shift": re.compile(r"\b(?:level[- ]shifts?|structural breaks?|breaks?|regime (?:changes?|shifts?)|changepoints?|"
                              r"change[- ]points?)\b", re.I),
    "outliers": re.compile(r"\b(?:outliers?|spikes?)\b", re.I),
    "missing": re.compile(r"\b(?:missing(?: values?| data| days?| weeks?| months?| observations?| points?)?|gaps?|nulls?)\b",
                          re.I),
}
_DENIAL = re.compile(r"\b(?:no|zero)\s+", re.I)
_DENIAL_STOP = re.compile(r"\b(?:were|was|are|is|found|detected|flagged|in|after|before|remain(?:ed|ing)?|needing|need(?:ed|s)?|"
                          r"that|which|to|beyond|other|than|apart|except|requiring|left|new|further|additional)\b|[.;:!?()\n]",
                          re.I)
_TREATED_AFTER = re.compile(r"^[\W_]{0,3}(?:\w+\s+){0,2}(?:were\s+|was\s+|is\s+|are\s+)?(?:filled|interpolated|imputed|dropped|"
                            r"removed|treated|clipped|winsori[sz]ed|adjusted|corrected|changed|replaced|tests?|testing|tested|"
                            r"models?|modell?ed|analysis|checks?|checked|treatment|adjustments?|corrections?|applied|used|"
                            r"included|assumed|handling|handled)\b", re.I)
_STRONG = re.compile(r"^\s*(?:major|large|significant|material|notable|severe|big)\b", re.I)
_NONE_AFTER = re.compile(r"^[^.;\n]{0,40}?(?::\s*(?:none|no)\b|\b(?:none|not|never)\s+(?:were\s+|was\s+)?(?:detected|found|"
                         r"flagged|seen|identified|present|observed)\b)", re.I)


def denied_findings(answer: str) -> list[tuple[str, str, bool]]:
    """(topic, sentence, strong_only) for each scan topic the answer says is absent: "no level shifts, outliers or missing
    values", "Level shifts / regime changes: None detected". strong_only is True for "no major outliers", which denies only
    major findings. A stance on treatment ("no gaps were filled", "no outliers needing treatment") is not a denial."""
    out: list[tuple[str, str, bool]] = []
    for unit in re.split(r"(?<=[.!?;])\s+|\n+", answer or ""):
        for m in _DENIAL.finditer(unit):
            rest = unit[m.end():m.end() + 80]
            stop = _DENIAL_STOP.search(rest)
            window = rest[:stop.start()] if stop else rest
            if stop and re.match(r"needing|need|requiring", stop.group(0), re.I):
                continue
            after = rest[len(window):]
            for topic, pat in SCAN_TOPICS.items():
                hit = pat.search(window)
                if not hit or _TREATED_AFTER.match(window[hit.end():] + after):
                    continue
                out.append((topic, unit.strip(), bool(_STRONG.match(window))))
        for topic, pat in SCAN_TOPICS.items():
            for hit in pat.finditer(unit):
                if _NONE_AFTER.match(unit[hit.end():]) and not any(t == topic and s == unit.strip() for t, s, _ in out):
                    out.append((topic, unit.strip(), False))
    return out


_COUNT_WORD = r"(?:\d+|one|two|three|four|five|six|seven|eight|nine|ten|several|multiple|many)"
_NO_NEGATION = r"(?:(?!\b(?:none|not|no|never|zero)\b)[^.;\n]){0,30}"
_ACKNOWLEDGED = {
    "level_shift": re.compile(_COUNT_WORD + r"\s+(?:major\s+|minor\s+)?(?:level[- ]shifts?|structural breaks?|breaks?|"
                              r"changepoints?)\b|\b(?:level[- ]shifts?|breaks?)\b" + _NO_NEGATION + r"\b(?:flagged|found|detected|kept)\b",
                              re.I),
    "outliers": re.compile(r"outlier share|\d+(?:\.\d+)?\s*%\s+(?:of\s+\w+\s+)?(?:were\s+)?outliers?|" + _COUNT_WORD +
                           r"\s+outliers?\b|\boutliers?\b" + _NO_NEGATION + r"\b(?:flagged|found|detected|kept|noted)\b", re.I),
    "missing": re.compile(r"\d[\d,]*\s+(?:missing|gaps?|nulls?|holidays?)\b|\b(?:filled|interpolat\w*|imputed)\b", re.I),
}


def acknowledges(answer: str, topic: str, dates: list[str] = ()) -> bool:
    """True when the answer reports the scan's finding on this topic somewhere: "the scan flagged 5 level shifts", "an
    outlier share of 4.4%", "65 missing values were filled", or a break date the scan found. Round 10 (SPX, CO2): an answer
    that listed the five shifts and then wrote "no regime change that would break the seasonal pattern" was blocked as a
    denial; a later phrase about regimes does not hide findings the answer has reported."""
    text = answer or ""
    for unit in re.split(r"(?<=[.!?;])\s+|\n+", text):
        # Round 13 (NGSTORAGE): "No level shifts or regime changes detected." matched "level shifts ... detected" and the
        # lower-case test missed the capital "No", so the denial counted as its own acknowledgement.
        for m in _ACKNOWLEDGED[topic].finditer(unit):
            if not re.search(r"\b(?:no|zero|not|never|without)\s+(?:\w+\s+){0,2}$", unit[:m.start()], re.I):
                return True
    return topic == "level_shift" and any(d and d in text for d in dates)


_SHIFT_DATE = re.compile(r"\b(?:level[- ]shifts?|structural breaks?|breaks?|regime (?:changes?|shifts?)|changepoints?)\b"
                         r"([^.;\n]{0,45})", re.I)
_DATE_IN = re.compile(r"(?<![\d.,])((?:19|20)\d{2})(?:-(\d{2})(?:-(\d{2}))?)?(?![\d,])")


def claimed_shift_dates(answer: str) -> list[tuple[int, str, "date | None"]]:
    """(year, sentence, date or None) for each date the answer places a level shift, break or regime change on: "one level
    shift (2020-03-23)" gives the day, "a break in early 2012" the year only (date None). Only dates within a few words after
    the shift word count, so a data span elsewhere in the sentence does not."""
    from datetime import date
    out: list[tuple[int, str, date | None]] = []
    for unit in sentences(answer):
        # Round 12 (GISTEMP): "Level shifts detected (e.g., post-1970s acceleration)": the dots of "e.g." ended the tail.
        unit = re.sub(r"\b(e\.g|i\.e)\.", lambda m: m.group(1).replace(".", ""), unit)
        for m in _SHIFT_DATE.finditer(unit):
            if negated(unit, m.start()):
                continue
            tail = m.group(1)
            if re.match(r"\s*(?:since|from|between|over|across|during|throughout|beyond)\b", tail, re.I):
                continue
            for d in _DATE_IN.finditer(tail):
                y, mo, dd = int(d.group(1)), d.group(2), d.group(3)
                try:
                    when = date(y, int(mo), int(dd) if dd else 15) if mo else None
                except ValueError:
                    when = None
                out.append((y, unit.strip(), when))
    return out


# "each aligns with known recessions or COVID" (round 11, AUUNEMP) attributes as surely as "due to".
_CAUSE = re.compile(r"\b(?:correspond(?:s|ed)? to|caused by|due to|driven by|because of|reflects?|reflecting|as a result of|"
                    r"align(?:s|ed)? with|coincid(?:es|ed|ing) with|linked to|attributable to|tied to|associated with|"
                    r"explained by|triggered by|(?:are|were|is|was|as)\s+(?:real|genuine|actual)\b[^.;\n]{0,30}?"
                    r"\b(?:episodes?|events?|regimes?|changes?|moves?|shifts?|movements?))\b|"
                    # "kept all 3 level shifts (real regime moves)" (round 13, NVDA)
                    r"\((?:real|genuine|actual)\s[^()]{0,30}?\b(?:episodes?|events?|regimes?|changes?|moves?|shifts?)\b", re.I)
_SHIFT_WORD = re.compile(r"\b(?:level[- ]shifts?|shifts?|breaks?|changepoints?|regimes?|spikes?|outliers?)\b", re.I)
_CITED = re.compile(r"\bS\d{1,3}\b|control_lab|backtest|walk-forward|ignore_ranges|\btested\b", re.I)


# A named event glossed next to a shift: "breaks (Pacific regime shift / satellite era)", "a level shift after the COVID
# lockdown", "outliers during the 2008 recession". Panel round 11 (GISTEMP, AUUNEMP): causes from memory with no cue word.
_EVENT = re.compile(r"\b(?:covid|pandemic|lockdowns?|recessions?|gfc|financial crisis|crisis|wars?|invasion|el ni[nñ]o|"
                    r"la ni[nñ]a|enso|regime shift|satellite era|warming|eruptions?|volcan\w*|tariffs?|stimulus|"
                    r"rate (?:hikes?|cuts?)|elections?|strikes?|mergers?|stock split|earnings|halving|policy change|"
                    r"reopening|shutdowns?|hurricanes?|droughts?|heatwaves?|bushfires?)\b", re.I)
_EVENT_GLOSS = re.compile(r"\([^()]{0,80}\)|\b(?:after|during|following|around|amid|with)\s+(?:the\s+)?[^.;,()]{0,50}", re.I)


def _event_glossed(unit: str) -> bool:
    """Whether a named event sits in a bracket or an after/during phrase within a few words after a shift word."""
    for m in _SHIFT_WORD.finditer(unit):
        if negated(unit, m.start()):
            continue
        tail = unit[m.end(): m.end() + 90]
        for g in _EVENT_GLOSS.finditer(tail):
            # "a regime break (e.g., oil spike, tariff shock) would invalidate the range" names a risk, not a cause.
            if g.start() <= 40 and _EVENT.search(g.group(0)) and not re.match(r"\(\s*(?:e\.g\.|for example|such as)", g.group(0), re.I):
                return True
    return False


_CLAUSE_END = re.compile(r"[.;\n]|\s[—–]\s")
_LEAD_IN = re.compile(r"(?:\b(?:because|since|as|and)\s+(?:they|it|these|those|this)?\s*|\b(?:they|it|these|those)\s+)$", re.I)


def drop_uncited_causes(answer: str) -> str:
    """The answer with each uncited cause taken out of its sentence: a cause clause ("because they reflect real inflation
    episodes", "as genuine regime changes", "corresponds to the 2022 crisis") becomes "(cause not established)", and an
    event gloss in brackets or an after/during phrase next to a shift is removed.

    Motivation: panel round 13 (AUUNEMP): two composer rewrites kept "kept as genuine regime changes", and the run ended
    partial on that phrase alone. Taking the cause out is the fix the check asks for, so it is applied directly once the
    rewrites have failed.
    """
    flagged = set(uncited_causes(answer))
    if not flagged:
        return answer
    out = answer
    for unit in flagged:
        new = unit
        m = _CAUSE.search(new)
        if m and m.group(0).startswith("("):
            close = new.find(")", m.end())
            stop = close + 1 if close >= 0 else m.end()
            new = new[:m.start()] + "(cause not established)" + new[stop:]
        elif m:
            end = _CLAUSE_END.search(new, m.end())
            stop = end.start() if end else len(new)
            lead = _LEAD_IN.search(new[:m.start()])
            start = lead.start() if lead else m.start()
            new = new[:start].rstrip() + " (cause not established)" + new[stop:]
        for sm in list(_SHIFT_WORD.finditer(new))[::-1]:
            tail_start = sm.end()
            for g in list(_EVENT_GLOSS.finditer(new[tail_start:tail_start + 90]))[::-1]:
                if g.start() <= 40 and _EVENT.search(g.group(0)):
                    a, b = tail_start + g.start(), tail_start + g.end()
                    new = new[:a].rstrip() + new[b:]
        out = out.replace(unit, new)
    return out


def uncited_causes(answer: str) -> list[str]:
    """Sentences that attribute a shift, break, spike or outlier to a cause, with no research source id (S1, S2 ...) and no
    test named in the same sentence: by a cue word ("due to", "reflects") or by a named event glossed next to it."""
    out: list[str] = []
    prev = ""
    for u in sentences(answer):
        # Round 12 (CPI): "Four level shifts were found. They were kept because they reflect real inflation episodes." The
        # cause sits in the next sentence, under a pronoun.
        about_shift = _SHIFT_WORD.search(u) or (_SHIFT_WORD.search(prev) and re.match(r"\W*(?:they|these|those|both|all|each|it)\b", u, re.I))
        if about_shift and not _CITED.search(u) and (_CAUSE.search(u) or _event_glossed(u)):
            out.append(u.strip())
        prev = u
    return out


# A method claim names a treatment and a verb of testing. Each treatment has the words a call that runs it would carry in its
# arguments (the code, the options). Only arguments count: a tool's printed output mentions "outlier" in any scan.
_METHOD_CLAIM = re.compile(r"\b(?:tested|backtested|computed|measured)\b", re.I)
_TREATMENTS = (
    (re.compile(r"winsori[sz]|outlier|spike", re.I), ("winsor", "outlier", "replace_outliers", "ignore_ranges", "hampel", "robust")),
    (re.compile(r"first week|exclud|truncat|context", re.I), ("context", "ignore_ranges", "truncat", "exclude", "ctx_")),
    (re.compile(r"changepoint|level[- ]shift|regime|structural break", re.I), ("changepoint", "level_shift", "ignore_ranges",
                                                                             "truncat", "regime")),
    (re.compile(r"\bprophet\b", re.I), ("fbprophet", "prophet import", "prophet(", "prophet.prophet")),
    (re.compile(r"znorm|z-norm|normalis|normaliz", re.I), ("znorm",)),
    (re.compile(r"\bsymmetric", re.I), ("symmetric",)),
    (re.compile(r"\basymmetric", re.I), ("asymmetric",)),
    (re.compile(r"covariate|holiday|driver", re.I), ("covariate", "driver", "holiday")),
)


def _carries(call_text: str, marker: str) -> bool:
    """Whether a call's arguments carry a marker as a word or the start of one. "symmetric" is not carried by "asymmetric"
    (audit round 6l, item 5: a claim of a symmetric loss passed on a call that used the asymmetric loss)."""
    return re.search(r"(?<![a-z0-9])" + re.escape(marker), call_text) is not None


def _call_arguments(history) -> list[str]:
    """The arguments of every call that did not fail, lower-cased, including the calls inside a sub-agent's evidence."""
    out: list[str] = []
    for h in history if isinstance(history, list) else []:
        if not isinstance(h, dict):
            continue
        if h.get("kind") == "tool" and not h.get("error") and "args" in h:
            out.append(_as_text(h.get("args")).lower())
        elif h.get("kind") == "subagent":
            out.extend(_call_arguments(h.get("evidence") or []))
    return out


def unsupported_method_claims(answer: str, history: list) -> list[str]:
    """The units of the answer that say a treatment was tested or computed, where some named treatment has no successful call
    whose arguments carry it. A unit that names no known treatment is not read (its claim cannot be checked)."""
    calls = _call_arguments(history)
    out: list[str] = []
    for unit in sentences(answer):
        m = _METHOD_CLAIM.search(unit)
        if not m or negated(unit, m.start()):
            continue
        for rx, markers in _TREATMENTS:
            if rx.search(unit) and not any(_carries(c, mk) for c in calls for mk in markers):
                out.append(unit.strip())
                break
    return out


_APPLIED = re.compile(r"\b(?:winsori[sz]ed|clipped|capped|trimmed|replaced|removed|dropped)\b[^.;\n]{0,30}?\b(?:outliers?|spikes?)\b|"
                      r"\b(?:outliers?|spikes?)\b[^.;\n]{0,30}?\b(?:were|was|been)\s+(?:winsori[sz]ed|clipped|capped|replaced|removed)\b", re.I)
_APPLIED_MARKERS = ("winsor", "clip(", "replace_outliers", "hampel_replace")
_HYPOTHETICAL = re.compile(r"\b(?:would|could|might|if|not)\b", re.I)
_TRIED = re.compile(r"\b(?:test(?:ed|ing)?|candidates?|control_lab|trial(?:led)?|tried)\b", re.I)


def unapplied_treatments(answer: str, history: list) -> list[str]:
    """Sentences that say outliers were winsorised, clipped, capped or removed, where no analysis code or forecast call
    applied such a treatment (control_lab only tests candidates; it does not change the series).

    Motivation: panel round 13 audit (NVDA): "winsorised the 2 outliers", where the ledger only planned a test, control_lab
    found no evidence for replace_outliers, and the forecast applied ignore_ranges alone.
    """
    calls: list[str] = []
    for h in _flat(history):
        if h.get("kind") != "tool" or h.get("error") or h.get("tool") == "control_lab":
            continue
        args = h.get("args") if isinstance(h.get("args"), dict) else {}
        if h.get("tool") == "sandbox_python" and isinstance(args.get("code"), str):
            # Only the code: a ledger line typed in a string ("will test winsorised copy in control_lab") applies nothing.
            calls.append(re.sub(r"(?s)\"\"\".*?\"\"\"|'''.*?'''|\"[^\"\n]*\"|'[^'\n]*'", "''", args["code"]).lower())
        else:
            calls.append(_as_text(args).lower())
    out = []
    for unit in sentences(answer):
        m = _APPLIED.search(unit)
        if not m or _HYPOTHETICAL.search(unit[max(0, m.start() - 30):m.end()]) or _TRIED.search(unit):
            continue
        if not any(mk in c for c in calls for mk in _APPLIED_MARKERS):
            out.append(unit.strip())
    return out


# Treatment stances (A4-04). A sentence that names a topic and a treatment verb gives the stance; a negation turns it into its
# opposite ("were not clipped" is a keep). The ledger's decisions and the answer are read the same way, so they can be compared.
_OPPOSITE = {"keep": "treat", "treat": "keep", "fill": "none", "none": "fill"}
_TOPIC_STANCES = (
    ("negatives", re.compile(r"negative|below zero", re.I), {
        "keep": re.compile(r"\bkeep|\bkept\b|\bretain|\bgenuine\b", re.I),
        "treat": re.compile(r"\bclip|\bfloor(?:ed)?\b|set to (?:zero|0)|truncat", re.I)}),
    ("outliers", re.compile(r"outlier|spike", re.I), {
        "keep": re.compile(r"\bkeep|\bkept\b|\bretain|real (?:spike|event|value)s?|\bgenuine\b", re.I),
        "treat": re.compile(r"winsori[sz]|\bremov|\bexclud|\breplac|\bcapped\b|\bclipped\b|\bignored\b|\bdropped\b", re.I)}),
    ("gaps", re.compile(r"\bgaps?\b|interpolat|imputed|\bfilled\b", re.I), {
        "fill": re.compile(r"interpolat|\bfilled\b|\bimputed\b|straight[- ]line", re.I),
        "none": re.compile(r"\bno gaps?\b|\bnot (?:filled|interpolated|imputed)\b|\bnot (?:a )?gaps?\b", re.I)}),
)
OPPOSED_STANCES = (("keep", "treat"), ("fill", "none"))


def treatment_stances(text: str) -> dict[str, set[str]]:
    """The stance the text takes on each topic (negatives, outliers, gaps), read sentence by sentence."""
    out: dict[str, set[str]] = {}
    for unit in sentences(text):
        for topic, topic_rx, stances in _TOPIC_STANCES:
            if not topic_rx.search(unit):
                continue
            for stance, rx in stances.items():
                for m in rx.finditer(unit):
                    out.setdefault(topic, set()).add(_OPPOSITE[stance] if negated(unit, m.start()) else stance)
    return out


# Headline totals (A7-03 follow-up). When the pack withholds its total, the answer must not headline one. A sentence that says
# "total" (or "in all", "altogether", "sum of") and carries a figure is a total claim; one that also says why the total is not
# a headline (withheld, floored, below zero, the health verdict) is the explanation, not a claim.
_TOTAL_WORD = re.compile(r"\b(?:total|in all|altogether|sum of|summed)\b", re.I)
_WITHHELD_WORD = re.compile(r"\bwithheld\b|\bwithhold|\bnot (?:a |the )?headline|\bno headline|\bfloored\b|\bbelow zero\b|"
                            r"\bhealth (?:check|verdict)\b|\bverdict\b", re.I)


# "Total" as a category label is not a sum: "ages 15+, total", "both sexes, total", "total (both sexes)", "the total
# unemployment rate". Round 21 (AUUNEMP, 2026-11-27): "ages 15+, total" was read as a headline total and was the only check
# that kept the answer partial.
_TOTAL_LABEL = re.compile(r"\b(?:ages?\s*\d+\s*\+?(?:\s*(?:and over|years))?|both sexes|sexes|sex|persons|all ages)\s*[,:/-]?\s*total\b"
                          r"|\btotal\s*\((?:both sexes|all ages|persons|sexes)[^)]*\)"
                          r"|\btotal\s+(?:unemployment|employment|participation|inflation|labou?r force)?\s*rates?\b", re.I)


def claimed_totals(answer: str) -> list[str]:
    """Sentences that state a total with a figure and do not explain why a total is withheld."""
    return [u for u in sentences(answer)
            if _TOTAL_WORD.search(_TOTAL_LABEL.sub(" ", u)) and figures(u) and not _WITHHELD_WORD.search(u)]


def explains_withheld(answer: str) -> bool:
    """Whether the answer says anywhere why the total is not a headline."""
    return bool(_WITHHELD_WORD.search(answer or ""))


# Words that name what a total's bound assumes. A sentence that gives a bound without them presents it as the range.
_TOGETHER_WORDS = re.compile(r"together|same way|same direction|at once|cautious|widest|worst|all miss|in step|compound|"
                             r"(?<!un)correlat|accumulat|in the same", re.I)
# Live electricity rep15 (2026-11-23): "if clients' errors cancel out" names the independent bound correctly and was blocked.
_INDEPENDENT_WORDS = re.compile(r"independen|cancel|offset|average out|uncorrelated|unrelated|separately|on their own|"
                                r"do not move together|don't move together|not together", re.I)
_WORD_SCALE = re.compile(r"^\s*(million|mn|m\b|thousand|k\b|billion|bn)", re.I)
_WORD_SCALES = {"million": 1e6, "mn": 1e6, "m": 1e6, "thousand": 1e3, "k": 1e3, "billion": 1e9, "bn": 1e9}


def _sentence_values(sentence: str) -> list[float]:
    """Every figure in a sentence as a value, with a written scale applied ("£1.26 million" is 1,260,000)."""
    out = []
    cleaned = _cleaned(sentence)
    for m in _NUM.finditer(cleaned):
        tok = m.group().rstrip(",")
        try:
            v, _, pct = _parse(tok)
        except ValueError:
            continue
        if pct:
            continue
        scale = _scale_of(tok)
        word = _WORD_SCALE.match(cleaned[m.end():m.end() + 12])
        if scale == 1.0 and word:
            scale = _WORD_SCALES[word.group(1).lower()]
        out.append(v * scale)
    return out


def bound_misnamed(answer: str, lo: float, hi: float, together: bool, others: tuple = ()) -> str | None:
    """The first sentence that states both edges of a total's bound (within 1 percent) without naming what the bound assumes:
    "all miss the same way at once" for the together bound, "independently" for the other. None when no sentence does.

    others: other (lo, hi) ranges of the same total (the measured range, the other bound). A stated figure counts for this
    bound only when it is at least as close to it as to the matching edge of every other range. Panel SPX (2026-11-26): a
    20-day total of an index level made the ranges differ by under 1 percent, and the measured range 132,064 to 145,108 was
    read as the together bound 130,785 to 146,388, which blocked a correct answer."""
    if not (lo and hi) or hi <= lo:
        return None
    need = _TOGETHER_WORDS if together else _INDEPENDENT_WORDS
    alts = {lo: [o[0] for o in others if o and o[0]], hi: [o[1] for o in others if o and o[1]]}
    prev = ""
    for s in sentences(answer):
        vals = _sentence_values(s)
        hit = lambda t: any(abs(v - t) <= 0.01 * abs(t) and all(abs(v - t) <= abs(v - a) for a in alts[t]) for v in vals)
        # "under that assumption" refers back to the assumption named in the sentence directly before it (stored 139db236).
        names = need.search(s) or (_ASSUMPTION_BACK.search(s) and need.search(prev))
        if hit(lo) and hit(hi) and not names:
            return s
        prev = s
    return None


_ASSUMPTION_BACK = re.compile(r"\b(?:that|this) assumption\b", re.I)


# ---------------------------------------------------------------------------------------------- entity-bound table rows --
# Panel round 8 (NBA, independent audit, 2026-11-27): the standings table gave Boston Celtics 13-5, .722 and a median of 44;
# the tools gave 12-6, .667 and 43. Every one of those numbers was printed somewhere else in the evidence, so the bag-of-numbers
# check above passed 18 wrong figures. A number in a table row that names an entity is a claim about that entity, so it is
# checked against that entity's own records: the tool objects that carry its name, and the printed lines that name it.
_CELL_NUM = re.compile(r"(?<![\w.])\d[\d,]*(?:\.\d+)?%?|(?<![\w\d])\.\d+%?")
_ENTITY = re.compile(r"[A-Za-z][A-Za-z'&.\- ]{2,58}[A-Za-z.]")
_SEPARATOR_ROW = re.compile(r"^\s*\|?\s*:?-{2,}")


def _numbers_in(obj, out: list[float]) -> None:
    if isinstance(obj, bool):
        return
    if isinstance(obj, (int, float)):
        out.append(float(obj))
    elif isinstance(obj, dict):
        for v in obj.values():
            _numbers_in(v, out)
    elif isinstance(obj, list):
        for v in obj:
            _numbers_in(v, out)
    elif isinstance(obj, str) and len(obj) < 200:
        for m in _CELL_NUM.finditer(obj):
            try:
                out.append(float(m.group().rstrip("%").replace(",", "")))
            except ValueError:
                pass


def _tool_objects(history: list):
    """The raw tool results (and full results) the run observed, including a sub-agent's own tool evidence."""
    for h in history:
        if not isinstance(h, dict):
            continue
        if h.get("kind") == "tool" and "result" in h:
            yield h["result"]
            if "full" in h:
                yield h["full"]
        elif h.get("kind") == "subagent":
            yield from _tool_objects(h.get("evidence") or [])


def _scalars_in(obj, out: list[float]) -> None:
    """The numbers of an object's own fields and nested objects, not of its lists (a path or a table of games)."""
    if isinstance(obj, dict):
        for v in obj.values():
            if isinstance(v, dict):
                _scalars_in(v, out)
            elif not isinstance(v, list):
                _numbers_in(v, out)


# Fields that name a real-world entity. A method or model name ("TimesFM", "seasonal_naive") is not one: its comparison
# figures live in other objects (round 5 CO2: "29% more accurate" is a verdict's gain_pct), so its rows are left to the
# bag-of-numbers check. Printed lines and keyed lists add to an entity's records but never make one on their own.
_NAME_FIELDS = frozenset({"team", "name", "entity", "club", "country", "region", "ticker", "symbol", "player", "series",
                          "home", "away", "company", "state", "city", "station", "site"})
_PATH_HEADER = re.compile(r"\b(?:after|game|games|round|rounds|week|weeks|matchday|gw|path|step|month)\b", re.I)


def _norm_name(text: str) -> str:
    """A name as compared: lower case, underscores and runs of spaces as one space ("Seasonal naive" is seasonal_naive)."""
    return re.sub(r"[_\s]+", " ", text).strip().lower()


def _entity_records(name: str, history: list) -> tuple[list[float], list[float]] | None:
    """(scalars, series) recorded about one entity, or None when the evidence holds no record of it. Scalars: the fields of
    a tool object that carries its name, and the printed lines that name it. Series: the numbers of a
    list kept under its name (a points path), read only for a table about a path."""
    want = _norm_name(name)
    scalars: list[float] = []
    series: list[float] = []
    lines: list[float] = []
    hit = False

    def walk(obj) -> None:
        nonlocal hit
        if isinstance(obj, dict):
            if any(isinstance(v, str) and _norm_name(v) == want and str(k).lower() in _NAME_FIELDS for k, v in obj.items()):
                hit = True
                _scalars_in(obj, scalars)
            for k, v in obj.items():
                if isinstance(k, str) and _norm_name(k) == want:
                    if isinstance(v, list):
                        for item in v:
                            if isinstance(item, dict):
                                _numbers_in({x: y for x, y in item.items() if x not in ("game", "step", "round")}, series)
                            else:
                                _numbers_in(item, series)
                    elif isinstance(v, dict):
                        _scalars_in(v, scalars)
                    else:
                        _numbers_in(v, scalars)
                walk(v)
        elif isinstance(obj, list):
            for v in obj:
                walk(v)
        elif isinstance(obj, str) and want in _norm_name(obj):
            for line in obj.splitlines():
                if want in _norm_name(line):
                    for m in _CELL_NUM.finditer(line):
                        try:
                            lines.append(float(m.group().rstrip("%").replace(",", "")))
                        except ValueError:
                            pass
    for obj in _tool_objects(history):
        walk(obj)
    if not hit:
        return None
    return scalars + lines, series


_DOMAIN = re.compile(r"\b((?:[a-z0-9-]+\.)+(?:gov|com|org|net|edu|int|io|info|au|uk|jp|de|fr|ca|eu))\b", re.I)


# A citation: the domain, then its date, "latest" or "undated" ("(bls.gov, undated)", "**proactiveinvestors.com, 2026-11-23**").
_CITED_AS = re.compile(r"\s*[,;:\u2013\u2014-]\s*(?:undated|latest|accessed|retrieved|\d{1,4}\b|(?:jan|feb|mar|apr|may|jun|jul|aug|"
                       r"sep|oct|nov|dec)[a-z]*\b)", re.I)


def _host(url: str) -> str:
    m = re.match(r"^[a-z]+://([^/:?#]+)", str(url or "").strip(), re.I)
    return m.group(1).lower() if m else ""


def _source_texts(history: list) -> dict[str, list[str]]:
    """What each fetched host returned, as text: research passages and fetched pages, by host name."""
    out: dict[str, list[str]] = {}

    def add(url: str, text: str) -> None:
        if text:                         # a passage with no address is kept under "", read when a line cites no site
            out.setdefault(_host(url), []).append(text)

    for h in history:
        if h.get("kind") == "subagent":
            for host, texts in _source_texts(h.get("evidence") or []).items():
                out.setdefault(host, []).extend(texts)
            continue
        if h.get("kind") != "tool":
            continue
        for res in (h.get("full"), h.get("result")):
            if not isinstance(res, dict):
                continue
            for src in res.get("sources") or []:
                if isinstance(src, dict):
                    add(src.get("final_url") or src.get("url") or "",
                        " ".join(str(p.get("text", "")) for p in src.get("passages") or [] if isinstance(p, dict)))
            if isinstance(res.get("text"), str):
                add(res.get("final_url") or res.get("url") or "", res["text"])
            elif res.get("final_url") or res.get("url"):      # a download: its row count, columns and first rows
                add(res.get("final_url") or res.get("url") or "", _as_text(res))
    return out


# A date in words or numbers ("20 Nov 2026", "Nov 20, 2026", "2026-11-20"): its digits are not a value.
DATE_WORDS = re.compile(r"\b\d{4}-\d{2}-\d{2}\b|\b\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?"
                        r"(?:\s+\d{4})?|\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2}(?:,?\s+\d{4})?",
                        re.I)
_QUOTED = re.compile(r"[\"\u201c\u201d]([^\"\u201c\u201d\n]{8,240})[\"\u201c\u201d]")


def _fold(text: str) -> str:
    """Text as quotes are compared: accents, dashes, quote marks and spacing folded, lower case."""
    import unicodedata
    t = unicodedata.normalize("NFKD", text or "").encode("ascii", "ignore").decode()
    t = re.sub(r"[\u2010-\u2015\-]+", "-", t)
    t = re.sub(r"[\"'`]", "", t)
    return re.sub(r"\s+", " ", t).strip().lower()


def quote_found(line: str, texts: dict[str, list[str]]) -> bool:
    """True when the line holds a quote (in double quotes, 8 characters or more) that appears in the text of the site the
    line cites, or of any fetched site when the line cites none."""
    quotes = [_fold(q) for q in _QUOTED.findall(line or "")]
    if not quotes:
        return False
    domains = {m.group(1).lower() for m in _DOMAIN.finditer(line)}
    hosts = [h for h in texts if any(h == d or h.endswith("." + d) or d.endswith("." + h) for d in domains)] or list(texts)
    pool = _fold(" ".join(t for h in hosts for t in texts[h]))
    return any(q and q in pool for q in quotes)


def source_bound_mismatches(answer: str, history: list) -> list[str]:
    """Figures in a line that names a source domain (such as "(bls.gov, 13 Nov 2026)") that the text fetched from that
    domain does not hold. A domain nothing was fetched from supports no figure.

    Motivation: panel round 10 audit. CPI cited "philadelphiafed.org" for "2.5-2.7%" and "fred.stlouisfed.org" for a 2.3%
    expectation; neither passage held a value, and both figures passed because the numbers were printed elsewhere. Round 9:
    NEM gave Queensland's price as New South Wales's; GISTEMP put a JMA figure under NOAA. A figure stated as a source's is
    checked against that source.
    Logic: lines are split; a line's domains are the domain names cited in it (followed by a date, "latest" or "undated"),
    as outside context is cited; a domain named in passing is not a citation. Its figures (years
    excepted) must be near a number in the texts of a fetched host equal to, or ending with, one of those domains.
    """
    texts = _source_texts(history)
    if not texts:
        return []
    missing: list[str] = []
    for line in re.split(r"\n+", answer or ""):
        domains = {m.group(1).lower() for m in _DOMAIN.finditer(line) if _CITED_AS.match(line[m.end():])}
        if not domains:
            continue
        pool_text = " ".join(t for host, ts in texts.items() for t in ts
                             if any(host == d or host.endswith("." + d) or d.endswith("." + host) for d in domains))
        pool = []
        for t in figures(pool_text, references=False):
            v, _, _ = _parse(t)
            pool.append(abs(v) * _scale_of(t))
        # Addresses, code spans and dates name things; their digits are not figures ("ABS,LF,1.0.0", "2028-01-25/26").
        body = re.sub(r"https?://\S+|`[^`]*`|\b\d{4}-\d{2}-\d{2}(?:/\d{1,2})?\b", " ", line)
        body = _DOMAIN.sub(" ", body)
        for token, _approx in _figures_in_context(body):
            if _YEAR.fullmatch(token):
                continue
            value, decimals, pct = _parse(token)
            value, scale = abs(value), _scale_of(token)
            cands = [value * scale] + ([value / 100.0] if pct else [])
            if any(_near(v, c, decimals + (2 if c != value * scale else 0), scale if c == value * scale else 1.0)
                   for v in pool for c in cands):
                continue
            if token not in missing:
                missing.append(token)
    return missing


def entity_row_mismatches(answer: str, history: list) -> list[str]:
    """Numbers in answer table rows that do not match the records of the entity the row names. A row whose entity has no
    record in the evidence is not judged here (the bag-of-numbers check still covers it)."""
    lines = (answer or "").splitlines()
    cache: dict[str, tuple[list[float], list[float]] | None] = {}
    out: list[str] = []
    header = ""
    for i, line in enumerate(lines):
        if not line.lstrip().startswith("|"):
            header = ""
            continue
        if _SEPARATOR_ROW.match(line):
            continue
        if i + 1 < len(lines) and _SEPARATOR_ROW.match(lines[i + 1]):
            header = line                                     # a header row names the columns of the rows below it
            continue
        cells = [c.strip().strip("*").strip() for c in line.strip().strip("|").split("|")]
        at = next((j for j, c in enumerate(cells) if _ENTITY.fullmatch(c) and not _YEAR.search(c)), None)
        if at is None:
            continue
        name = cells[at]
        if name.lower() not in cache:
            cache[name.lower()] = _entity_records(name, history)
        entry = cache[name.lower()]
        if entry is None:
            continue
        records = entry[0] + (entry[1] if _PATH_HEADER.search(header) else [])
        if not records:
            continue
        for cell in cells[at + 1:]:
            for m in _CELL_NUM.finditer(cell.replace("\u2011", "-")):
                token = m.group().rstrip(",")
                value, decimals, pct = _parse(token if not token.startswith(".") else "0" + token)
                options = [value] + ([value / 100.0] if pct else [])
                if not any(_near(r, o, decimals + (2 if (pct and o != value) else 0)) for r in records for o in options):
                    if token not in out:
                        out.append(token)
    return out
