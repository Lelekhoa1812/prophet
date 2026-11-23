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
#
# A list separator is a comma followed by a space, so a thousands group ("customers 1,200") is a measured value, not a list, and
# it is not blanked. A number that continues into a decimal or a thousands group is not an identifier either (probe: "Customers
# 1,200 bought" lost its figure and "Store 3.5" left ".5" behind before this).
_IDENTIFIER = re.compile(
    rf"\b{_LABEL_NOUN}s?\s*#?\s*\d{{1,4}}(?:\s*(?:,\s+|-|\u2013|\u2014|to|through)\s*\d{{1,4}})*\b(?![.,]\d)", re.I)
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


def _near(value: float, target: float, decimals: int, scale: float = 1.0) -> bool:
    return abs(value - target) <= 0.5 * 10 ** -decimals * scale + 1e-12


def _as_text(obj) -> str:
    """Evidence as text for matching. Unlike ``canonical`` (compact, used for idempotency keys), the separators are
    spaced: compact JSON reads ``[360,365,370,375]`` as one thousands-separated number, so no figure in it would match."""
    return json.dumps(obj, sort_keys=True, separators=(", ", ": "), ensure_ascii=False, default=str)


def _observed(history: list):
    """What the run observed: tool results, the person's answers, and a sub-agent's own tool evidence. Arguments,
    rationale, the planner's words and a sub-agent's written answer are never yielded: the written answer is a claim
    made by the model, so a figure in it is checked against tool results, not accepted because the model wrote it
    (audit 2026-11-22, item 1)."""
    for h in history:
        kind = h.get("kind")
        if kind == "tool" and "result" in h:
            yield _as_text(h["result"])
            if "full" in h:              # what the tool really returned, before its summary cut it (round 4, live panels)
                yield _as_text(h["full"])
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
    for t in figures(evidence_text(history, goal)):
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
    return missing


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
    for t in figures(evidence_text(history, goal)):
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
    re.compile(r"\b(?:weekly|cycle|season|seasonal|period|periodic)\s{0,2}(?:\(|of\s+|:)?\s{0,2}(\d{1,3})[- ](?:day|step|period)?\b", re.I),
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


def claimed_periods(answer: str) -> list[tuple[int, str]]:
    """The cycle lengths, in steps, that the answer states: "weekly (7-day) cycle" gives 7. Negated statements and lag
    comparisons ("the same period 260 days earlier") are left out."""
    out: list[tuple[int, str]] = []
    for unit in sentences(answer):
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
_SHIFT_COUNT = re.compile(r"\b(\d{1,2}|one|two|three|four|five|six|seven|eight|nine|ten)\s+"
                          r"(?:level[- ]shifts?|regime shifts?|structural breaks?|breaks?|changepoints?)\b", re.I)


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


_CAUSE = re.compile(r"\b(?:correspond(?:s|ed)? to|caused by|due to|driven by|because of|reflects?|reflecting|as a result of)\b", re.I)
_SHIFT_WORD = re.compile(r"\b(?:level[- ]shifts?|shifts?|breaks?|changepoints?|regimes?|spikes?|outliers?)\b", re.I)
_CITED = re.compile(r"\bS\d{1,3}\b|control_lab|backtest|walk-forward|ignore_ranges|\btested\b", re.I)


def uncited_causes(answer: str) -> list[str]:
    """Sentences that attribute a shift, break, spike or outlier to a cause, with no research source id (S1, S2 ...) and no
    test named in the same sentence."""
    return [u.strip() for u in sentences(answer) if _CAUSE.search(u) and _SHIFT_WORD.search(u) and not _CITED.search(u)]


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
    (re.compile(r"symmetric", re.I), ("symmetric",)),
    (re.compile(r"covariate|holiday|driver", re.I), ("covariate", "driver", "holiday")),
)


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
            if rx.search(unit) and not any(mk in c for c in calls for mk in markers):
                out.append(unit.strip())
                break
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


def claimed_totals(answer: str) -> list[str]:
    """Sentences that state a total with a figure and do not explain why a total is withheld."""
    return [u for u in sentences(answer) if _TOTAL_WORD.search(u) and figures(u) and not _WITHHELD_WORD.search(u)]


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


def bound_misnamed(answer: str, lo: float, hi: float, together: bool) -> str | None:
    """The first sentence that states both edges of a total's bound (within 1 percent) without naming what the bound assumes:
    "all miss the same way at once" for the together bound, "independently" for the other. None when no sentence does."""
    if not (lo and hi) or hi <= lo:
        return None
    need = _TOGETHER_WORDS if together else _INDEPENDENT_WORDS
    for s in sentences(answer):
        vals = _sentence_values(s)
        hit = lambda t: any(abs(v - t) <= 0.01 * abs(t) for v in vals)
        if hit(lo) and hit(hi) and not need.search(s):
            return s
    return None
