"""Split a page into quotable passages, score them against the research queries, and verify quotes against the page.

Motivation vs Logic
Motivation: a whole web page is too long to put in front of the planner, and a summary written from memory is not evidence.
The agent needs the few passages that answer its questions, with their position in the page, and a deterministic way to
check that a quotation really appears in the text it cites.
Logic: passages are whole paragraphs packed up to a size, with a short paragraph repeated as overlap; an oversized
paragraph is cut into windows. Scoring is a small BM25 variant with the document set being the passages of this page,
stop words removed, and numbers given a bonus because a figure is usually what the question is after. Quote checking
compares text after collapsing whitespace and typographic quotes, and is otherwise exact, so a quotation is only
accepted when it is really in the page. Figure checking uses the same normalisation: every number in a
draft must appear in some fetched passage, or it is returned as unverified.
"""
from __future__ import annotations

import math
import re
import unicodedata
from decimal import Decimal, InvalidOperation
from dataclasses import dataclass

STOP_WORDS = frozenset("""
a an and are as at be been but by can could did do does for from had has have how in into is it its of on or
over that the their them then there these they this to was were what when where which who why will with would
about after also any more most not no than too very via vs we our you your
""".split())
TOKEN = re.compile(r"[a-z]+|\d+(?:[.,]\d+)*")
PARAGRAPH = re.compile(r"[^\n]+(?:\n(?!\s*\n)[^\n]+)*")
K1, B = 1.2, 0.75
NUMBER_BONUS = 0.4
NUMBER_BONUS_CAP = 1.6
NUMBER_QUERY_WEIGHT = 1.5


@dataclass(frozen=True)
class Passage:
    text: str
    start: int


def _windows(text: str, start: int, end: int, size: int, overlap: int) -> list[tuple[int, int]]:
    """Overlapping windows over one oversized paragraph, each ending at a space where one is near the limit."""
    spans = []
    pos = start
    while True:
        stop = min(pos + size, end)
        if stop < end:
            cut = text.rfind(" ", pos + size // 2, stop)
            if cut > pos:
                stop = cut
        spans.append((pos, stop))
        if stop >= end:
            break
        nxt = stop - overlap
        pos = nxt if nxt > pos else stop
    return spans


def split_passages(text: str, size: int = 700, overlap: int = 100) -> list[Passage]:
    """Paragraph-aware passages of at most about ``size`` characters. A paragraph is never split unless it alone is
    longer than ``size``; then it is cut into overlapping windows."""
    if size < 50:
        raise ValueError("size must be at least 50 characters")
    if not 0 <= overlap < size:
        raise ValueError("overlap must be at least 0 and smaller than size")
    blocks = [(m.start(), m.end()) for m in PARAGRAPH.finditer(text) if text[m.start():m.end()].strip()]
    spans: list[tuple[int, int]] = []
    current: tuple[int, int] | None = None
    previous: tuple[int, int] | None = None
    for start, end in blocks:
        if end - start > size:
            if current:
                spans.append(current)
                current = None
            spans.extend(_windows(text, start, end, size, overlap))
            previous = None
            continue
        if current is None:
            current = (start, end)
        elif end - current[0] > size:
            spans.append(current)
            # repeat the previous paragraph as overlap when it is short enough
            if previous is not None and previous[1] - previous[0] <= overlap:
                current = (previous[0], end)
            else:
                current = (start, end)
        else:
            current = (current[0], end)
        previous = (start, end)
    if current:
        spans.append(current)
    passages = []
    for start, end in spans:
        chunk = text[start:end].strip()
        if chunk:
            lead = len(text[start:end]) - len(text[start:end].lstrip())
            passages.append(Passage(text=chunk, start=start + lead))
    return passages


def tokens(text: str) -> list[str]:
    """Lower-case word and number tokens with stop words removed. Typographic apostrophes are normalised first."""
    folded = unicodedata.normalize("NFKC", text).lower().replace("’", "'")
    return [t for t in TOKEN.findall(folded) if t not in STOP_WORDS and (len(t) > 1 or t.isdigit())]


def _is_number(token: str) -> bool:
    return any(ch.isdigit() for ch in token)


def score_passages(passages: list[Passage], queries: list[str]) -> list[float]:
    """BM25-lite score per passage, in the same order. Inverse document frequency is taken over these passages."""
    query_terms: dict[str, float] = {}
    for query in queries:
        for term in tokens(query):
            query_terms[term] = NUMBER_QUERY_WEIGHT if _is_number(term) else 1.0
    if not passages or not query_terms:
        return [0.0] * len(passages)
    docs = [tokens(p.text) for p in passages]
    n = len(docs)
    avg_len = max(sum(len(d) for d in docs) / n, 1.0)
    df = {term: sum(1 for d in docs if term in d) for term in query_terms}
    scores = []
    for doc in docs:
        length = max(len(doc), 1)
        total = 0.0
        for term, weight in query_terms.items():
            tf = doc.count(term)
            if not tf:
                continue
            idf = math.log(1 + (n - df[term] + 0.5) / (df[term] + 0.5))
            total += weight * idf * tf * (K1 + 1) / (tf + K1 * (1 - B + B * length / avg_len))
        numbers = {t for t in doc if _is_number(t)}
        if total > 0 and numbers:
            total += min(NUMBER_BONUS * len(numbers), NUMBER_BONUS_CAP)
        scores.append(round(total, 4))
    return scores


def best_passages(page, queries: list[str], k: int = 4, *, size: int = 700, overlap: int = 100) -> list[dict]:
    """The k best-scoring, non-overlapping passages of a page (a Page or a string) as [{text, score, start}].
    Passages that match no query are not returned."""
    text = page if isinstance(page, str) else getattr(page, "text", "") or ""
    if not text or not queries or k < 1:
        return []
    passages = split_passages(text, size=size, overlap=overlap)
    scores = score_passages(passages, queries)
    ranked = sorted(zip(passages, scores), key=lambda pair: (-pair[1], pair[0].start))
    chosen: list[dict] = []
    taken: list[tuple[int, int]] = []
    for passage, score in ranked:
        if score <= 0 or len(chosen) >= k:
            break
        end = passage.start + len(passage.text)
        if any(passage.start < b and a < end for a, b in taken):
            continue
        taken.append((passage.start, end))
        chosen.append({"text": passage.text, "score": score, "start": passage.start})
    return chosen


def _normalise_for_quote(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    for src, dst in (("‘", "'"), ("’", "'"), ("“", '"'), ("”", '"'), ("–", "-"),
                     ("—", "-"), (" ", " ")):
        text = text.replace(src, dst)
    return " ".join(text.split())


def quote_in_page(quote: str, page_text: str) -> bool:
    """True when the quotation appears in the page text after collapsing whitespace and typographic quotes.
    Case and punctuation are kept, so a changed word or figure is caught."""
    needle = _normalise_for_quote(quote or "")
    if not needle:
        return False
    return needle in _normalise_for_quote(page_text or "")


NUMBER = re.compile(r"(?<![\w/])\d+(?:[.,]\d+)*(?!\w)")
THOUSANDS = re.compile(r"\d{1,3}(?:,\d{3})+(?:\.\d+)?")


def _number_key(token: str) -> str:
    """A number token in one form: '1,234' and '1234' match, '78.50' and '78.5' match, '2,5' is read as 2.5."""
    if THOUSANDS.fullmatch(token):
        token = token.replace(",", "")
    else:
        token = token.replace(",", ".")  # a lone comma is a decimal mark
    try:
        return f"{Decimal(token).normalize():f}"
    except InvalidOperation:
        return token


def _checkable(token: str) -> bool:
    """Years 1900 to 2100 and whole numbers under 10 are left out: they are dates and counts, not claims to check."""
    if "." in token or "," in token:
        return True
    value = int(token)
    return not (1900 <= value <= 2100 or value < 10)


def verify_figures(text: str, sources: list[dict]) -> list[str]:
    """Numbers in ``text`` that appear in none of the sources' passages or text, in order of appearance, once each.
    Years 1900 to 2100 and whole numbers under 10 are ignored. Sources are dicts with ``passages`` ([{text}]) and/or
    ``text``; the same normalisation as quote_in_page is applied to both sides. An empty list means every figure
    was found in the evidence (it does not check that the figure is attached to the right claim)."""
    known: set[str] = set()
    for source in sources or []:
        if isinstance(source, str):
            chunks = [source]
        elif isinstance(source, dict):
            chunks = [source["text"]] if isinstance(source.get("text"), str) else []
            chunks += [p["text"] for p in source.get("passages") or []
                       if isinstance(p, dict) and isinstance(p.get("text"), str)]
        else:
            continue
        for chunk in chunks:
            known.update(_number_key(t) for t in NUMBER.findall(_normalise_for_quote(chunk)))
    missing: list[str] = []
    for token in NUMBER.findall(_normalise_for_quote(text or "")):
        if not _checkable(token):
            continue
        if _number_key(token) not in known and token not in missing:
            missing.append(token)
    return missing


# Calendar dates in running text (A3-07). A dated sentence is kept as a passage even when no query word appears, so a
# dated event such as "OPEC+ meeting on 29 November 2026" is not lost (round 3: the page scored 0 and was dropped).
MONTHS = {"jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6, "jul": 7, "aug": 8, "sep": 9, "oct": 10,
          "nov": 11, "dec": 12}
_MONTH_WORD = r"(?:january|february|march|april|may|june|july|august|september|october|november|december|" \
              r"jan|feb|mar|apr|jun|jul|aug|sept|sep|oct|nov|dec)"
DATE_EXPR = re.compile(rf"\b(?:\d{{1,2}}\s+{_MONTH_WORD}(?:\s+\d{{4}})?|{_MONTH_WORD}\s+\d{{1,2}}(?:,?\s+\d{{4}})?|"
                       rf"{_MONTH_WORD}\s+\d{{4}}|\d{{4}}-\d{{2}}-\d{{2}})\b", re.I)
STATED = re.compile(rf"\b(?:published|released|updated|posted|issued|as of)\s+(?:on\s+)?(\d{{1,2}}\s+{_MONTH_WORD}"
                    rf"(?:\s+\d{{4}})?|{_MONTH_WORD}\s+\d{{1,2}}(?:,?\s+\d{{4}})?)", re.I)
HEAD_CHARS = 320
DATE_SCORE = 0.1


def parse_date_expr(expr: str, default_year: int) -> tuple[str, bool] | None:
    """(ISO date, year was written in the text) for an expression from DATE_EXPR or STATED; None when unreadable.
    A missing year takes default_year and is reported as not written, so the caller can label it as inferred."""
    value = expr.strip().lower()
    iso = re.fullmatch(r"(\d{4})-(\d{2})-(\d{2})", value)
    if iso:
        return f"{iso.group(1)}-{iso.group(2)}-{iso.group(3)}", True
    day_first = re.fullmatch(r"(\d{1,2})\s+([a-z]+)(?:\s+(\d{4}))?", value)
    month_first = re.fullmatch(r"([a-z]+)\s+(\d{1,2})(?:,?\s+(\d{4}))?", value)
    if day_first:
        day, month_word, year = day_first.groups()
    elif month_first:
        month_word, day, year = month_first.groups()
    else:
        month_only = re.fullmatch(r"([a-z]+)\s+(\d{4})", value)
        if not month_only:
            return None
        month_word, day, year = month_only.group(1), "1", month_only.group(2)
    month = MONTHS.get(month_word[:3])
    if month is None or not 1 <= int(day) <= 31:
        return None
    try:
        return f"{int(year or default_year):04d}-{month:02d}-{int(day):02d}", year is not None
    except ValueError:
        return None


def stated_date(text: str, default_year: int) -> tuple[str, bool] | None:
    """The first date a page states about its own publication ("released November 10"), read with parse_date_expr."""
    match = STATED.search(text or "")
    return parse_date_expr(match.group(1), default_year) if match else None


def has_date(text: str) -> bool:
    return bool(DATE_EXPR.search(text or ""))


def select_passages(page, queries: list[str], k: int = 4, *, size: int = 700, overlap: int = 100) -> list[dict]:
    """The passages to quote from a page (A3-07). Query matches come first, best score first. Then passages that carry
    a calendar date (reason "date"), even with no query word. When nothing is chosen, the opening of the page is given
    (reason "head") so a title-only or unmatched page is still visible. Each item has text, score, start and reason."""
    text = page if isinstance(page, str) else getattr(page, "text", "") or ""
    if not text or k < 1:
        return []
    passages = split_passages(text, size=size, overlap=overlap)
    scores = score_passages(passages, queries) if queries else [0.0] * len(passages)
    candidates = [(-score, passage.start, "query", score, passage) for passage, score in zip(passages, scores)
                  if score > 0]
    candidates += [(0.0, passage.start, "date", DATE_SCORE, passage) for passage, score in zip(passages, scores)
                   if score <= 0 and has_date(passage.text)]
    candidates.sort(key=lambda c: (c[0], c[1]))
    chosen: list[dict] = []
    taken: list[tuple[int, int]] = []
    for _, _, reason, score, passage in candidates:
        if len(chosen) >= k:
            break
        end = passage.start + len(passage.text)
        if any(passage.start < b and a < end for a, b in taken):
            continue
        taken.append((passage.start, end))
        chosen.append({"text": passage.text, "score": score, "start": passage.start, "reason": reason})
    if not chosen:
        head = text[:HEAD_CHARS].strip()
        if head:
            chosen.append({"text": head, "score": 0.0, "start": 0, "reason": "head"})
    return chosen
