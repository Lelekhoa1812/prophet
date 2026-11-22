"""Split a page into quotable passages, score them against the research queries, and verify quotes against the page.

Motivation vs Logic
Motivation: a whole web page is too long to put in front of the planner, and a summary written from memory is not evidence.
The agent needs the few passages that answer its questions, with their position in the page, and a deterministic way to
check that a quotation really appears in the text it cites.
Logic: passages are whole paragraphs packed up to a size, with a short paragraph repeated as overlap; an oversized
paragraph is cut into windows. Scoring is a small BM25 variant with the document set being the passages of this page,
stop words removed, and numbers given a bonus because a figure is usually what the question is after. Quote checking
compares text after collapsing whitespace and typographic quotes, and is otherwise exact, so a quotation is only
accepted when it is really in the page.
"""
from __future__ import annotations

import math
import re
import unicodedata
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
