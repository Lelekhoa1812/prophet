"""Deterministic checks of the claims a researcher sub-agent reports: is the quote in the page the mission stored, and does the value
sit inside it.

Motivation vs Logic
Motivation: a 2026 study found that 81% of unsupported numeric claims carried the right value from the wrong source. A model
verifier accepted 36% of claims it could not trace. The researcher role promises that every claim has a source id, a date and a
quote that appears in the fetched text, and nothing checked that promise (quote_in_page was never called). A figure that is in
the page but attached to the wrong entity, or a quote the page no longer holds, must be caught in code, not by another model.
Logic: a claim is a record (value, unit, as-of date, entity, source url, source id, quote, locator, snapshot sha256, extraction
method). The status (verified, unverified, ambiguous) is set only by verify_claim, never by the report. Verification is literal
and whitespace-normalised: the quote must be in the stored page text, the value literal must be inside the quote, a value that the
page prints more than once needs a locator next to the quote, and a page whose sha256 differs from the snapshot the sub-agent read
is unverified. No model is called. The lead reads the result through the ledger step that stores it, so a resumed run reads the
same statuses.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass, replace
from typing import Any

from prophet_agent.research.fetch import stored_page
from prophet_agent.research.passages import _normalise_for_quote, quote_in_page

STATUSES = ("verified", "unverified", "ambiguous")
MAX_CLAIMS = 30            # claims read from one report; a longer list is cut and the cut is not checked silently
LOCATOR_WINDOW = 400       # characters before a quote in which a locator must appear
CLAIM_FIELDS = ("value", "unit", "as_of", "entity", "source_id", "url", "quote", "locator")

CLAIM_LINE = re.compile(r"^\s*(?:[-*]\s*)?(?:\*\*)?CLAIM(?:\*\*)?\s*[|:]\s*(.*?)\s*$", re.I)
PAIR_SPLIT = re.compile(r"\s*\|\s*(?=[a-z_]+\s*=)", re.I)
KEY_VALUE = re.compile(r"^\s*([a-z_]+)\s*=\s*(.*?)\s*$", re.I | re.S)


@dataclass(frozen=True)
class Claim:
    value: str = ""             # the number or value literal as the page prints it
    unit: str = ""
    as_of: str = ""             # the date the source gives for the value
    entity: str = ""
    url: str = ""               # the source url (from the report, or the source id's url in the sub-agent's evidence)
    source_id: str = ""         # the S-id the research tool gave the source
    quote: str = ""
    locator: str = ""           # a label next to the quote, needed when the value appears more than once on the page
    snapshot_sha256: str = ""   # the page's sha256 when the sub-agent read it
    method: str = "report_line"  # how the claim was extracted
    status: str = ""            # set by verify_claim only; the report cannot set it
    reason: str = ""


def _unquote(text: str) -> str:
    text = text.strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in "\"'":
        return text[1:-1].strip()
    return text


def parse_claims(report: Any) -> list[Claim]:
    """The claims a researcher report lists, one per line of the form
    ``CLAIM | value=78.50 | unit=USD/bbl | as_of=2026-11-20 | entity=Brent | source_id=S2 | quote="Brent settled at 78.50"``.

    Motivation vs Logic
    Motivation: the lead needs each figure tied to the words of its source. A free-text report hides which words back which number.
    Logic: one line per claim, a CLAIM marker and key=value pairs split on pipes that precede a key. Unknown keys are ignored, a
    status key is not read (the report cannot set a status), and a report with no CLAIM line returns an empty list.
    """
    if not isinstance(report, str):
        return []
    claims: list[Claim] = []
    for line in report.splitlines():
        m = CLAIM_LINE.match(line)
        if not m:
            continue
        fields: dict[str, str] = {}
        for part in PAIR_SPLIT.split(m.group(1)):
            kv = KEY_VALUE.match(part)
            if kv and kv.group(1).lower() in CLAIM_FIELDS:
                fields[kv.group(1).lower()] = _unquote(kv.group(2))[:1000]
        if not fields:
            continue
        claims.append(Claim(value=fields.get("value", ""), unit=fields.get("unit", ""), as_of=fields.get("as_of", ""),
                            entity=fields.get("entity", ""), url=fields.get("url", ""), source_id=fields.get("source_id", ""),
                            quote=fields.get("quote", ""), locator=fields.get("locator", "")))
        if len(claims) == MAX_CLAIMS:
            break
    return claims


def _literal_matches(value: str, text: str) -> list[int]:
    """Start positions of the value as a literal number or word, not inside a longer number ('78.5' is not in '178.5')."""
    pattern = re.compile(r"(?<![\w.,])" + re.escape(value) + r"(?![\w]|[.,]\d)")
    return [m.start() for m in pattern.finditer(text)]


def _locator_near_quote(locator: str, quote: str, page_text: str) -> bool:
    """True when the locator appears in the window before any occurrence of the quote, or inside the quote."""
    needle = _normalise_for_quote(locator).casefold()
    page = _normalise_for_quote(page_text)
    q = _normalise_for_quote(quote)
    if not needle or not q:
        return False
    start = page.find(q)
    while start != -1:
        window = page[max(0, start - LOCATOR_WINDOW): start + len(q)].casefold()
        if needle in window:
            return True
        start = page.find(q, start + 1)
    return False


def verify_claim(claim: Claim, page_text: str | None, page_sha: str | None) -> Claim:
    """The claim with its status and reason set, checked against the stored page text.

    Motivation vs Logic
    Motivation: a claim is verified only when the page the mission stored says what the report says, in the same words.
    Logic, in order: no page text or no quote or no value is unverified. A page whose sha256 differs from the snapshot the
    sub-agent read is unverified (the quote may be gone). The quote must be in the page after whitespace and typographic quote
    normalisation (quote_in_page). The value literal must be inside the quote. A value the page prints more than once is
    ambiguous unless a locator names the place, and a locator must appear next to the quote. Otherwise the claim is verified.
    """
    def done(status: str, reason: str) -> Claim:
        return replace(claim, status=status, reason=reason)

    if not isinstance(page_text, str) or not page_text.strip():
        return done("unverified", "no cached page text for this source")
    if claim.snapshot_sha256 and page_sha and claim.snapshot_sha256 != page_sha:
        return done("unverified", "the page changed after the sub-agent read it; the quote may no longer be there")
    if not _normalise_for_quote(claim.quote):
        return done("unverified", "the claim gives no quote")
    value = _normalise_for_quote(claim.value)
    if not value:
        return done("unverified", "the claim gives no value")
    if not quote_in_page(claim.quote, page_text):
        return done("unverified", "the quote does not appear in the stored page text")
    if not _literal_matches(value, _normalise_for_quote(claim.quote)):
        return done("unverified", f"the value {claim.value!r} is not inside the quote")
    count = len(_literal_matches(value, _normalise_for_quote(page_text)))
    if count > 1 and not claim.locator:
        return done("ambiguous", f"the value {claim.value!r} appears {count} times in the page; a locator is needed to say which")
    if claim.locator and not _locator_near_quote(claim.locator, claim.quote, page_text):
        return done("unverified", f"the locator {claim.locator!r} does not appear next to the quote")
    return done("verified", "the quote and the value appear in the stored page text")


def _registry(evidence: Any) -> tuple[dict[str, str], dict[str, str]]:
    """Source url -> sha256 the sub-agent read, and source id -> url, from the tool results in its evidence."""
    sha_by_url: dict[str, str] = {}
    url_by_id: dict[str, str] = {}

    def note(src: dict) -> None:
        urls = [str(src[k]) for k in ("url", "final_url") if isinstance(src.get(k), str) and src.get(k)]
        if not urls:
            return
        for url in urls:
            if isinstance(src.get("sha256"), str) and src["sha256"]:
                sha_by_url[url] = src["sha256"]
        if isinstance(src.get("id"), str) and src["id"]:
            url_by_id[src["id"]] = urls[0]

    for h in evidence if isinstance(evidence, list) else []:
        result = h.get("result") if isinstance(h, dict) else None
        if not isinstance(result, dict):
            continue
        note(result)
        for src in result.get("sources") or []:
            if isinstance(src, dict):
                note(src)
    return sha_by_url, url_by_id


def summarise(claims: list[dict]) -> str:
    counts = {s: sum(1 for c in claims if c.get("status") == s) for s in STATUSES}
    noun = "claim" if len(claims) == 1 else "claims"
    return f"{len(claims)} {noun}: {counts['verified']} verified, {counts['unverified']} unverified, {counts['ambiguous']} ambiguous"


def check_report_claims(report: Any, evidence: Any) -> dict:
    """Every claim the report lists, verified against the cached text of its source. Returns the claims as plain dicts (stored
    with the ledger step) and a one-line summary. Reads the research cache; it never calls a model.

    Motivation vs Logic
    Motivation: the researcher's report reaches the lead as prose. The lead cannot tell which figure has a quote behind it.
    Logic: each claim's url comes from the report or from its source id in the sub-agent's evidence. The snapshot sha256 is the
    one the sub-agent's own tool results recorded, so a page that changed since is unverified. The cached page is read with
    stored_page, and verify_claim decides.
    """
    claims = parse_claims(report)
    if not claims:
        return {"claims": [], "summary": ""}
    sha_by_url, url_by_id = _registry(evidence)
    pages: dict[str, dict | None] = {}
    checked: list[Claim] = []
    for claim in claims:
        url = claim.url or url_by_id.get(claim.source_id, "")
        snapshot = claim.snapshot_sha256 or sha_by_url.get(url, "")
        claim = replace(claim, url=url, snapshot_sha256=snapshot)
        if not url:
            checked.append(replace(claim, status="unverified",
                                   reason="the claim names no source url, and its source id matches no source the sub-agent read"))
            continue
        if url not in pages:
            pages[url] = stored_page(url)
        page = pages[url]
        checked.append(verify_claim(claim, page.get("text") if page else None, page.get("sha256") if page else None))
    rows = [asdict(c) for c in checked]
    return {"claims": rows, "summary": summarise(rows)}
