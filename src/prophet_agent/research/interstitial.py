"""Recognise pages that are walls, not content: adblock notices, consent screens, bot checks, CAPTCHAs, login walls and
paywalls. Such a page is never counted as a fetched source.

Motivation vs Logic
Motivation: in a live EPL mission two "Adblocker detected! Please consider reading this notice." pages came back from
web_fetch with ok true and were fetched six times. The planner then treated them as sources, and the stall rule fired
far too late. A wall is not evidence, and a bot check is a control that the agent must respect, not climb over.
Logic: the classifier only reads what the page shows (title, visible text, HTTP status) and the markers the page
itself carries (a challenge script, a CAPTCHA widget, a consent manager). It never tries to get past a wall. A page
counts as a wall only when a specific phrase or marker is present, and the generic words (cookies, captcha, login)
alone are never enough. Most walls are short; a long article that mentions a cookie policy or a newsletter CAPTCHA is
therefore not flagged, because the length rule and the phrase rule must both hold for the softer kinds. Each verdict
carries the evidence that matched, so the planner can see why a page was refused.
"""
from __future__ import annotations

from dataclasses import dataclass

KINDS = ("adblock_wall", "consent_wall", "bot_check", "captcha", "login_wall", "paywall")
LABELS = {"adblock_wall": "adblock wall", "consent_wall": "consent wall", "bot_check": "bot check",
          "captcha": "CAPTCHA", "login_wall": "login wall", "paywall": "paywall"}
SHORT_TEXT = 1500      # visible characters: a page this short that carries a wall marker is the wall itself
PAYWALL_TEXT = 3000    # a paywalled article shows a teaser; a longer page with the phrase is a real article

ADBLOCK_PHRASES = ("adblocker detected", "ad blocker detected", "adblock detected", "ad-blocker detected",
                   "please disable your ad blocker", "disable your adblocker", "turn off your ad blocker",
                   "turn off your adblock", "we noticed you are using an ad blocker")
BOT_PHRASES = ("verify you are human", "checking your browser before accessing", "checking if the site connection is secure",
               "unusual traffic from your", "press & hold to confirm you are human", "are you a robot")
BOT_HTML = ("cf-chl-", "_cf_chl_opt", "challenge-platform", "cf-browser-verification")
CAPTCHA_HTML = ("g-recaptcha", "grecaptcha", "recaptcha/api", "google.com/recaptcha", "h-captcha", "hcaptcha.com",
                "cf-turnstile", "challenges.cloudflare.com/turnstile")
CAPTCHA_PHRASES = ("complete the captcha", "solve the captcha", "please complete the security check",
                   "prove you are not a robot", "i am not a robot", "i'm not a robot")
CONSENT_HTML = ("cookiebot", "onetrust", "consentmanager", "sp_message", "qc-cmp", "didomi", "cookie-consent")
CONSENT_PHRASES = ("accept all", "reject all", "manage preferences", "manage cookies", "cookie settings",
                   "cookie preferences", "we use cookies", "we value your privacy", "before you continue",
                   "consent to cookies", "accept cookies")
LOGIN_PHRASES = ("sign in to continue", "log in to continue", "login to continue", "sign in to view",
                 "log in to view", "login required", "you must log in", "you must be logged in",
                 "please sign in to continue", "sign in to read", "members only")
PAYWALL_PHRASES = ("subscribe to continue reading", "subscribe to read", "this article is for subscribers",
                   "subscribers only", "to continue reading, subscribe", "to keep reading, subscribe",
                   "you have reached your free article limit", "you've reached your free article limit",
                   "become a subscriber to continue")


@dataclass(frozen=True)
class Verdict:
    """kind is "none" or one of KINDS. evidence lists what matched, such as "title: 'adblocker detected'"."""
    kind: str = "none"
    evidence: tuple[str, ...] = ()

    @property
    def blocked(self) -> bool:
        return self.kind != "none"

    @property
    def label(self) -> str:
        return LABELS.get(self.kind, "none")


def _hits(haystack: str, phrases: tuple[str, ...], where: str) -> list[str]:
    return [f"{where}: {phrase!r}" for phrase in phrases if phrase in haystack]


def classify_page(title: str = "", text: str = "", status: int | None = None, html: str = "") -> Verdict:
    """Classify one fetched page. Pass the HTML when there is some; a cached page has only its text and title.

    Motivation vs Logic
    Motivation: the same wall must be recognised whether it arrives as a raw response or as a stored text, and a
    rendered page must be checked after its scripts have run.
    Logic: the checks run in order of how specific they are: adblock, bot check, CAPTCHA, consent, login, paywall. The
    first kind that matches is returned with its evidence; "none" comes back when nothing matches. Phrases are compared
    in lower case. Markers in the HTML are used only where the HTML is available, and each softer kind also needs a
    short visible text.
    """
    title_l = " ".join((title or "").split()).lower()
    text_l = " ".join((text or "").split()).lower()
    html_l = (html or "").lower()
    short = len(text_l) < SHORT_TEXT
    status_note = f"HTTP {status}" if status and status >= 400 else ""

    # 1. Adblock notice: the phrase in the title, or in a short page.
    evidence = _hits(title_l, ADBLOCK_PHRASES, "title")
    if not evidence and short:
        evidence = _hits(text_l, ADBLOCK_PHRASES, "text")
    if evidence:
        return Verdict("adblock_wall", tuple(evidence))

    # 2. Bot check: challenge phrases, or the Cloudflare "Just a moment" page with its challenge markers.
    evidence = _hits(title_l, BOT_PHRASES, "title")
    if short and not evidence:
        evidence = _hits(text_l, BOT_PHRASES, "text")
    if not evidence and title_l.startswith("just a moment") and (short or _hits(html_l, BOT_HTML, "html")):
        evidence = [f"title: {title_l[:60]!r}"]
    if not evidence and short:
        evidence = _hits(html_l, BOT_HTML, "html")
    if evidence:
        return Verdict("bot_check", tuple(evidence + ([status_note] if status_note else [])))

    # 3. CAPTCHA widget or challenge text on a short page.
    if short:
        evidence = _hits(html_l, CAPTCHA_HTML, "html") + _hits(text_l, CAPTCHA_PHRASES, "text")
        if evidence:
            return Verdict("captcha", tuple(evidence))

    # 4. Consent screen: a consent manager, or two consent phrases, on a short page.
    if short:
        evidence = _hits(html_l, CONSENT_HTML, "html")
        phrases = _hits(text_l + " " + title_l, CONSENT_PHRASES, "text")
        if evidence or len(phrases) >= 2:
            return Verdict("consent_wall", tuple(evidence + phrases))

    # 5. Login wall: a sign-in phrase on a short page, or a password form with a sign-in heading on a short page.
    if short:
        evidence = _hits(title_l + " " + text_l, LOGIN_PHRASES, "text")
        if not evidence and 'type="password"' in html_l and ("sign in" in text_l or "log in" in text_l
                                                             or "login" in text_l):
            evidence = ["html: password form"]
        if evidence:
            return Verdict("login_wall", tuple(evidence))

    # 6. Paywall: a subscription phrase on a page that shows only a teaser.
    if len(text_l) < PAYWALL_TEXT:
        evidence = _hits(title_l, PAYWALL_PHRASES, "title") or _hits(text_l, PAYWALL_PHRASES, "text")
        if evidence:
            return Verdict("paywall", tuple(evidence))

    return Verdict()
