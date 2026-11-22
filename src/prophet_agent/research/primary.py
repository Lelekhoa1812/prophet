"""Registry of primary pages for each market series family, and the family a query belongs to.

Motivation vs Logic
Motivation: search engines did not surface the official pages for oil or FX in live runs. The EIA, OPEC, IEA, ECB, BOJ and
Fed pages carry the dated facts a forecast needs, and they are known in advance. Fetching them directly is cheaper and
more reliable than hoping a news query returns them.
Logic: a fixed mapping from series family to a short list of {name, url, what}. The URLs are a starting list maintained
by hand; they are fetched through the same netguard and robots path as any other page, so the registry adds no new
network rules. Detection is keyword based: the family with the most matching words in the query wins, and ties go to the
order of the table. No family matches when no keyword does, and then only the search runs.
"""
from __future__ import annotations

import re

PRIMARY_SOURCES: dict[str, tuple[dict[str, str], ...]] = {
    "oil": (
        {"name": "EIA Weekly Petroleum Status Report", "url": "https://www.eia.gov/petroleum/supply/weekly/",
         "what": "US crude stocks, refinery runs and imports, weekly"},
        {"name": "EIA Short-Term Energy Outlook", "url": "https://www.eia.gov/outlooks/steo/",
         "what": "EIA monthly price and supply forecast"},
        {"name": "EIA WTI spot price history", "url": "https://www.eia.gov/dnav/pet/hist/RWTCD.htm",
         "what": "daily WTI spot price"},
        {"name": "OPEC Monthly Oil Market Report", "url": "https://www.opec.org/opec_web/en/publications/338.htm",
         "what": "OPEC demand, supply and price outlook"},
        {"name": "IEA Oil Market Report", "url": "https://www.iea.org/data-and-statistics/data-product/oil-market-report-omr",
         "what": "IEA monthly supply, demand and stock outlook"},
    ),
    "gas": (
        {"name": "EIA Natural Gas Weekly Update", "url": "https://www.eia.gov/naturalgas/weekly/",
         "what": "US natural gas storage and prices, weekly"},
        {"name": "EIA Henry Hub spot price history", "url": "https://www.eia.gov/dnav/ng/hist/rngwhhdD.htm",
         "what": "daily Henry Hub spot price"},
        {"name": "EIA Short-Term Energy Outlook", "url": "https://www.eia.gov/outlooks/steo/",
         "what": "EIA monthly price and supply forecast"},
    ),
    "fx": (
        {"name": "ECB euro foreign exchange reference rates",
         "url": "https://www.ecb.europa.eu/stats/policy_and_exchange_rates/euro_reference_exchange_rates/html/index.en.html",
         "what": "daily ECB reference rates, euro against major currencies"},
        {"name": "ECB monetary policy decisions", "url": "https://www.ecb.europa.eu/press/govcdec/mopo/html/index.en.html",
         "what": "ECB rate decisions and statements"},
        {"name": "ECB calendar", "url": "https://www.ecb.europa.eu/press/calendars/mgcgc/html/index.en.html",
         "what": "ECB governing council meeting dates"},
        {"name": "BOJ monetary policy meeting schedule", "url": "https://www.boj.or.jp/en/mopo/mpmsche_minu/index.htm",
         "what": "Bank of Japan policy meeting dates and minutes"},
        {"name": "Federal Reserve FOMC calendar", "url": "https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm",
         "what": "FOMC meeting dates"},
        {"name": "IMF exchange rates", "url": "https://www.imf.org/external/np/fin/data/rms_rep.aspx",
         "what": "IMF representative exchange rates, per US dollar"},
    ),
    "rates": (
        {"name": "Federal Reserve FOMC calendar", "url": "https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm",
         "what": "FOMC meeting dates"},
        {"name": "Federal Reserve H.15 selected interest rates", "url": "https://www.federalreserve.gov/releases/h15/",
         "what": "US Treasury and policy rates, daily"},
        {"name": "ECB monetary policy decisions", "url": "https://www.ecb.europa.eu/press/govcdec/mopo/html/index.en.html",
         "what": "ECB rate decisions and statements"},
        {"name": "Bank of England Bank Rate", "url": "https://www.bankofengland.co.uk/monetary-policy/the-interest-rate-bank-rate",
         "what": "UK policy rate and decisions"},
        {"name": "BOJ monetary policy meeting schedule", "url": "https://www.boj.or.jp/en/mopo/mpmsche_minu/index.htm",
         "what": "Bank of Japan policy meeting dates and minutes"},
    ),
    "equities": (
        {"name": "SEC newsroom", "url": "https://www.sec.gov/newsroom",
         "what": "US securities regulator announcements"},
        {"name": "Federal Reserve press releases", "url": "https://www.federalreserve.gov/newsevents/pressreleases.htm",
         "what": "policy and supervision announcements"},
        {"name": "BEA release schedule", "url": "https://www.bea.gov/news/schedule",
         "what": "US GDP and income release dates"},
    ),
    "inflation": (
        {"name": "BLS CPI release schedule", "url": "https://www.bls.gov/schedule/news_release/cpi.htm",
         "what": "US consumer price index release dates"},
        {"name": "BLS consumer price index", "url": "https://www.bls.gov/cpi/",
         "what": "US consumer price index, latest release"},
        {"name": "Eurostat release calendar", "url": "https://ec.europa.eu/eurostat/news/release-calendar",
         "what": "euro area flash HICP and other release dates"},
        {"name": "ONS inflation and price indices", "url": "https://www.ons.gov.uk/economy/inflationandpriceindices",
         "what": "UK consumer price inflation"},
        {"name": "Japan Statistics Bureau consumer price index", "url": "https://www.stat.go.jp/english/data/cpi/index.html",
         "what": "Japan consumer price index"},
    ),
    "gold": (
        {"name": "LBMA prices and data", "url": "https://www.lbma.org.uk/prices-and-data",
         "what": "London gold benchmark prices"},
        {"name": "USGS gold statistics", "url": "https://www.usgs.gov/centers/national-minerals-information-center/gold-statistics-and-information",
         "what": "US gold production, trade and consumption statistics"},
    ),
}

FAMILY_KEYWORDS: dict[str, frozenset[str]] = {
    "oil": frozenset("oil wti brent crude opec petroleum gasoline".split()),
    "gas": frozenset("gas henry lng natural".split()),
    "gold": frozenset("gold xau bullion".split()),
    "fx": frozenset("fx forex currency currencies exchange eur usd jpy gbp aud cny yen euro dollar".split()),
    "rates": frozenset("rate rates yield yields treasury bond bonds fomc ecb boj policy".split()),
    "equities": frozenset("stock stocks equity equities shares index nasdaq earnings".split()),
    "inflation": frozenset("inflation cpi hicp deflation prices".split()),
}
WORD = re.compile(r"[a-z]+")


def detect_family(text: str) -> str | None:
    """The series family whose keywords appear most often in ``text``; None when no keyword appears."""
    words = WORD.findall((text or "").lower())
    best, best_hits = None, 0
    for family, keywords in FAMILY_KEYWORDS.items():
        hits = sum(1 for word in words if word in keywords)
        if hits > best_hits:
            best, best_hits = family, hits
    return best


def registry_pages(family: str | None) -> list[dict[str, str]]:
    """Copies of the primary pages for a family; an empty list for None or an unknown family."""
    return [dict(page) for page in PRIMARY_SOURCES.get(family or "", ())]
