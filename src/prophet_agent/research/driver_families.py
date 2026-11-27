"""The driver families a forecast of each kind of target has to account for, and what counts as covering them.

Motivation vs Logic
Motivation (owner review, 2026-11-27): the panel forecasts were built from each series' own history with, at most, one
covariate tried. The owner asked for the political, geopolitical, economic and other fundamental factors that drive the
outcome to be searched broadly and then measured, so a forecast reads as the work of an analyst desk and not a single model
fit. The EPL mission rated Brighton the title favourite from eleven matches of goals because nothing else entered the model.
Logic: a target is matched to a domain (policy rate, labour market, precious metal, equity index, currency, commodity, sports
league, or general). Each domain lists its driver families: what the family is, the data that measures it (named series or
sites to fetch), and search queries for the news and official commentary that move it. It also lists the benchmarks a
forecast is compared with (official projections, market-implied paths, consensus) and the minimum coverage: families
researched, families tested against the target, benchmarks quoted, and scenarios applied. ``runtime.prediction_quality``
holds a finish until the coverage is met, with every finding traced to a call in the ledger.
"""
from __future__ import annotations

import re
from typing import Any


def _f(key: str, what: str, data: list[str], queries: list[str]) -> dict:
    return {"key": key, "what": what, "data": data, "queries": queries}


# Families shared by the macro domains. Data hints name series ids or sites the open-data and market tools reach.
_MONETARY = _f("monetary_policy", "central bank stance, decisions, guidance, minutes and the board's own projections",
               ["policy rate history (RBA F1, FRED FEDFUNDS, ECB MRO)", "meeting calendar and statements"],
               ["{place} central bank statement latest decision guidance", "{place} central bank minutes outlook risks"])
_INFLATION = _f("inflation", "headline and underlying inflation, expectations and wages",
                ["CPI and trimmed mean (ABS, RBA G1; FRED CPIAUCSL, PCEPILFE)", "inflation expectations (FRED T5YIE, T10YIE)",
                 "wage price index"],
                ["{place} inflation latest release trimmed mean", "{place} inflation expectations survey"])
_LABOUR = _f("labour_market", "unemployment, employment growth, hours, participation, vacancies and job ads",
             ["unemployment rate (ABS 6202, RBA H5, FRED UNRATE, ILOSTAT)", "job vacancies and job ads", "hours worked"],
             ["{place} labour force survey latest unemployment", "{place} job vacancies hiring intentions"])
_ACTIVITY = _f("activity", "output growth, business and consumer surveys, retail spending, PMIs",
               ["GDP growth (ABS, World Bank, FRED GDPC1)", "PMI and business confidence", "retail sales"],
               ["{place} GDP growth latest quarter", "{place} PMI business conditions latest"])
_FISCAL = _f("fiscal_policy", "budget stance, public spending, tax changes, stimulus, election promises",
             ["budget balance and spending plans", "government bond issuance"],
             ["{place} budget fiscal stimulus spending plans", "{place} election economic policy platform"])
_EXTERNAL = _f("external_trade", "exports, terms of trade, tariffs, trading partners' demand (China, US), FDI",
               ["exports and terms of trade", "trading partner growth", "tariff schedules"],
               ["{place} exports tariffs trade outlook", "{place} China demand exports latest"])
_GEOPOLITICS = _f("geopolitics", "wars, sanctions, trade disputes, elections and policy shocks that move risk and prices",
                  ["geopolitical risk index (Caldara-Iacoviello GPR)", "dated event timeline from news"],
                  ["{place} geopolitical risk outlook {year}", "tariffs sanctions conflict impact {target}"])
_COMMODITIES = _f("commodities_energy", "oil, gas, metals and food prices feeding costs, incomes and inflation",
                  ["Brent and WTI (FRED DCOILBRENTEU, market CL=F)", "iron ore, copper, gas prices"],
                  ["oil price outlook {year}", "commodity prices impact {place}"])
_FINANCIAL = _f("financial_conditions", "bond yields, credit spreads, equity and housing prices, bank lending",
                ["10-year yield (FRED DGS10, RBA F2)", "credit spreads (FRED BAMLH0A0HYM2)", "house prices and credit growth"],
                ["{place} financial conditions credit growth mortgage", "{place} bond yields outlook"])
_MARKET_PATH = _f("market_pricing", "what traders price for the target: futures, swaps, options-implied ranges",
                  ["rate futures or OIS implied path (ASX rate tracker, CME FedWatch)", "futures curve for the target"],
                  ["{target} market implied path futures pricing", "{place} cash rate futures expectations"])
_GLOBAL = _f("global_spillover", "the Fed, ECB, China and global growth acting through rates, currency and demand",
             ["Fed funds and US 10-year (FRED)", "USD index (FRED DTWEXBGS)", "global PMI"],
             ["Fed outlook rate path {year}", "global growth outlook IMF {year}"])
_STRUCTURAL = _f("structural", "population, migration, participation trends, productivity, sector mix and reforms",
                 ["population and migration (ABS, World Bank)", "participation rate", "productivity"],
                 ["{place} migration population growth labour supply", "{place} structural reform productivity"])
_SENTIMENT = _f("sentiment", "consumer and business confidence, surveys of expectations, news tone",
                ["consumer confidence", "business confidence", "expectations surveys"],
                ["{place} consumer confidence latest", "{place} business confidence survey latest"])

DOMAINS: dict[str, dict[str, Any]] = {
    "sports_league": {
        "match": r"premier league|\bepl\b|la liga|bundesliga|serie a|ligue 1|\bnba\b|\bnfl\b|\bnhl\b|\bafl\b|\bnrl\b|"
                 r"\bseason\b|\bleague\b|\bclubs?\b|title race|relegation|\bfixtures?\b",
        "families": [
            _f("multi_season_strength", "club strength over earlier seasons, so a hot start is not read as strength",
               ["earlier seasons' results (football-data.co.uk/mmz4281/<YYZZ>/E0.csv, fixturedownload, openfootball)"],
               ["{target} last season final table points"]),
            _f("chance_quality", "expected goals and shots: how good the chances were, not how many went in",
               ["HxG, AxG, shots and shots on target per match (football-data.co.uk results files)"],
               ["{target} expected goals xG table this season"]),
            _f("market_prices", "bookmaker match prices and outright odds: everything the market knows, priced",
               ["closing prices per match (AvgCH, AvgCD, AvgCA), over/under 2.5", "outright title and top-four odds"],
               ["{target} title odds outright betting latest", "{target} top four relegation odds"]),
            _f("squad_and_transfers", "squad value, signings and sales, depth for the fixture load",
               ["squad market values (Transfermarkt)", "transfer window ins and outs"],
               ["{target} squad value transfers summer window", "{target} January transfer window rumours"]),
            _f("availability", "injuries, suspensions and international duty of key players",
               ["injury and suspension lists"], ["{target} injury news latest key players out"]),
            _f("management", "manager changes, tactics, form under the current manager",
               ["manager tenure and record"], ["{target} manager sacked appointed latest"]),
            _f("fixture_load", "remaining schedule strength, European games, congestion",
               ["remaining fixtures with opponent strength", "European competition calendar"],
               ["{target} fixture difficulty remaining schedule", "{target} Champions League fixture congestion"]),
        ],
        "benchmarks": ["bookmaker outright odds (title, top four, relegation), overround removed",
                       "published model projections (Opta supercomputer, FiveThirtyEight-style ratings, ClubElo)"],
        "minimum": {"researched": 4, "tested": 0, "benchmarks": 1, "scenarios": 0},
    },
    "policy_rate": {
        "match": r"cash rate|policy rate|interest rate|official rate|\bocr\b|fed funds|federal funds|bank rate|"
                 r"\brba\b|\bfomc\b|\becb\b|refinancing rate",
        "families": [_MONETARY, _INFLATION, _LABOUR, _ACTIVITY, _MARKET_PATH, _GLOBAL, _FINANCIAL, _FISCAL, _EXTERNAL,
                     _GEOPOLITICS, _COMMODITIES, _SENTIMENT],
        "benchmarks": ["the central bank's own forecasts (Statement on Monetary Policy, dot plot)",
                       "market-implied path from rate futures or OIS", "economists' consensus (major banks' rate calls)"],
        "minimum": {"researched": 5, "tested": 2, "benchmarks": 1, "scenarios": 2},
    },
    "labour_market": {
        "match": r"unemploy|jobless|employment|labou?r (?:market|force)|payrolls|participation rate",
        "families": [_ACTIVITY, _LABOUR, _MONETARY, _INFLATION, _FISCAL, _EXTERNAL, _GEOPOLITICS, _STRUCTURAL,
                     _SENTIMENT, _FINANCIAL, _GLOBAL],
        "benchmarks": ["official projections (central bank forecasts, treasury budget forecasts)",
                       "international projections (IMF World Economic Outlook, World Bank, ADB, OECD)",
                       "economists' consensus"],
        "minimum": {"researched": 5, "tested": 2, "benchmarks": 1, "scenarios": 2},
    },
    "precious_metal": {
        "match": r"\bgold\b|silver|platinum|palladium|bullion|xau",
        "families": [
            _f("real_rates", "US real yields: the opportunity cost of holding a metal that pays nothing",
               ["US 10-year TIPS yield (FRED DFII10)", "US 10-year nominal (FRED DGS10)"],
               ["real yields gold outlook {year}"]),
            _f("us_dollar", "the dollar's strength, since metals are priced in dollars",
               ["broad dollar index (FRED DTWEXBGS, market DX-Y.NYB)"], ["dollar outlook {year} gold"]),
            _f("central_bank_buying", "official sector purchases (PBoC, RBI, Poland, Turkey) and reserve diversification",
               ["World Gold Council central bank statistics"], ["central bank gold buying latest World Gold Council"]),
            _f("investment_flows", "ETF holdings, futures positioning (CFTC COT), physical demand from China and India",
               ["gold ETF holdings", "CFTC commitments of traders"], ["gold ETF flows latest month", "gold positioning COT"]),
            _MONETARY | {"what": "the Fed's path and the market's pricing of cuts or hikes"},
            _INFLATION | {"what": "inflation and inflation expectations: the hedge demand"},
            _GEOPOLITICS | {"what": "wars, sanctions, reserve freezes and elections that raise safe-haven demand"},
            _f("risk_appetite", "equity drawdowns and volatility that send money to havens",
               ["VIX (FRED VIXCLS)", "S&P 500 (market ^GSPC)"], ["gold safe haven equities selloff"]),
            _FISCAL | {"what": "US deficits, debt ceiling and debasement worries"},
            _f("supply", "mine output, recycling and costs", ["mine production"], ["gold mine supply outlook"]),
        ],
        "benchmarks": ["bank forecasts (Goldman Sachs, JPMorgan, UBS price targets)", "futures curve",
                       "World Gold Council outlook"],
        "minimum": {"researched": 5, "tested": 2, "benchmarks": 1, "scenarios": 2},
    },
    "equity_index": {
        "match": r"s&p|sp ?500|\bspx\b|nasdaq|dow jones|asx ?200|ftse|nikkei|stock index|equit|stock market|shares",
        "families": [
            _f("earnings", "earnings growth, revisions and guidance", ["forward EPS and revisions (FactSet)"],
               ["S&P 500 earnings growth forecast {year}", "earnings revisions latest quarter"]),
            _f("valuation", "forward P/E, equity risk premium against bond yields", ["forward P/E", "ERP"],
               ["S&P 500 forward PE valuation latest"]),
            _MONETARY | {"what": "the Fed's path, liquidity and balance sheet"},
            _FINANCIAL | {"what": "yields, credit spreads and liquidity"},
            _ACTIVITY | {"what": "growth, PMIs and recession odds"},
            _INFLATION,
            _f("volatility", "VIX, realised volatility and drawdown regimes", ["VIX (FRED VIXCLS, market ^VIX)"],
               ["VIX volatility outlook"]),
            _f("concentration_themes", "mega-cap and AI capital spending concentration", ["top-10 weight", "capex plans"],
               ["AI capex mega cap concentration S&P 500 risk"]),
            _FISCAL | {"what": "tax, tariffs, deficits and elections"},
            _GEOPOLITICS,
            _f("flows_positioning", "buybacks, fund flows, positioning and sentiment surveys", ["AAII, fund flows"],
               ["equity fund flows positioning latest", "stock buybacks {year}"]),
        ],
        "benchmarks": ["strategists' year-end targets (Goldman, Morgan Stanley, JPMorgan)", "futures and options-implied range"],
        "minimum": {"researched": 5, "tested": 2, "benchmarks": 1, "scenarios": 2},
    },
    "currency": {
        "match": r"exchange rate|aud/?usd|eur/?usd|usd/?jpy|\bforex\b|\bfx\b|currency",
        "families": [_MONETARY, _f("rate_differential", "interest rate gap between the two economies",
                                   ["2-year yield spread"], ["{target} rate differential outlook"]),
                     _COMMODITIES, _EXTERNAL, _GLOBAL, _GEOPOLITICS, _FINANCIAL, _INFLATION, _SENTIMENT],
        "benchmarks": ["bank forecasts", "forward rates"],
        "minimum": {"researched": 5, "tested": 2, "benchmarks": 1, "scenarios": 2},
    },
    "commodity": {
        "match": r"\boil\b|brent|\bwti\b|copper|iron ore|natural gas|\blng\b|wheat|coal|lithium",
        "families": [_f("supply", "production, OPEC+ decisions, outages, inventories", ["inventories (EIA)"],
                        ["{target} supply OPEC inventories latest"]),
                     _f("demand", "global and Chinese demand", ["China imports", "global PMI"],
                        ["{target} demand China outlook"]),
                     _GEOPOLITICS, _GLOBAL, _FINANCIAL, _MONETARY, _MARKET_PATH, _SENTIMENT],
        "benchmarks": ["futures curve", "agency outlooks (EIA, IEA, OPEC)", "bank forecasts"],
        "minimum": {"researched": 5, "tested": 2, "benchmarks": 1, "scenarios": 2},
    },
    "general": {
        "match": r"",
        "families": [_f("demand", "what drives demand for the measured thing", ["related series"],
                        ["{target} demand drivers"]),
                     _f("price_and_supply", "prices, capacity and supply constraints", ["prices"],
                        ["{target} supply price outlook"]),
                     _f("calendar", "season, holidays, events", ["calendar"], ["{target} seasonal pattern events"]),
                     _ACTIVITY, _MONETARY, _FISCAL, _GEOPOLITICS, _SENTIMENT],
        "benchmarks": ["published outlooks or industry forecasts for the same measure"],
        "minimum": {"researched": 4, "tested": 1, "benchmarks": 1, "scenarios": 2},
    },
}

ORDER = ("sports_league", "policy_rate", "labour_market", "precious_metal", "equity_index", "currency", "commodity")


def detect(goal: str) -> str:
    """The domain whose words appear in the goal, in a fixed order (a league before a rate, a rate before a labour market);
    'general' when none does."""
    text = goal or ""
    for name in ORDER:
        if re.search(DOMAINS[name]["match"], text, re.I):
            return name
    return "general"


_PLACES = (("Australia", r"australia|\brba\b|\basx\b|\bau(?:d|s)?\b"), ("Vietnam", r"viet ?nam|\bvn\b"),
           ("United States", r"united states|\bus\b|\bfed\b|s&p|nasdaq|dow"), ("United Kingdom", r"\buk\b|britain|england"),
           ("euro area", r"euro area|eurozone|\becb\b"), ("China", r"china"), ("Japan", r"japan|\bboj\b"))


def place_of(goal: str) -> str:
    for name, rx in _PLACES:
        if re.search(rx, goal or "", re.I):
            return name
    return "the economy"


def plan(goal: str, domain: str | None = None, year: int | None = None) -> dict:
    """The families, benchmarks and minimums for one goal, with the search queries filled in for its place and target."""
    import datetime as dt
    name = domain if domain in DOMAINS else detect(goal)
    spec = DOMAINS[name]
    place = place_of(goal)
    target = re.sub(r"\s+", " ", (goal or "").strip())[:80] or "the target"
    yr = year or dt.date.today().year
    fams = []
    for f in spec["families"]:
        fams.append({"family": f["key"], "what": f["what"], "data": list(f["data"]),
                     "queries": [q.format(place=place, target=target, year=yr) for q in f["queries"]]})
    return {"domain": name, "place": place, "families": fams, "benchmarks": list(spec["benchmarks"]),
            "minimum": dict(spec["minimum"])}


def family_keys(domain: str) -> list[str]:
    return [f["key"] for f in DOMAINS.get(domain, DOMAINS["general"])["families"]]
