"""league_backtest: score the season simulator on past seasons of the same league, against simple rules.

Motivation vs Logic
Motivation (Mission 10, owner review 2026-11-28): an EPL forecast quoted title and relegation chances with only a match-level
backtest beside them, so nobody could say how many points its final table is usually out by. A final-table forecast is
judged on the final table.
Logic: the complete seasons before the current one are fetched from football-data.co.uk (one keyless CSV per season and
league; the current season is the one that started in August of this year, or of last year before August), and
forecast.sports.season_backtest cuts each after the given matchdays, forecasts every club's final points by pace, last season,
and the rating model on goals, with earlier seasons, and with closing prices, and scores the forecasts against the final
table. The harness runs it once for a league-table goal; the planner can also call it. Every figure is measured on the
fetched files.
"""
from __future__ import annotations

import datetime as dt
from collections.abc import Callable
from typing import Any

from prophet_agent.forecast.sports import parse_matches, season_backtest
from prophet_agent.opendata import market
from prophet_agent.runtime.runner import Tool

ALLOWED = {"league", "seasons", "cuts"}
LEAGUES = {"E0": "English Premier League", "E1": "English Championship", "SP1": "La Liga", "D1": "Bundesliga",
           "I1": "Serie A", "F1": "Ligue 1", "SC0": "Scottish Premiership", "N1": "Eredivisie"}
BASE = "https://football-data.co.uk/mmz4281/{code}/{league}.csv"    # the www host answers with a redirect to this one
DEFAULT_SEASONS = 6
MAX_SEASONS = 10


def season_code(start_year: int) -> str:
    return f"{start_year % 100:02d}{(start_year + 1) % 100:02d}"


def complete_seasons(today: dt.date, n: int) -> list[int]:
    """Start years of the ``n`` complete seasons before the current one, oldest first."""
    current = today.year if today.month >= 8 else today.year - 1
    return list(range(current - n, current))


def _fetch(url: str) -> bytes:
    return market._get(url, timeout_s=market.DEFAULT_TIMEOUT_S, resolver=None)[0]


def _run(args: dict, _key: str, get: Callable[[str], bytes] | None = None,
         today: Callable[[], dt.date] | None = None) -> dict[str, Any]:
    extra = sorted(set(args) - ALLOWED)
    if extra:
        raise ValueError(f"unknown arguments {extra}; allowed: {sorted(ALLOWED)}")
    league = args.get("league", "E0")
    if league not in LEAGUES:
        raise ValueError(f"league must be one of {sorted(LEAGUES)} (football-data.co.uk codes)")
    n = args.get("seasons", DEFAULT_SEASONS)
    if not isinstance(n, int) or isinstance(n, bool) or not 2 <= n <= MAX_SEASONS:
        raise ValueError(f"seasons must be a whole number from 2 to {MAX_SEASONS}")
    cuts = args.get("cuts", [8, 12, 19])
    if (not isinstance(cuts, list) or not cuts or len(cuts) > 5
            or not all(isinstance(c, int) and not isinstance(c, bool) and 1 <= c <= 37 for c in cuts)):
        raise ValueError("cuts must be a list of 1 to 5 matchdays from 1 to 37, for example [8, 12, 19]")
    fetch = get or _fetch
    years = complete_seasons((today or dt.date.today)(), n)
    seasons, missing, sources = [], [], []
    for y in years:
        url = BASE.format(code=season_code(y), league=league)
        try:
            text = fetch(url).decode("utf-8-sig", errors="replace")
            ms = parse_matches(text, f"{league}_{season_code(y)}.csv")
        except (market.MarketError, ValueError) as exc:
            missing.append({"season": f"{y}/{(y + 1) % 100:02d}", "url": url, "reason": str(exc)[:150]})
            continue
        seasons.append((f"{y}/{(y + 1) % 100:02d}", ms))
        sources.append(url)
    res = season_backtest(seasons, tuple(sorted(set(cuts))))
    return {"league": league, "league_name": LEAGUES[league], "sources": sources, "missing": missing, **res}


def observe(result: dict) -> dict:
    lines = []
    for s in result.get("summary", []):
        extra = ", ".join(f"{k.replace('brier_', 'Brier ')} {s[k]}" for k in ("brier_title", "brier_top4", "brier_relegation")
                          if k in s)
        lines.append(f"after matchday {s['cut']}: {s['model']} final-points MAE {s['mae']} (RMSE {s['rmse']}) over "
                     f"{s['seasons']} seasons" + (f"; {extra}" if extra else ""))
    paired = [f"after matchday {p['cut']}: the rating model with history and prices beat {p['against']} in {p['wins']} of "
              f"{p['seasons']} seasons (MAE {p['mae_gain_pct']:+.1f}%)" for p in result.get("paired", [])]
    return {"league": result.get("league_name"), "seasons_scored": result.get("seasons_scored"), "summary": lines,
            "paired": paired, "missing": [f"{m['season']}: {m['reason']}" for m in result.get("missing", [])],
            "figure_label": result.get("figure_label"), "method": result.get("method"), "sources": result.get("sources"),
            "next": ("state the final-points MAE at the cut nearest this season's matchday, and the paired wins, beside the "
                     "simulated table; quote each club's own range, never a sum of ranges")}


def league_backtest_tool(get: Callable[[str], bytes] | None = None, today: Callable[[], dt.date] | None = None) -> Tool:
    return Tool("league_backtest", lambda args, key: _run(args, key, get, today), idempotent=True,
                doc="league_backtest: score the season simulator's final table on past seasons of the league",
                title="League backtest", activity="Scoring the season model on past seasons", observe=observe)


LEAGUE_BACKTEST_SPEC = {
    "name": "league_backtest",
    "title": "League backtest",
    "activity": "Scoring the season model on past seasons",
    "description": "Score the season simulator on the league's past complete seasons (fetched from football-data.co.uk): each "
                   "season is cut after the given matchdays and every club's final points are forecast by points pace, last "
                   "season's points, and the rating model (goals only; with earlier seasons; with earlier seasons and closing "
                   "prices, the simulate_season default). Returns final-points MAE and RMSE per model and cut, Brier scores of "
                   "title, top-four and relegation chances, and paired wins. Quote these beside a simulated final table.",
    "args": {"league": f"optional football-data.co.uk league code, one of {sorted(LEAGUES)}; default E0 (Premier League)",
             "seasons": f"optional number of complete past seasons, 2 to {MAX_SEASONS}; default {DEFAULT_SEASONS}",
             "cuts": "optional list of matchdays to cut each season after, for example [8, 12, 19]"},
}
