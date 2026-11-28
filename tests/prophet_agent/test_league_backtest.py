"""Mission 10, M10-3: the season simulator is scored on past seasons' final tables, and the harness runs it for a league goal.

Seasons here are synthetic (seeded double round robins from fixed team strengths), so no figure is an observed result.
"""
import datetime as dt

import numpy as np
import pytest

from prophet_agent.forecast import sports as sp
from prophet_agent.runtime import goal
import importlib
lb = importlib.import_module("prophet_agent.tools.league_backtest_tool")

TEAMS = [f"Club {c}" for c in "ABCDEFGHIJ"]


def _season(year: int, strength: dict[str, float], seed: int) -> list[sp.Match]:
    rng = np.random.default_rng(seed)
    pairs = [(h, a) for h in strength for a in strength if h != a]
    rng.shuffle(pairs)
    start = dt.date(year, 8, 10)
    out = []
    for k, (h, a) in enumerate(pairs):
        lam = np.exp(0.25 + strength[h] - strength[a] + 0.1)
        mu = np.exp(0.25 + strength[a] - strength[h])
        out.append(sp.Match((start + dt.timedelta(days=k // 5 * 7)).isoformat(), h, a, int(rng.poisson(lam)),
                            int(rng.poisson(mu))))
    return out


def _seasons(n=4):
    strength = {t: 0.6 - 0.12 * i for i, t in enumerate(TEAMS)}
    return [(f"{2020 + i}/{(21 + i) % 100:02d}", _season(2020 + i, strength, i)) for i in range(n)]


def test_rating_model_beats_pace_on_stable_strengths():
    res = sp.season_backtest(_seasons(), cuts=(8, 12), n_sims=200)
    assert res["seasons_scored"] == ["2021/22", "2022/23", "2023/24"]
    by = {(s["cut"], s["model"]): s for s in res["summary"]}
    # Strengths do not change between seasons, so earlier seasons must help the early cut.
    assert by[(8, "goals_history")]["mae"] < by[(8, "goals_only")]["mae"]
    assert by[(8, "goals_history")]["mae"] < by[(8, "pace")]["mae"]
    assert {"brier_title", "brier_top4", "brier_relegation"} <= set(by[(12, "history_market")])
    assert "brier_title" not in by[(8, "pace")]
    assert all(isinstance(p["mae_gain_pct"], float) for p in res["paired"])


def test_partial_season_is_not_scored_and_one_season_is_refused():
    seasons = _seasons(3)
    seasons[-1] = (seasons[-1][0], seasons[-1][1][:40])
    res = sp.season_backtest(seasons, cuts=(8,), n_sims=100)
    assert res["seasons_scored"] == ["2021/22"]
    with pytest.raises(ValueError, match="at least two"):
        sp.season_backtest(seasons[:1])


def test_promoted_clubs_are_those_missing_from_the_latest_history_season():
    hist = _season(2023, {t: 0.0 for t in TEAMS[:8] + ["Old X", "Old Y"]}, 1)
    assert sp.promoted_clubs(hist, TEAMS) == ["Club I", "Club J"]
    assert sp.promoted_clubs([], TEAMS) == []


def test_promoted_prior_moves_a_new_club_below_the_mean():
    played = _season(2024, {t: 0.0 for t in TEAMS}, 3)[:50]
    plain = sp.fit_dixon_coles(played, teams=TEAMS)
    shifted = sp.fit_dixon_coles(played, teams=TEAMS, prior_means={"Club J": sp.PROMOTED_PRIOR})
    j, T = TEAMS.index("Club J"), len(TEAMS)
    assert shifted.theta[j] < plain.theta[j] and shifted.theta[T + j] < plain.theta[T + j]


def test_tool_fetches_complete_seasons_only_and_reports_missing():
    seasons = dict(_seasons(4))
    csv_of = {}
    for label, ms in seasons.items():
        rows = ["Div,Date,HomeTeam,AwayTeam,FTHG,FTAG"] + [
            f"E0,{dt.date.fromisoformat(m.date):%d/%m/%Y},{m.home},{m.away},{m.home_goals},{m.away_goals}" for m in ms]
        csv_of[lb.season_code(int(label[:4]))] = "\n".join(rows).encode()
    asked = []

    def get(url):
        asked.append(url)
        code = url.split("/")[-2]
        if code not in csv_of:
            raise lb.market.MarketError("HTTP 404")
        return csv_of[code]

    res = lb._run({"seasons": 5, "cuts": [8]}, "k", get=get, today=lambda: dt.date(2024, 11, 28))
    assert [u.split("/")[-2] for u in asked] == ["1920", "2021", "2122", "2223", "2324"]
    assert all(u.startswith("https://football-data.co.uk/") for u in asked)
    assert res["missing"][0]["season"] == "2019/20"
    assert res["seasons_scored"] == ["2021/22", "2022/23", "2023/24"]
    view = lb.observe(res)
    assert view["summary"] and view["paired"] and "never a sum" in view["next"]
    with pytest.raises(ValueError, match="league must be"):
        lb._run({"league": "XX"}, "k", get=get)


def test_complete_seasons_follow_the_august_start():
    assert lb.complete_seasons(dt.date(2026, 11, 28), 2) == [2024, 2025]
    assert lb.complete_seasons(dt.date(2026, 5, 1), 2) == [2023, 2024]


def test_harness_runs_the_league_backtest_once_for_a_football_goal():
    tools = {"league_backtest": object()}
    g = "Predict the final 2026/27 English Premier League table"
    step = goal._league_backtest_step(g, [], tools)
    assert step["tool"] == "league_backtest" and step["args"] == {"league": "E0"}
    done = [{"kind": "tool", "tool": "league_backtest", "error": None}]
    assert goal._league_backtest_step(g, done, tools) is None
    assert goal._league_backtest_step("Forecast the NBA standings", [], tools) is None
    assert goal._league_backtest_step(g, [], {}) is None


def test_pi_ratings_move_towards_the_winner_and_feed_the_season_backtest():
    """Mission 10 batch 2: pi-ratings (Constantinou and Fenton 2013) as a rating candidate in the season backtest."""
    from prophet_agent.forecast import sports as S
    ms = [S.Match(f"2025-08-{d:02d}", "A", "B", 3, 0) for d in range(1, 11)]
    R, seen = S.pi_ratings(ms)
    assert R["A"][0] > 0 > R["B"][1] and seen[0] == (0.0, 1) and seen[-1][0] > 0
