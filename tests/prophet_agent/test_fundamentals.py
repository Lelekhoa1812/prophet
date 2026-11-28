"""The fundamentals layer: driver families, scenario mixing, and the league model's signals (owner review, 2026-11-27)."""
from __future__ import annotations

import numpy as np
import pytest

from prophet_agent.forecast import scenarios as sc
from prophet_agent.forecast.sports import (Match, club_key, finishing_gaps, implied_goals, parse_matches, simulate_season,
                                           _poisson_outcomes)
from prophet_agent.research import driver_families as dfam
from prophet_agent.tools.fundamentals_tool import _driver_map


@pytest.mark.parametrize("goal,domain", [
    ("Find Australia's monthly unemployment rate (seasonally adjusted) and forecast it", "labour_market"),
    ("RBA cash rate target over the next 5 years", "policy_rate"),
    ("Research who can win the English Premier League in the 2026/27 season", "sports_league"),
    ("Forecast the gold price month by month", "precious_metal"),
    ("Find the S&P 500 index and forecast it", "equity_index"),
    ("Forecast weekly umbrella sales", "general"),
])
def test_domains_are_read_from_the_goal(goal, domain):
    assert dfam.detect(goal) == domain


def test_plan_fills_queries_for_the_place_and_names_minimums():
    p = _driver_map({"action": "plan", "target": "Vietnam unemployment rate, next 2 years"}, "k")
    assert p["domain"] == "labour_market" and p["place"] == "Vietnam"
    assert any("Vietnam" in q for f in p["families"] for q in f["queries"])
    assert p["minimum"]["tested"] == 2 and "geopolitics" in [f["family"] for f in p["families"]]


def test_record_refuses_findings_without_evidence_or_with_an_unknown_family():
    base = {"action": "record", "target": "RBA cash rate"}
    with pytest.raises(ValueError, match="evidence"):
        _driver_map({**base, "findings": [{"family": "inflation", "signal": "CPI 3%", "kind": "researched",
                                           "evidence": ""}]}, "k")
    with pytest.raises(ValueError, match="not one of"):
        _driver_map({**base, "findings": [{"family": "astrology", "signal": "x", "kind": "researched",
                                           "evidence": "https://a.example/x"}]}, "k")


def test_scenarios_shift_the_distribution_by_their_probability():
    h = 12
    q = {0.1: [9.0] * h, 0.3: [9.6] * h, 0.5: [10.0] * h, 0.7: [10.4] * h, 0.9: [11.0] * h}
    s = sc.validate([{"name": "hike", "probability": 0.5, "shift": 2.0, "start": 4, "ramp": 2, "shift_sd": 0.0, "evidence": "web_research S1"}], h, None)
    m = sc.mix(q, s, n=40000)
    assert abs(m["point"][0] - 10.0) < 0.05                 # before the start: the base
    assert m["quantiles"]["0.9"][-1] > 11.8                 # half the paths moved up by 2
    assert m["scenarios"][0]["median"][-1] == pytest.approx(12.0)
    assert m["scenarios"][0]["median"][3] == pytest.approx(11.0)   # halfway up the two-step ramp
    assert m["weighted_last_shift"] == pytest.approx(1.0)


def test_scenario_validation_names_the_fix():
    with pytest.raises(ValueError, match="at most 1"):
        sc.validate([{"name": "a", "probability": 0.7, "shift": 1, "evidence": "web_research S1"},
                     {"name": "b", "probability": 0.6, "shift": 1, "evidence": "web_research S1"}], 5, None)
    with pytest.raises(ValueError, match="exactly one of shift"):
        sc.validate([{"name": "a", "probability": 0.2}], 5, None)
    got = sc.validate([{"name": "a", "probability": 0.2, "shift_pct": -5, "start": "2027-03", "evidence": "web_research S1"}], 3,
                      ["2027-01-01", "2027-02-01", "2027-03-01"])
    assert got[0]["start"] == 3 and got[0]["kind"] == "pct"


def test_implied_goals_reproduce_the_prices():
    lam, mu = 1.9, 0.8
    h, d, a, o = _poisson_outcomes(lam, mu)
    g = implied_goals((h, d, a), o)
    assert g[0] == pytest.approx(lam, abs=0.05) and g[1] == pytest.approx(mu, abs=0.05)


def test_football_data_columns_give_xg_and_market():
    text = ("﻿Div,Date,HomeTeam,AwayTeam,FTHG,FTAG,HxG,AxG,AvgCH,AvgCD,AvgCA,AvgC>2.5,AvgC<2.5\n"
            "E0,21/08/2026,Arsenal,Coventry,3,0,1.88,0.2,1.2,6.7,14,1.57,2.38\n")
    m = parse_matches(text, "E0.csv")[0]
    assert m.date == "2026-08-21" and m.xg == (1.88, 0.2)
    assert sum(m.market) == pytest.approx(1.0) and m.market[0] > 0.75 and 0.5 < m.over25 < 0.7


def test_club_names_match_across_sites():
    assert club_key("Manchester City FC") == club_key("Man City")
    assert club_key("Nottingham Forest FC") == club_key("Nott'm Forest")
    assert club_key("Brighton & Hove Albion FC") == club_key("Brighton")
    assert club_key("Tottenham Hotspur") == club_key("Spurs")


def _league(rng, teams, strength, n_rounds, start="2025-08-01", with_signals=False):
    import datetime as dt
    out, day = [], dt.date.fromisoformat(start)
    for r in range(n_rounds):
        order = rng.permutation(len(teams))
        for i in range(0, len(teams) - 1, 2):
            h, a = order[i], order[i + 1]
            lam, mu = np.exp(0.25 + strength[h] - strength[a] * 0.5), np.exp(strength[a] - strength[h] * 0.5)
            hg, ag = rng.poisson(lam), rng.poisson(mu)
            ph, pd, pa, _ = _poisson_outcomes(lam, mu)
            out.append(Match(day.isoformat(), teams[h], teams[a], int(hg), int(ag), None,
                             (float(lam), float(mu)) if with_signals else None,
                             (ph, pd, pa) if with_signals else None))
        day += dt.timedelta(days=7)
    return out


def test_a_lucky_start_is_not_read_as_strength_once_history_and_signals_enter():
    rng = np.random.default_rng(3)
    teams = [f"T{i}" for i in range(8)]
    strength = np.linspace(0.5, -0.5, 8)              # T0 strongest, T5 weak
    history = _league(rng, teams, strength, 28, "2024-08-01", with_signals=True)
    current = _league(rng, teams, strength, 8, "2025-08-01", with_signals=True)
    # T5 wins its eight games 4-0: a hot start, against an expected-goal record that says otherwise.
    current = [Match(m.date, m.home, m.away, 4 if m.home == "T5" else 0 if m.away == "T5" else m.home_goals,
                     4 if m.away == "T5" else 0 if m.home == "T5" else m.away_goals, None, m.xg, m.market)
               if "T5" in (m.home, m.away) else m for m in current]
    thin = simulate_season(current, n_sims=2000, backtest=False, xg_weight=0, market_weight=0)
    full = simulate_season(current, n_sims=2000, backtest=False, history=history)
    top = lambda r: r["ratings"][0]["team"]          # ratings are sorted by strength
    assert any("thin ratings" in n for n in thin["notes"])
    assert top(thin) == "T5"                          # eight 4-0 wins read as the best side
    assert top(full) != "T5"                          # history, xG and prices say otherwise
    assert full["settings"]["signals"]["history_matches"] > 0 and full["settings"]["signals"]["xg"] > 0
    assert any(r["team"] == "T5" and r["gap"] > 4 for r in finishing_gaps([m for m in current if m.played]))


def test_market_probabilities_need_their_pricing_url_and_are_labelled():
    base = {"name": "RBA hikes in February", "probability": 0.3, "shift": 0.25}
    with pytest.raises(ValueError, match="URL of the pricing"):
        sc.validate([{**base, "probability_source": "market"}], 6, None)
    with pytest.raises(ValueError, match="market"):
        sc.validate([{**base, "probability_source": "guess"}], 6, None)
    ok = sc.validate([{**base, "probability_source": "market",
                       "evidence": "https://www.asx.com.au/markets/trade-our-derivatives-market/futures-market/rba-rate-tracker"}],
                     6, None)
    assert ok[0]["probability_source"] == "market"
    assert sc.validate([{**base, "evidence": "web_research S2"}], 6, None)[0]["probability_source"] == "judgement"
    with pytest.raises(ValueError, match="evidence must name the tool output"):
        sc.validate([base], 6, None)                                # M10-6: a judgement shift still names its source
    with pytest.raises(ValueError, match="moves nothing"):
        sc.validate([{**base, "shift": 0, "evidence": "web_research S2"}], 6, None)
    mixed = sc.mix({0.1: [3.0] * 6, 0.5: [3.5] * 6, 0.9: [4.0] * 6}, ok)
    assert mixed["scenarios"][0]["probability_source"] == "market"


def test_club_adjustments_move_the_remaining_matches_and_outright_prices_are_compared():
    from prophet_agent.forecast.sports import compare_outright
    from prophet_agent.tools.sports_tool import _adjustments
    rng = np.random.default_rng(5)
    teams = [f"T{i}" for i in range(8)]
    current = _league(rng, teams, np.linspace(0.5, -0.5, 8), 8, "2025-08-01", with_signals=False)
    base = simulate_season(current, n_sims=3000, backtest=False)
    hurt = simulate_season(current, n_sims=3000, backtest=False,
                           adjustments=_adjustments([{"team": "T0", "shift": -0.4, "reason": "striker and keeper out",
                                                      "evidence": "https://club.example/injuries"}]))
    exp = lambda r, t: next(x["expected_final"] for x in r["teams"] if x["team"] == t)
    assert exp(hurt, "T0") < exp(base, "T0") - 2
    assert "T0 -0.40 log goals" in hurt["settings"]["adjustments"][0]
    with pytest.raises(ValueError, match="evidence"):
        _adjustments([{"team": "T0", "shift": -0.1, "reason": "injuries"}])
    with pytest.raises(ValueError, match="-0.5 to 0.5"):
        _adjustments([{"team": "T0", "shift": -2, "reason": "x", "evidence": "https://x.example"}])
    rows = [{"team": "Brighton & Hove Albion", "title": 0.45}, {"team": "Arsenal", "title": 0.40},
            {"team": "Everton", "title": 0.15}]
    cmp = compare_outright(rows, {"title": {"Brighton": 21.0, "Arsenal": 1.5, "Everton": 101.0}}, "https://odds.example")
    assert abs(sum(cmp["market"]["title"].values()) - 1) < 1e-9          # every club priced: the margin is taken out
    gap = {g["team"]: g["gap"] for g in cmp["gaps"]}
    assert gap["Brighton & Hove Albion"] > 0.3 and gap["Arsenal"] < -0.3 and 0.10 <= gap["Everton"] < 0.2
    assert "Brighton & Hove Albion title model 45%" in cmp["note"]
    with pytest.raises(ValueError, match="no club named"):
        compare_outright(rows, {"title": {"Spurs": 5.0}})


def test_plan_gives_ready_lane_briefs_that_cover_every_family_once():
    p = dfam.plan("Forecast the RBA cash rate target over the next 5 years")
    lanes = p["lanes"]
    covered = [f["family"] for f in p["families"] if sum(f["family"] + ":" in ln["task"] for ln in lanes) == 1]
    assert covered == [f["family"] for f in p["families"]]
    assert all(len(ln["name"].split()) <= 4 and ln["deliverable"] and ln["agent_type"] == "researcher" for ln in lanes)
    assert "market pricing" in lanes[-1]["task"] and "The other lanes cover" in lanes[0]["task"]
