"""Sports model tests: standings, the football parser, Dixon-Coles, season simulation, backtests and win_loss mode.

All data here is synthetic and generated with a fixed seed; no figure from these tests is an observed result.
"""
import numpy as np
import pytest
from scipy import optimize, stats

from prophet_agent.forecast import sports as S
from prophet_agent.forecast.sports import Match


def _league(seed: int = 1, n_teams: int = 20, rounds: int = 2, spread: float = 0.35, home: float = 0.25, rho: float = -0.1):
    """Double round robin with known attack and defence; returns (matches, names, true attack, true defence)."""
    rng = np.random.default_rng(seed)
    names = [f"Team {i:02d}" for i in range(n_teams)]
    att = rng.normal(0, spread, n_teams)
    dfn = rng.normal(0, spread, n_teams)
    out = []
    for _ in range(rounds):
        for h in range(n_teams):
            for a in range(n_teams):
                if h == a:
                    continue
                lam = np.exp(att[h] - dfn[a] + home)
                mu = np.exp(att[a] - dfn[h])
                grid = S.score_grid(lam, mu, rho)
                k = rng.choice(grid.size, p=grid.ravel())
                hg, ag = divmod(int(k), grid.shape[1])
                out.append(Match(None, names[h], names[a], hg, ag))
    return out, names, att, dfn


def _strength_corr(model, att, dfn):
    T = model.n_teams
    fitted = model.theta[:T] - model.theta[T:2 * T]
    true = att - dfn
    return stats.spearmanr(true, fitted).statistic


# ---------------------------------------------------------------- standings and parsing

def test_standings_points_goal_difference_and_tie_breaks():
    m = [Match(None, "A", "B", 2, 0), Match(None, "C", "D", 1, 0), Match(None, "A", "C", 0, 0),
         Match(None, "B", "D", 3, 1), Match(None, "A", "D", 1, 1), Match(None, "B", "C", 0, 2)]
    table = S.standings(m)
    by = {r["team"]: r for r in table}
    # A: win (3), draw (1), draw (1) = 5. C: draw (1), win (3), win (3) = 7. B: win (3) = 3. D: draw (1) = 1.
    assert by["A"]["points"] == 5
    assert by["A"]["won"] == 1 and by["A"]["drawn"] == 2 and by["A"]["lost"] == 0
    assert by["A"]["goals_for"] == 3 and by["A"]["goals_against"] == 1 and by["A"]["goal_difference"] == 2
    assert {t: r["points"] for t, r in by.items()} == {"A": 5, "B": 3, "C": 7, "D": 1}
    assert [r["team"] for r in table] == ["C", "A", "B", "D"]
    assert [r["position"] for r in table] == [1, 2, 3, 4]


def test_standings_tie_break_by_goal_difference_then_goals_for():
    m = [Match(None, "X", "Y", 1, 0), Match(None, "Z", "W", 3, 0)]
    m += [Match(None, "X", "W", 0, 0), Match(None, "Z", "Y", 0, 0)]
    m += [Match(None, "X", "Z", 0, 0), Match(None, "Y", "W", 0, 0)]
    table = S.standings(m)
    order = [r["team"] for r in table]
    assert order.index("Z") < order.index("X")          # Z: +3 goal difference; X: +1 on equal points
    # P and R both have 4 points; R has the better goal difference (+3 against +1), so R is above P
    tied = S.standings([Match(None, "P", "Q", 2, 1), Match(None, "R", "Q", 3, 0), Match(None, "P", "R", 0, 0)])
    assert {r["team"]: r["points"] for r in tied} == {"P": 4, "R": 4, "Q": 0}
    assert [r["team"] for r in tied] == ["R", "P", "Q"]


def test_standings_custom_points_rule():
    m = [Match(None, "A", "B", 1, 0), Match(None, "B", "C", 1, 1)]
    by = {r["team"]: r["points"] for r in S.standings(m, points=(2, 1, 0))}
    assert by == {"A": 2, "B": 1, "C": 1}


def test_football_json_parser_reads_played_and_unplayed_matches():
    doc = {"name": "Sample", "matches": [
        {"round": "Matchday 1", "date": "2026-08-15", "team1": "Alpha FC", "team2": "Beta FC",
         "score": {"ht": [1, 0], "ft": [2, 1]}},
        {"round": "Matchday 1", "date": "2026-08-16", "team1": "Gamma FC", "team2": "Delta FC", "score": {"ft": [0, 0]}},
        {"round": "Matchday 2", "date": "2026-08-22", "team1": "Beta FC", "team2": "Gamma FC"},
    ]}
    matches = S.parse_football_json(doc)
    assert [m.played for m in matches] == [True, True, False]
    assert (matches[0].home, matches[0].away, matches[0].home_goals, matches[0].away_goals) == ("Alpha FC", "Beta FC", 2, 1)
    assert matches[2].home_goals is None and matches[2].date == "2026-08-22"
    assert S.parse_matches(__import__("json").dumps(doc), "en.1.json")[0].home == "Alpha FC"


def test_football_json_parser_refuses_bad_documents():
    with pytest.raises(ValueError, match="not a football.json"):
        S.parse_football_json({"rows": []})
    with pytest.raises(ValueError, match="score.ft"):
        S.parse_football_json({"matches": [{"team1": "A", "team2": "B", "score": {"ft": [1]}}]})


def test_csv_reader_reads_fixtures_and_refuses_bad_rows():
    text = "date,home,away,home_goals,away_goals,group\n2026-08-01,A,B,2,1,East\n2026-08-08,B,A,,,East\n"
    m = S.read_matches_csv(text)
    assert m[0].played and m[0].group == "East" and not m[1].played
    with pytest.raises(ValueError, match="both scores or neither"):
        S.read_matches_csv("home,away,home_goals,away_goals\nA,B,1,\n")
    with pytest.raises(ValueError, match="cannot play itself"):
        S.read_matches_csv("home,away\nA,A\n")
    with pytest.raises(ValueError, match="columns home and away"):
        S.read_matches_csv("team,opponent\nA,B\n")


# ---------------------------------------------------------------- Dixon-Coles

def test_low_score_correction_changes_only_the_four_low_cells():
    lam, mu, rho = 1.3, 1.1, -0.15
    base = S.score_grid(lam, mu, 0.0)
    corr = S.score_grid(lam, mu, rho)
    ratio = corr / base
    # the correction multiplies each cell by tau and one common normalising constant c, so ratio = c * tau on the
    # four low cells and c everywhere else
    c = ratio[2, 2]
    assert ratio[1, 1] / c == pytest.approx(1 - rho, rel=1e-9)
    assert ratio[0, 0] / c == pytest.approx(1 - lam * mu * rho, rel=1e-9)
    assert ratio[0, 1] / c == pytest.approx(1 + lam * rho, rel=1e-9)
    assert ratio[1, 0] / c == pytest.approx(1 + mu * rho, rel=1e-9)
    assert np.allclose(ratio[2:, :] / c, 1.0) and np.allclose(ratio[:, 2:] / c, 1.0)
    assert corr[0, 0] > base[0, 0]          # rho below zero raises 0-0 relative to the independent model


def test_score_grid_without_correction_is_independent_poisson_and_sums_to_one():
    grid = S.score_grid(1.4, 0.9, 0.0)
    indep = np.outer(stats.poisson.pmf(np.arange(13), 1.4), stats.poisson.pmf(np.arange(13), 0.9))
    indep /= indep.sum()
    assert np.allclose(grid, indep, atol=1e-12)
    h, d, a = S.outcome_probs(grid)
    assert h + d + a == pytest.approx(1.0, abs=1e-12)


def test_gradient_matches_numerical_gradient():
    matches, names, _, _ = _league(seed=3, n_teams=6, rounds=2)
    T = len(names)
    idx = {t: i for i, t in enumerate(names)}
    hi = np.array([idx[m.home] for m in matches])
    ai = np.array([idx[m.away] for m in matches])
    x = np.array([m.home_goals for m in matches], float)
    y = np.array([m.away_goals for m in matches], float)
    w = np.linspace(0.5, 1.0, len(matches))
    rng = np.random.default_rng(0)
    theta = np.concatenate([rng.normal(0, 0.2, 2 * T), [0.2, -0.05]])
    args = (hi, ai, x, y, w, T, 0.25)
    _, g = S._objective(theta, *args)
    num = optimize.approx_fprime(theta, lambda t: S._objective(t, *args)[0], 1e-6)
    assert np.allclose(g, num, rtol=1e-4, atol=1e-4)


def test_fit_recovers_the_order_of_team_strengths():
    matches, names, att, dfn = _league(seed=2, spread=0.35)
    model = S.fit_dixon_coles(matches, teams=names)
    assert model.converged
    assert _strength_corr(model, att, dfn) > 0.8
    assert model.home_advantage == pytest.approx(0.25, abs=0.12)
    assert -0.4 < model.rho < 0.2


def test_ridge_keeps_a_team_with_few_matches_from_being_extreme():
    matches, names, _, _ = _league(seed=4, n_teams=12, rounds=2)
    sparse = names[0]
    kept = [m for m in matches if sparse not in (m.home, m.away)]
    # the sparse team: four matches, all heavy wins
    extra = [Match(None, sparse, names[i], 5, 0) for i in range(1, 5)]
    shrunk = S.fit_dixon_coles(kept + extra, teams=names, prior_sd=0.5)
    loose = S.fit_dixon_coles(kept + extra, teams=names, prior_sd=10.0)
    assert abs(shrunk.theta[0]) < abs(loose.theta[0])     # the prior pulls the sparse team's attack toward the mean
    assert abs(shrunk.theta[0]) < 1.0


def test_fit_refuses_too_few_matches_with_a_reason():
    matches, names, _, _ = _league(seed=5, n_teams=10, rounds=1)
    with pytest.raises(ValueError, match="too few matches to rate the teams"):
        S.fit_dixon_coles(matches[:20], teams=names)


def test_sampling_follows_the_corrected_joint_distribution():
    rng = np.random.default_rng(7)
    lam, mu, rho = 1.2, 0.9, -0.2
    n = 200_000
    x, y = S._sample_scores(np.full(n, lam), np.full(n, mu), np.full(n, rho), rng)
    grid = S.score_grid(lam, mu, rho)
    observed = np.mean((x == 0) & (y == 0))
    assert observed == pytest.approx(grid[0, 0], abs=0.003)
    assert np.mean((x == 1) & (y == 1)) == pytest.approx(grid[1, 1], abs=0.003)


# ---------------------------------------------------------------- season simulation

def _split_league(seed=1, spread=0.35, played_count=300):
    matches, names, att, dfn = _league(seed=seed, spread=spread)
    played, future = matches[:played_count], [Match(None, m.home, m.away, None, None) for m in matches[played_count:]]
    return played, future, names, att, dfn


def test_simulation_is_reproducible_with_a_seed():
    played, future, names, _, _ = _split_league()
    a = S.simulate_season(played, future, n_sims=2000, seed=11, backtest=False)
    b = S.simulate_season(played, future, n_sims=2000, seed=11, backtest=False)
    c = S.simulate_season(played, future, n_sims=2000, seed=12, backtest=False)
    assert a["teams"] == b["teams"]
    assert a["teams"] != c["teams"]


def test_probabilities_sum_to_one_title_and_the_number_of_places_for_each_outcome():
    played, future, names, _, _ = _split_league()
    out = S.simulate_season(played, future, n_sims=3000, seed=0, backtest=False)
    rows = out["teams"]
    assert sum(r["title"] for r in rows) == pytest.approx(1.0, abs=1e-9)
    assert sum(r["top_places"] for r in rows) == pytest.approx(4.0, abs=1e-9)
    assert sum(r["relegation"] for r in rows) == pytest.approx(3.0, abs=1e-9)
    for r in rows:
        assert 0 <= r["title"] <= r["top_places"] <= 1
        assert r["final"]["p10"] <= r["final"]["median"] <= r["final"]["p90"]


def test_simulated_final_points_are_consistent_with_played_points_and_remaining_games():
    played, future, names, _, _ = _split_league(played_count=300)
    out = S.simulate_season(played, future, n_sims=500, seed=0, backtest=False)
    table = {r["team"]: r for r in S.standings(played)}
    for row in out["teams"]:
        t = row["team"]
        remaining = sum(1 for m in future if t in (m.home, m.away))
        assert row["current"] == table[t]["points"]
        assert row["final"]["p10"] >= row["current"] - 1e-9          # points never fall in a simulated season
        assert row["final"]["p90"] <= row["current"] + 3 * remaining + 1e-9


def test_default_remaining_schedule_is_a_double_round_robin_completed_from_results():
    # 8 teams, double round robin = 56 ordered pairs per round; the first 40 entries (distinct pairs) are played
    matches, names, _, _ = _league(seed=6, n_teams=8, rounds=2)
    out = S.simulate_season(matches[:40], None, n_sims=200, seed=0, backtest=False)
    assert out["settings"]["n_remaining"] == 16
    assert any("double round robin" in n for n in out["notes"])


def test_points_paths_have_one_entry_per_game_and_measured_games_come_first():
    played, future, names, _, _ = _split_league()
    out = S.simulate_season(played, future, n_sims=500, seed=0, teams_of_interest=[names[0]], backtest=False)
    path = out["points_paths"][names[0]]
    measured = [e for e in path if e["measured"]]
    simulated = [e for e in path if not e["measured"]]
    assert measured and simulated
    assert path == measured + simulated
    assert path[-1]["game"] == len(measured) + len(simulated)
    assert all(e["p10"] <= e["median"] <= e["p90"] for e in simulated)


def test_simulation_refuses_unknown_sport_and_small_sample():
    played, future, _, _, _ = _split_league()
    with pytest.raises(ValueError, match="sport must be"):
        S.simulate_season(played, future, sport="cricket", n_sims=100)
    with pytest.raises(ValueError, match="simulations must be"):
        S.simulate_season(played, future, n_sims=5)


# ---------------------------------------------------------------- backtests

def test_football_backtest_beats_equal_probabilities_on_a_strong_synthetic_league():
    matches, names, _, _ = _league(seed=8, n_teams=20, rounds=2, spread=0.6)
    bt = S.backtest_football(matches, names, half_life_days=None, prior_sd=0.5)
    assert bt["origins"] >= 3
    dc, eq = bt["models"]["dixon_coles"], bt["models"]["equal"]
    assert dc["rps"] < eq["rps"]
    assert dc["brier"] < eq["brier"]
    versus_equal = [c for c in bt["comparisons"] if c["comparison"] == "dixon_coles vs equal (rps)"][0]
    assert versus_equal["verdict"] == "model_better"
    assert bt["models"]["dixon_coles"]["brier"] == pytest.approx(
        bt["models"]["dixon_coles"]["brier_home"] + bt["models"]["dixon_coles"]["brier_draw"]
        + bt["models"]["dixon_coles"]["brier_away"])


def test_rps_is_zero_for_a_perfect_forecast_and_brier_is_two_for_a_wrong_certain_one():
    p = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])       # home certain; away certain
    o = np.array([2, 2])                                    # both home wins
    rps, be = S._rps_brier(p, o)
    assert rps[0] == pytest.approx(0.0) and rps[1] == pytest.approx(1.0)
    assert be[1].sum() == pytest.approx(2.0)


def test_winloss_backtest_beats_equal_odds_on_a_strong_margin_effect():
    rng = np.random.default_rng(9)
    n_teams = 20
    names = [f"Side {i:02d}" for i in range(n_teams)]
    rating = rng.normal(0, 8, n_teams)
    matches = []
    for _ in range(3):
        for h in range(n_teams):
            for a in range(n_teams):
                if h != a:
                    margin = 3 + rating[h] - rating[a] + rng.normal(0, 11)
                    hg = int(round(100 + margin / 2))
                    ag = int(round(100 - margin / 2))
                    if hg == ag:
                        hg += 1
                    matches.append(Match(None, names[h], names[a], hg, ag))
    bt = S.backtest_winloss(matches, names, half_life_days=None, prior_sd=10.0)
    assert bt["models"]["margin_model"]["brier"] < bt["models"]["equal"]["brier"]
    assert bt["models"]["margin_model"]["log_loss"] < bt["models"]["equal"]["log_loss"]
    assert [c["verdict"] for c in bt["comparisons"] if "equal (brier)" in c["comparison"]] == ["model_better"]


# ---------------------------------------------------------------- win_loss mode

def _winloss_league(seed=10, n_teams=12, games_each_way=2, spread=8.0, groups=False):
    rng = np.random.default_rng(seed)
    names = [f"Club {i:02d}" for i in range(n_teams)]
    rating = rng.normal(0, spread, n_teams)
    matches = []
    for _ in range(games_each_way):
        for h in range(n_teams):
            for a in range(n_teams):
                if h != a:
                    margin = 3 + rating[h] - rating[a] + rng.normal(0, 11)
                    hg, ag = int(round(100 + margin / 2)), int(round(100 - margin / 2))
                    if hg == ag:
                        hg += 1
                    # the group column is the home team's group, as the tool documents
                    matches.append(Match(None, names[h], names[a], hg, ag,
                                         ("East" if h < n_teams // 2 else "West") if groups else None))
    return matches, names, rating


def test_winloss_fit_recovers_rating_order_and_refuses_draws():
    matches, names, rating = _winloss_league(seed=10, games_each_way=3)
    model = S.fit_margin(matches, teams=names, prior_sd=8.0)
    fitted = model.beta[1:]
    assert stats.spearmanr(rating, fitted).statistic > 0.85
    assert model.sigma > 5
    with pytest.raises(ValueError, match="no draws"):
        S.fit_margin(matches + [Match(None, names[0], names[1], 50, 50)], teams=names)


def test_winloss_fixture_mode_reports_per_team_wins_in_range_and_never_a_sum():
    matches, names, _ = _winloss_league(seed=11, games_each_way=2)
    played, future = matches[:200], [Match(None, m.home, m.away, None, None) for m in matches[200:260]]
    out = S.simulate_season(played, future, sport="win_loss", n_sims=2000, seed=0, backtest=False)
    assert out["metric"] == "wins" and out["settings"]["n_remaining"] == 60
    for row in out["teams"]:
        assert 0 <= row["final"]["p10"] <= row["final"]["median"] <= row["final"]["p90"]
        assert row["final"]["p90"] <= 4 * (len(names) - 1)     # each team plays 4 matches per opponent in this league
        assert 0 <= row["playoff"] <= 1
    assert "do not add" in out["reading"]
    assert all(p["p10"] <= p["p90"] for path in out["points_paths"].values() for p in path if not p["measured"])


def test_winloss_schedule_mode_uses_games_per_team_and_the_opponent_mix():
    matches, names, _ = _winloss_league(seed=12, games_each_way=1)
    out = S.simulate_season(matches[:100], None, sport="win_loss", games_per_team=30, n_sims=1500, seed=0, backtest=False)
    for row in out["teams"]:
        assert row["final"]["p90"] <= 30 and row["final"]["p10"] >= row["current"]
    assert any("opponent mix" in n for n in out["notes"])
    assert out["settings"]["n_remaining"] == sum(max(0, 30 - sum(1 for m in matches[:100] if t in (m.home, m.away)))
                                                 for t in names)


def test_winloss_schedule_mode_needs_games_per_team_when_there_is_no_schedule():
    matches, names, _ = _winloss_league(seed=13, games_each_way=1)
    with pytest.raises(ValueError, match="games_per_team"):
        S.simulate_season(matches, None, sport="win_loss", n_sims=100, backtest=False)


def test_playoff_line_per_group_sums_to_the_places_in_each_group():
    matches, names, _ = _winloss_league(seed=14, games_each_way=2, groups=True)
    played, future = matches[:200], [Match(None, m.home, m.away, None, None, m.group) for m in matches[200:240]]
    out = S.simulate_season(played, future, sport="win_loss", n_sims=1500, seed=0, playoff_places=2, backtest=False)
    for group in ("East", "West"):
        total = sum(r["playoff"] for r in out["teams"] if r.get("group") == group)
        assert total == pytest.approx(2.0, abs=1e-9)
