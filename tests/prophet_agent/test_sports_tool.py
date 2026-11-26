"""The simulate_season tool end to end, offline: uploaded files in, compact view out, catalogue and modeler role wiring."""
import json
from pathlib import Path

import numpy as np
import pytest

from prophet_agent.forecast import sports as S
from prophet_agent.runtime import agents
from prophet_agent.tools import default_tools, tool_catalogue
from prophet_agent.tools.sports_tool import SIMULATE_SEASON_SPEC, observe_simulate, simulate_season_tool


def _write_league(tmp_path, n_teams=8, rounds=2, played_rounds=1):
    rng = np.random.default_rng(21)
    names = [f"Side {i}" for i in range(n_teams)]
    att = rng.normal(0, 0.4, n_teams)
    dfn = rng.normal(0, 0.4, n_teams)
    rows = ["date,home,away,home_goals,away_goals"]
    fixtures = ["date,home,away"]
    day = 0
    for rnd in range(rounds):
        for h in range(n_teams):
            for a in range(n_teams):
                if h == a:
                    continue
                day += 1
                date = f"2026-{1 + day // 28:02d}-{1 + day % 28:02d}"
                if rnd < played_rounds:
                    grid = S.score_grid(np.exp(att[h] - dfn[a] + 0.2), np.exp(att[a] - dfn[h]), -0.1)
                    k = int(rng.choice(grid.size, p=grid.ravel()))
                    hg, ag = divmod(k, grid.shape[1])
                    rows.append(f"{date},{names[h]},{names[a]},{hg},{ag}")
                else:
                    fixtures.append(f"{date},{names[h]},{names[a]}")
    results = tmp_path / "results.csv"
    results.write_text("\n".join(rows) + "\n", encoding="utf-8")
    fix = tmp_path / "fixtures.csv"
    fix.write_text("\n".join(fixtures) + "\n", encoding="utf-8")
    return {"results.csv": results, "fixtures.csv": fix}, names


def _call(tool, args):
    return tool.fn(args, "test-key")


def test_tool_runs_end_to_end_on_uploaded_csv_files(tmp_path):
    files, names = _write_league(tmp_path)
    tool = simulate_season_tool(files)
    out = _call(tool, {"results": "results.csv", "fixtures": "fixtures.csv", "simulations": 800, "seed": 3,
                       "teams": [names[0], "Nobody FC"], "backtest": True})
    assert out["sport"] == "football" and out["figure_label"] == "simulated from fitted ratings"
    assert out["settings"]["n_sims"] == 800 and out["settings"]["seed"] == 3
    assert out["settings"]["n_remaining"] == 56            # the second round of 8 teams in the fixtures file
    assert len(out["teams"]) == len(names)
    assert sum(r["title"] for r in out["teams"]) == pytest.approx(1.0)
    assert set(out["points_paths"]) == {names[0]}
    assert "teams not in the file were ignored: Nobody FC" in out["notes"]
    assert out["backtest"]["models"]["equal"]["rps"] > 0
    assert "do not add" in out["reading"]


def test_tool_is_reproducible_for_the_same_seed(tmp_path):
    files, _ = _write_league(tmp_path)
    tool = simulate_season_tool(files)
    a = _call(tool, {"results": "results.csv", "simulations": 500, "seed": 9, "backtest": False})
    b = _call(tool, {"results": "results.csv", "simulations": 500, "seed": 9, "backtest": False})
    assert a["teams"] == b["teams"]


def test_tool_reads_an_openfootball_json_file(tmp_path):
    # 4 teams, double round robin (24 matches); the last two have no score and are the remaining fixtures
    teams = ["Alpha", "Beta", "Gamma", "Delta"]
    entries = [(h, a) for h in teams for a in teams if h != a] * 2
    matches = []
    for i, (h, a) in enumerate(entries):
        item = {"round": f"Matchday {i // 6 + 1}", "date": f"2026-08-{1 + i:02d}", "team1": h, "team2": a}
        if i < len(entries) - 2:
            item["score"] = {"ft": [1 + i % 2, i % 3]}
        matches.append(item)
    path = tmp_path / "en.1.json"
    path.write_text(json.dumps({"name": "Sample", "matches": matches}), encoding="utf-8")
    out = _call(simulate_season_tool({"en.1.json": path}),
                {"results": "en.1.json", "simulations": 200, "backtest": False})
    assert out["settings"]["n_played"] == 22 and out["settings"]["n_remaining"] == 2
    assert {r["team"] for r in out["teams"]} == set(teams)


def test_tool_win_loss_with_games_per_team_and_observe_view(tmp_path):
    rng = np.random.default_rng(5)
    names = [f"Franchise {i}" for i in range(6)]
    rows = ["date,home,away,home_goals,away_goals,group"]
    for h in range(6):
        for a in range(6):
            if h != a:
                hg = int(100 + rng.normal(0, 10))
                ag = int(100 + rng.normal(0, 10))
                if hg == ag:
                    hg += 1
                rows.append(f"2026-01-01,{names[h]},{names[a]},{hg},{ag},{'East' if h < 3 else 'West'}")
    path = tmp_path / "games.csv"
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    tool = simulate_season_tool({"games.csv": path})
    out = _call(tool, {"sport": "win_loss", "results": "games.csv", "games_per_team": 20, "simulations": 300,
                       "playoff_places": 2, "backtest": False})
    assert out["metric"] == "wins"
    assert all(r["final"]["p90"] <= 20 for r in out["teams"])
    view = observe_simulate(out)
    assert len(view["standings"]) <= 20 and "note" in view and "teams" in view


@pytest.mark.parametrize("args, message", [
    ({"simulations": 100}, "exactly one of results"),
    ({"results": "x.csv", "source": {"run_id": "0" * 32, "file": "out/x.csv"}}, "exactly one of results"),
    ({"results": "missing.csv"}, "must be one of the uploaded file names"),
    ({"results": "results.csv", "sport": "cricket"}, "sport must be"),
    ({"results": "results.csv", "surprise": 1}, "unknown arguments"),
    ({"results": "results.csv", "simulations": 3}, "simulations must be a whole number"),
    ({"results": "results.csv", "teams": "Side 0"}, "teams must be a list"),
    ({"results": "results.csv", "half_life_days": -1}, "half_life_days must be a positive number"),
    ({"results": "results.csv", "backtest": "yes"}, "backtest must be true or false"),
])
def test_tool_refuses_bad_arguments_with_a_reason(tmp_path, args, message):
    files, _ = _write_league(tmp_path, n_teams=4, rounds=1, played_rounds=1)
    with pytest.raises(ValueError, match=message):
        _call(simulate_season_tool(files), args)


def test_tool_refuses_a_win_loss_run_with_no_schedule(tmp_path):
    files, _ = _write_league(tmp_path, n_teams=4, rounds=1, played_rounds=1)
    with pytest.raises(ValueError, match="games_per_team"):
        _call(simulate_season_tool(files), {"sport": "win_loss", "results": "results.csv", "simulations": 50})


def test_tool_is_in_the_catalogue_and_the_modeler_role():
    cat = tool_catalogue()
    assert "simulate_season" in cat
    assert cat["simulate_season"].doc.startswith(SIMULATE_SEASON_SPEC["description"][:40])
    assert "simulate_season" in default_tools({"a.csv": Path("a.csv")})
    modeler = agents.get("modeler")
    assert modeler is not None and "simulate_season" in modeler.tools
