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
    assert "standings" not in view and len(view["teams"]) <= 6 and "note" in view   # standings stay in the full result


@pytest.mark.parametrize("args, message", [
    ({"simulations": 100}, "exactly one of results"),
    ({"results": "x.csv", "source": {"run_id": "0" * 32, "file": "out/x.csv"}}, "exactly one of results"),
    ({"results": "missing.csv"}, "must be one of the mission's file names"),
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


# --- open data in: football.json landed by download_web_data, football-data.co.uk CSV ---------------------------------

def _football_json(names, played=12):
    matches, k = [], 0
    for h in names:
        for a in names:
            if h == a:
                continue
            k += 1
            entry = {"round": f"Matchday {1 + k // 4}", "date": f"2026-{8 + k // 30:02d}-{1 + k % 28:02d}",
                     "team1": h, "team2": a}
            if k <= played:
                entry["score"] = {"ft": [k % 3, (k + 1) % 2]} if k % 2 else [k % 2, k % 3]   # both live spellings
            matches.append(entry)
    return {"name": "Synthetic League 2026/27", "matches": matches}


def test_a_bare_score_list_counts_as_the_full_time_score():
    """2026-27 en.1.json (live) gives some results as "score": [0, 0]; they are played matches, not fixtures."""
    doc = _football_json(["A", "B", "C", "D"], played=6)
    played = [m for m in S.parse_football_json(doc) if m.played]
    assert len(played) == 6


def test_football_data_co_uk_columns_and_day_month_dates_are_read():
    text = "Div,Date,HomeTeam,AwayTeam,FTHG,FTAG\nE0,22/08/26,Arsenal,Spurs,2,1\nE0,23/08/2026,Spurs,Arsenal,,\n"
    first, second = S.read_matches_csv(text)
    assert (first.date, first.home, first.away, first.home_goals, first.away_goals) == ("2026-08-22", "Arsenal", "Spurs", 2, 1)
    assert second.date == "2026-08-23" and not second.played


def test_download_web_data_lands_football_json_and_simulate_season_reads_it(tmp_path, monkeypatch):
    """The open-research path: the planner downloads the league file, then simulates from it in the same mission."""
    import httpx
    from prophet_agent.datasets import DatasetStore
    from prophet_agent.opendata import fetch
    from prophet_agent.research import netguard
    from prophet_agent.tools.opendata_tools import OpenDataContext

    names = [f"Club {i}" for i in range(6)]
    body = json.dumps(_football_json(names, played=26)).encode()
    handler = lambda request: httpx.Response(200, headers={"content-type": "application/json"}, content=body)  # noqa: E731
    monkeypatch.setattr(fetch, "open_client", lambda timeout_s: httpx.Client(transport=httpx.MockTransport(handler),
                                                                             follow_redirects=False))
    monkeypatch.setattr(netguard, "_default_resolver", lambda host, port: ["93.184.216.34"])
    store = DatasetStore(tmp_path / "datasets")
    tools = default_tools(opendata=OpenDataContext(store=store, attach=lambda _id: None))
    landed = tools["download_web_data"].fn({"url": "https://raw.example.org/2026-27/en.1.json", "name": "league.json"},
                                           "k1")
    file_name = landed["sandbox_file"].split("/", 1)[1]
    out = tools["simulate_season"].fn({"results": file_name, "simulations": 200, "backtest": False}, "k2")
    teams = {row["team"] for row in out["standings"]}
    assert teams == set(names)


def test_earlier_seasons_fit_the_ratings_only(tmp_path):
    """Early season: too few current matches to rate the clubs, so last season's results fit the ratings, and the table
    and simulation still start from this season. A club absent this season is not in the simulated table."""
    names = [f"Club {i}" for i in range(6)]
    current = _football_json(names, played=8)
    last = _football_json(names[:5] + ["Gone FC"], played=30)
    for match in last["matches"]:
        match["date"] = match["date"].replace("2026-", "2025-")      # last season's dates
    (tmp_path / "now.json").write_text(json.dumps(current), encoding="utf-8")
    (tmp_path / "last.json").write_text(json.dumps(last), encoding="utf-8")
    tool = simulate_season_tool({"now.json": tmp_path / "now.json", "last.json": tmp_path / "last.json"})
    with pytest.raises(ValueError, match="too few matches"):
        tool.fn({"results": "now.json", "simulations": 100, "backtest": False}, "k1")
    out = tool.fn({"results": "now.json", "history": ["last.json"], "simulations": 100, "backtest": False}, "k2")
    assert {row["team"] for row in out["teams"]} == set(names)
    assert out["settings"]["n_played"] == 8
    assert any("fitted the ratings only" in note for note in out["notes"])


def test_fixturedownload_csv_and_json_layouts_are_read():
    """Round 2 EPL: fixturedownload.com had the full 2026-27 results (openfootball stopped at 20 September)."""
    text = ("Match Number,Round Number,Date,Location,Home Team,Away Team,Result\n"
            "1,1,21/08/2026 20:00,Emirates Stadium,Arsenal,Coventry,3 - 0\n"
            "2,1,22/08/2026 12:30,MKM Stadium,Hull,Man Utd,\n")
    first, second = S.read_matches_csv(text)
    assert (first.date, first.home, first.away, first.home_goals, first.away_goals) == ("2026-08-21", "Arsenal", "Coventry", 3, 0)
    assert not second.played
    feed = json.dumps([{"MatchNumber": 1, "DateUtc": "2026-08-21 19:00:00Z", "HomeTeam": "Arsenal", "AwayTeam": "Coventry",
                        "HomeTeamScore": 3, "AwayTeamScore": 0},
                       {"MatchNumber": 2, "DateUtc": "2026-08-22 11:30:00Z", "HomeTeam": "Hull", "AwayTeam": "Man Utd",
                        "HomeTeamScore": None, "AwayTeamScore": None}])
    a, b = S.parse_matches(feed, "epl-2026.json")
    assert (a.date, a.home_goals, a.away_goals) == ("2026-08-21", 3, 0) and not b.played


def test_the_current_file_given_as_history_is_not_counted_twice(tmp_path):
    names = [f"Club {i}" for i in range(6)]
    doc = _football_json(names, played=26)
    (tmp_path / "now.json").write_text(json.dumps(doc), encoding="utf-8")
    tool = simulate_season_tool({"now.json": tmp_path / "now.json"})
    out = tool.fn({"results": "now.json", "history": ["now.json"], "simulations": 100, "backtest": False}, "k")
    note = next(n for n in out["notes"] if n.startswith("history:"))
    assert "0 matches from earlier seasons" in note and "not counted twice" in note


def test_an_unknown_results_name_points_to_source():
    with pytest.raises(ValueError, match="source"):
        simulate_season_tool({}).fn({"results": "out_file.csv"}, "k")
