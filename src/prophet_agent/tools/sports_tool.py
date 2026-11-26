"""simulate_season: fit a sports model on a results file and simulate the rest of the season.

Motivation vs Logic
Motivation: a planner asked for a league table forecast summed points ranges across teams, which is meaningless, and had no
model of a season. The tool gives each team its own range and probabilities, with the backtest beside them.
Logic: the results come from one of the person's uploaded files or an analysis run's out/ file, read only. A CSV or an
openfootball football.json is parsed by forecast.sports. The full result is kept in the ledger; the planner sees a compact view.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from prophet_agent.forecast.sports import (DEFAULT_HALF_LIFE_DAYS, DEFAULT_POINTS, DEFAULT_PRIOR_SD, FOOTBALL, WIN_LOSS,
                                           parse_matches, placeholder_rows, simulate_season)
from prophet_agent.runtime.runner import Tool
from prophet_agent.tools.series_io import source_path

MAX_SIMS = 100_000
ALLOWED = {"sport", "results", "history", "source", "fixtures", "teams", "simulations", "seed", "points", "half_life_days",
           "prior_sd", "relegation_places", "top_places", "playoff_places", "games_per_team", "backtest"}


def _named_text(name: Any, datasets: dict[str, Path], label: str) -> tuple[str, str]:
    if not isinstance(name, str) or name not in datasets:
        raise ValueError(f"{label} must be one of the mission's file names {sorted(datasets)}; a file a sandbox run "
                         "wrote is passed as source {run_id, file} instead")
    return Path(datasets[name]).read_text(encoding="utf-8"), name


def _int(args: dict, key: str, default: int, low: int, high: int) -> int:
    value = args.get(key, default)
    if not isinstance(value, int) or isinstance(value, bool) or not low <= value <= high:
        raise ValueError(f"{key} must be a whole number from {low} to {high}")
    return value


def _simulate(args: dict, _key: str, datasets: dict[str, Path]) -> dict[str, Any]:
    extra = sorted(set(args) - ALLOWED)
    if extra:
        raise ValueError(f"unknown arguments {extra}; allowed: {sorted(ALLOWED)}")
    sport = args.get("sport", FOOTBALL)
    if sport not in (FOOTBALL, WIN_LOSS):
        raise ValueError("sport must be football (draws possible, Poisson goals) or win_loss (no draws, score margins)")
    has_file, has_source = args.get("results") is not None, args.get("source") is not None
    if has_file == has_source:
        raise ValueError("give exactly one of results (an uploaded file name) or source (an analysis run's out/ file)")
    if has_file:
        text, name = _named_text(args["results"], datasets, "results")
    else:
        path = source_path(args)
        text, name = path.read_text(encoding="utf-8"), path.name
    matches = parse_matches(text, name)
    history = None
    if args.get("history") is not None:
        names = args["history"] if isinstance(args["history"], list) else [args["history"]]
        if not 1 <= len(names) <= 4:
            raise ValueError("history must name 1 to 4 results files of earlier seasons")
        history = []
        played = {(m.date, m.home, m.away) for m in matches if m.played}
        for hist_name in names:
            htext, hname = _named_text(hist_name, datasets, "history")
            earlier = parse_matches(htext, hname)
            # Round 6 (EPL): the current season's file was passed as history beside its own parsed results, so every
            # played match was fitted twice. History is earlier seasons only.
            shared = sum(1 for m in earlier if m.played and (m.date, m.home, m.away) in played)
            if shared:
                raise ValueError(f"history file {hname} holds {shared} of the matches already in the results; history "
                                 "names earlier seasons' files only. Leave history out, or name last season's file")
            history.extend(earlier)
    fixtures = None
    if args.get("fixtures") is not None:
        ftext, fname = _named_text(args["fixtures"], datasets, "fixtures")
        fixtures = parse_matches(ftext, fname)
    teams = args.get("teams")
    if teams is not None and (not isinstance(teams, list) or not all(isinstance(t, str) for t in teams)):
        raise ValueError("teams must be a list of team names as they appear in the results")
    points = args.get("points", list(DEFAULT_POINTS))
    if (not isinstance(points, list) or len(points) != 3 or not all(isinstance(v, int) and not isinstance(v, bool)
                                                                   for v in points)):
        raise ValueError("points must be three whole numbers: win, draw, loss")
    half_life = args.get("half_life_days", DEFAULT_HALF_LIFE_DAYS)
    if half_life is not None and (not isinstance(half_life, (int, float)) or isinstance(half_life, bool) or half_life <= 0):
        raise ValueError("half_life_days must be a positive number, or null for no time decay")
    prior_sd = args.get("prior_sd", DEFAULT_PRIOR_SD)
    if not isinstance(prior_sd, (int, float)) or isinstance(prior_sd, bool) or prior_sd <= 0:
        raise ValueError("prior_sd must be a positive number")
    games = args.get("games_per_team")
    if games is not None:
        games = _int(args, "games_per_team", 0, 1, 200)
    backtest = args.get("backtest", True)
    if not isinstance(backtest, bool):
        raise ValueError("backtest must be true or false")
    result = simulate_season(
        matches, fixtures, sport=sport, n_sims=_int(args, "simulations", 10_000, 10, MAX_SIMS),
        seed=_int(args, "seed", 0, 0, 2**31 - 1), points=tuple(points),
        half_life_days=None if half_life is None else float(half_life), prior_sd=float(prior_sd),
        teams_of_interest=teams, relegation_places=_int(args, "relegation_places", 3, 0, 10),
        top_places=_int(args, "top_places", 4, 0, 10), playoff_places=_int(args, "playoff_places", 8, 1, 30),
        games_per_team=games, backtest=backtest, history=history)
    if (placeholders := placeholder_rows(text, name)):
        result.setdefault("notes", []).append(f"{placeholders} unplayed rows whose teams are not known yet (such as 'To be "
                                              "announced') were left out of the remaining schedule")
    return result


def compact_path(path: Any) -> list:
    """The unplayed part of one team's points path: every fifth game and the last, as [game, median, p10, p90]."""
    if not isinstance(path, list):
        return []
    ahead = [p for p in path if isinstance(p, dict) and not p.get("measured")]
    keep = [p for i, p in enumerate(ahead) if i % 5 == 4 or i == len(ahead) - 1]
    return [[p.get("game"), p.get("median"), p.get("p10"), p.get("p90")] for p in keep]


def observe_simulate(result: dict) -> dict:
    """The planner's view: the team table, settings, backtest and the first 20 standings and ratings. The ledger keeps all."""
    if "teams" not in result:
        return result
    # Panel round 4 (EPL): the points paths were 11,315 of 24,905 characters, and the planner paged eight times to read a
    # game-by-game table it never used. The paths stay in the full result for charts and checks.
    # Round 6 (EPL): with the paths gone the evaluator could not see the points path the goal asks for. A compact path stays:
    # the unplayed games every fifth game and the last, as [game, median, p10, p90].
    out = {k: v for k, v in result.items() if k not in ("standings", "ratings", "points_paths")}
    out["standings"] = (result.get("standings") or [])[:20]
    out["ratings"] = (result.get("ratings") or [])[:20]
    if isinstance(result.get("points_paths"), dict):
        out["points_path_summary"] = {team: compact_path(path) for team, path in result["points_paths"].items()}
    out["note"] = ("per-team ranges shown; points_path_summary gives each team's path as [game, median, p10, p90] every "
                   "fifth unplayed game and the last; the full paths, standings and ratings are kept in the ledger. Do not "
                   "sum ranges across teams.")
    return out


def simulate_season_tool(datasets: dict[str, Path] | None = None, live: Any = None) -> Tool:
    """The tool over the person's files and, when ``live`` (the mission's LiveMounts) is given, every file the mission
    downloaded so far, such as an openfootball JSON landed by download_web_data. Read at each call, as the sandbox does."""
    data = dict(datasets or {})

    def files() -> dict[str, Path]:
        return {**data, **(live.mounts() if live is not None else {})}

    return Tool("simulate_season", lambda args, key: _simulate(args, key, files()), idempotent=True,
                observe=observe_simulate)


SIMULATE_SEASON_SPEC = {
    "name": "simulate_season",
    "title": "Season simulation",
    "activity": "Simulating the rest of the season",
    "description": "Fit a sports model on a results file and simulate the rest of the season many times. sport football "
                   "(default): Dixon-Coles Poisson goals with time decay and a low-score correction; the table uses points "
                   "and the usual tie-breaks. sport win_loss (no draws, for example basketball): a score-margin model. Give "
                   "results as a mission CSV (date, home, away, home_goals, away_goals, optional group) or openfootball "
                   "football.json, or as source (an analysis run's out/ file). Remaining matches come from fixtures (a mission "
                   "file), else unplayed rows in results, else for football a double round robin, else for win_loss "
                   "games_per_team per team. Returns per team: final total (median, 10 and 90 per cent), probability of the title, "
                   "top places, relegation (football) or the playoff line (win_loss, per group when a group column is given), "
                   "points or wins paths for teams of interest, and a backtest against equal odds and simple rules with "
                   "measured verdicts. Each team's range stands alone; never add ranges across teams. Figures are simulated "
                   "from fitted ratings.",
    "args": {
        "sport": "football (default) or win_loss",
        "results": "name of a results file the mission holds (uploaded, or landed by download_web_data; CSV or football.json)",
        "source": "object {run_id, file} naming a results CSV or JSON in an analysis run's out/ folder, instead of results",
        "history": "optional name, or list of up to 4 names, of earlier seasons' results files; used to fit the ratings only "
                   "(early in a season the current results are too few)",
        "fixtures": "name of a mission file of remaining matches (optional; not needed when the results file lists unplayed matches, as fixturedownload does)",
        "teams": "list of team names to show paths for (optional; default the top four by current standing)",
        "simulations": "number of simulated seasons, 10 to 100000 (default 10000)",
        "seed": "random seed for reproducibility (default 0)",
        "points": "[win, draw, loss] points, football only (default [3, 1, 0])",
        "half_life_days": "time-decay half life in days, or null for none (default 365; with no dates, in matches)",
        "prior_sd": "prior standard deviation of a team rating (default 0.5)",
        "relegation_places": "bottom places relegated, football (default 3)",
        "top_places": "places counted as top, football (default 4)",
        "playoff_places": "places in the playoff line, win_loss, per group (default 8)",
        "games_per_team": "games per team in the full season, win_loss with no fixtures (for example 82)",
        "backtest": "true to run the rolling backtest (default true)",
    },
}
