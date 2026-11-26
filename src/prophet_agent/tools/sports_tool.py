"""simulate_season: fit a sports model on a results file and simulate the rest of the season.

Motivation vs Logic
Motivation: a planner asked for a league table forecast summed points ranges across teams, which is meaningless, and had no
model of a season. The tool gives each team its own range and probabilities, with the backtest beside them.
Logic: the results come from one of the person's uploaded files or an analysis run's out/ file, read only. A CSV or an
openfootball football.json is parsed by forecast.sports. The full result is kept in the ledger; the planner sees a compact view.
"""
from __future__ import annotations

import json
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
        for hist_name in names:
            htext, hname = _named_text(hist_name, datasets, "history")
            history.extend(parse_matches(htext, hname))
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


VIEW_TEAMS = 6           # team rows in the planner's view when no teams were named
VIEW_PATHS = 6           # teams whose compact points path is shown


def compact_path(path: Any, every: int = 5) -> list:
    """The unplayed part of one team's points path: every ``every``-th game and the last, as [game, median, p10, p90]."""
    if not isinstance(path, list):
        return []
    ahead = [p for p in path if isinstance(p, dict) and not p.get("measured")]
    keep = [p for i, p in enumerate(ahead) if i % every == every - 1 or i == len(ahead) - 1]
    return [[p.get("game"), p.get("median"), p.get("p10"), p.get("p90")] for p in keep]


def _rounded(value: Any) -> Any:
    if isinstance(value, float):
        return round(value, 3)
    if isinstance(value, dict):
        return {k: _rounded(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_rounded(v) for v in value]
    return value


def _compact_backtest(bt: Any) -> Any:
    """Each model's overall scores and each comparison's verdict and mean difference (intervals stay in the full result)."""
    if not isinstance(bt, dict) or "comparisons" not in bt:
        return bt
    keep = {k: bt.get(k) for k in ("n_test", "origins", "reading") if k in bt}
    if isinstance(bt.get("models"), dict):        # the overall scores only; per-outcome scores stay in the full result
        keep["models"] = {m: {k: v for k, v in sc.items() if k in ("brier", "rps", "log_loss", "mae")}
                          for m, sc in bt["models"].items() if isinstance(sc, dict)}
    keep["comparisons"] = [{k: c.get(k) for k in ("comparison", "verdict", "mean_difference")}
                           for c in bt.get("comparisons") or [] if isinstance(c, dict)]
    return _rounded(keep)


VIEW_BUDGET = 3250       # characters of the view: the observation (OBS_CHARS 4,000) also carries the call's arguments (round 14: 2,900 left two named NBA teams out once ten better teams were listed)
LEAD_COLUMNS = ("team", "current", "expected_final", "p10", "median", "p90")
SKIP_COLUMNS = ("expected_rank_in_group",)


def _cell(v: Any) -> Any:
    v = _rounded(v)
    return int(v) if isinstance(v, float) and v.is_integer() else v


def team_table(rows: list[dict]) -> dict:
    """Team rows as one table: {"columns": [...], "rows": [[...], ...]}. The final total's 10th, 50th and 90th per cent
    become p10, median and p90; expectations keep one decimal; probabilities three. Columns beyond the lead ones are the
    probabilities the rows carry (playoff, title, top, relegation), in the order first seen."""
    extra: list[str] = []
    for r in rows:
        for k in r:
            if k not in ("team", "current", "expected_final", "final", *SKIP_COLUMNS, *extra):
                extra.append(k)
    cols = [*LEAD_COLUMNS, *extra]
    out = []
    for r in rows:
        final = r.get("final") if isinstance(r.get("final"), dict) else {}
        cell = {**r, "p10": final.get("p10"), "median": final.get("median"), "p90": final.get("p90")}
        if isinstance(cell.get("expected_final"), float):
            cell["expected_final"] = round(cell["expected_final"], 1)
        out.append([_cell(cell.get(c)) for c in cols])
    return {"columns": cols, "rows": out}


def extremes(rows: list[dict]) -> dict:
    """The highest and lowest expected_final and probability over every team row, as {column: [[team, max], [team, min]]}.

    Motivation: panel round 11 (NBA): the view held the 14 teams the call named, and the answer said playoff probabilities
    "range from 72% (Lakers)", while an unnamed team (Spurs) had 73.5 %. A claim across teams needs every team's extremes.
    """
    out: dict = {}
    cols: list[str] = []
    for r in rows:
        for k, v in r.items():
            if k not in ("team", "current", "final", *SKIP_COLUMNS, *cols) and isinstance(v, (int, float)) \
                    and not isinstance(v, bool) and (k == "expected_final" or 0.0 <= v <= 1.0):
                cols.append(k)
    for c in cols:
        vals = [(r.get(c), str(r.get("team"))) for r in rows if isinstance(r.get(c), (int, float))
                and not isinstance(r.get(c), bool)]
        if len(vals) < 2:
            continue
        hi, lo = max(vals), min(vals)
        out[c] = [[hi[1], _cell(round(hi[0], 1) if c == "expected_final" else hi[0])],
                  [lo[1], _cell(round(lo[0], 1) if c == "expected_final" else lo[0])]]
    return out


def skipped_leaders(rows: list[dict], shown: list[dict], limit: int = 10) -> list[list]:
    """Teams not shown whose expected_final is above the lowest shown team's, best first, as compact rows."""
    named = {r.get("team") for r in shown}
    floor = min((r["expected_final"] for r in shown if isinstance(r.get("expected_final"), (int, float))), default=None)
    if floor is None:
        return []
    better = sorted((r for r in rows if r.get("team") not in named and isinstance(r.get("expected_final"), (int, float))
                     and r["expected_final"] > floor), key=lambda r: -r["expected_final"])[:limit]
    out = []
    for r in better:
        final = r.get("final") if isinstance(r.get("final"), dict) else {}
        out.append([r.get("team"), _cell(r.get("current")), _cell(round(r["expected_final"], 1)), _cell(final.get("p10")),
                    _cell(final.get("p90"))])
    return out


def _size(view: dict) -> int:
    return len(json.dumps(view, ensure_ascii=False, separators=(",", ":")))


def observe_simulate(result: dict, teams_of_interest: list | None = None) -> dict:
    """The planner's view, sized to fit one observation: the rows of the teams the call asked about (or the first six) as
    one table, the backtest verdicts, and compact points paths. The full result, with every team, the standings, ratings
    and paths, stays in the ledger.

    Motivation: panel round 4 (EPL) paged eight times through game-by-game paths; round 6 (EPL) lost the path the goal asks
    for when they were dropped; round 7 (NBA) the view was 13,592 characters, so it was clipped and the planner paged
    18 times through the full result. Round 9 (NBA): the call named 15 teams, the view showed the first six by standing
    (only three of them named), and the planner paged 25 times to find the rest, then stated a Brier score no tool produced.
    Round 10 (NBA): the view of 14 named teams was 3,681 characters, but with the call's arguments the observation passed
    4,000, was clipped, and the planner paged 32 times.
    Logic: the teams asked about are the keys of points_paths (simulate_season keeps paths for exactly those teams), unless
    every team has a path. Rows are one table. Paths are dropped first, then rows from the end, until the view fits
    VIEW_BUDGET; what is left out is named.
    """
    if "teams" not in result:
        return result
    rows = [r for r in result.get("teams") or [] if isinstance(r, dict)]
    paths_all = result.get("points_paths") if isinstance(result.get("points_paths"), dict) else {}
    named = [t for t in (teams_of_interest or []) if isinstance(t, str)]
    if not named and paths_all and len(paths_all) < len(rows):
        named = list(paths_all)
    shown = [r for r in rows if r.get("team") in named] if named else []
    shown = shown or rows[:VIEW_TEAMS]
    out = {k: v for k, v in result.items() if k not in ("standings", "ratings", "points_paths", "teams", "backtest")}
    if isinstance(result.get("settings"), dict):    # the fields an answer cites; the rest stay in the full result
        out["settings"] = {k: v for k, v in result["settings"].items()
                           if k in ("model", "n_played", "n_remaining", "n_sims", "points_rule", "games_per_team")}
    if "backtest" in result:
        out["backtest"] = _compact_backtest(result["backtest"])
    out["note"] = ("every figure the answer needs is in this view: the teams table (final total p10, median, p90 and "
                   "probabilities), the backtest verdicts and points_path_summary ([game, median, p10, p90], where game is "
                   "the team's game number in the season, played games included; every tenth game and the last); do not "
                   "page through the full result. Do not sum ranges across teams.")
    paths = [(t, compact_path(paths_all[t], every=10)) for t in [r.get("team") for r in shown] if t in paths_all][:VIEW_PATHS]
    if len(shown) < len(rows):
        ext = extremes(rows)
        if ext:
            out["all_teams_extremes"] = ext
        missed = skipped_leaders(rows, shown)
        if missed:
            # Panel round 12 (NBA): the call named ten teams from memory; the current leaders (Spurs, Lakers, 76ers) were
            # not among them and the evaluator refused the answer for "the top teams in each conference".
            out["better_teams_not_named"] = missed
            out["note"] += (" better_teams_not_named: unnamed teams above a named one [team, current, expected_final, p10, "
                            "p90]; a top-teams ranking includes them.")
            out["note"] += " all_teams_extremes: [team, value] high and low over every team; use it for any range across teams."
    left_out: list[str] = []
    while True:
        out["teams"] = team_table(shown)
        out["points_path_summary"] = dict(paths)
        if len(rows) > len(shown):
            out["teams_not_shown"] = len(rows) - len(shown)
        if left_out:
            out["named_teams_not_shown"] = left_out
        if _size(out) <= VIEW_BUDGET or (not paths and len(shown) <= 1):
            break
        # Every path goes before any named row (round 12, NBA: one path was kept while nine named teams were dropped).
        if paths:
            paths = paths[:-1]
        elif len(shown) > 1:
            left_out.insert(0, str(shown[-1].get("team")))
            shown = shown[:-1]
    if not out["points_path_summary"]:
        del out["points_path_summary"]
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
