"""Sports outcome models: standings, season simulation, and rolling-origin checks against simple rules.

Motivation vs Logic
Motivation: a season is a set of remaining fixtures, not a time series. A league table forecast needs an outcome probability
for each fixture and a simulation of the whole remaining schedule, with uncertainty in the ratings. Every model must be scored
against simple rules on past matches before its numbers are quoted.
Logic: ``sport="football"`` (draws possible) fits a Dixon-Coles model: independent Poisson goals with the low-score correction
tau, time-decayed likelihood, and ridge-shrunk attack and defence, maximised with analytic gradients. ``sport="win_loss"`` (no
draws) fits a normal score-margin model by weighted ridge least squares. Ratings are drawn from a Gaussian (Laplace)
approximation to their posterior, one draw per simulated season, and each remaining game is simulated from those draws.
Every team's range is its own; ranges are never summed across teams. Figures are model outputs labelled as simulated from
fitted ratings, not observed results.
"""
from __future__ import annotations

import csv
import re
import datetime as dt
import io
import json
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import optimize
from scipy.special import gammaln, ndtr

DEFAULT_POINTS = (3, 1, 0)
DEFAULT_HALF_LIFE_DAYS = 365.0      # a default, not measured here; the notes list the time-decay value as unchecked
DEFAULT_PRIOR_SD = 0.5              # prior sd of a team rating on the log (football) or points (win_loss) scale
MAX_GOALS = 12
FOOTBALL = "football"
WIN_LOSS = "win_loss"
FIGURE_LABEL = "simulated from fitted ratings"


@dataclass(frozen=True)
class Match:
    date: str | None
    home: str
    away: str
    home_goals: int | None
    away_goals: int | None
    group: str | None = None

    @property
    def played(self) -> bool:
        return self.home_goals is not None and self.away_goals is not None


# ---------------------------------------------------------------- reading results

def _score(text: str, label: str, line: int) -> int | None:
    if text == "":
        return None
    try:
        value = int(text)
    except ValueError:
        raise ValueError(f"row {line}: {label} must be a whole number, got {text!r}")
    if value < 0:
        raise ValueError(f"row {line}: {label} cannot be negative")
    return value


# Other spellings of the columns: openfootball flattened (team1, team2), football-data.co.uk (HomeTeam, AwayTeam, FTHG,
# FTAG, dates as dd/mm/yy or dd/mm/yyyy) and common variants.
_CSV_ALIASES = {"team1": "home", "team2": "away", "hometeam": "home", "awayteam": "away", "home_team": "home",
                "away_team": "away", "fthg": "home_goals", "ftag": "away_goals", "hg": "home_goals", "ag": "away_goals",
                "home_score": "home_goals", "away_score": "away_goals", "homegoals": "home_goals",
                "awaygoals": "away_goals", "div": "group_code"}
_DMY = re.compile(r"^\d{1,2}/\d{1,2}/\d{2,4}$")


def read_matches_csv(text: str) -> list[Match]:
    """Matches from a CSV with columns home, away and optionally date, home_goals, away_goals, group. A row with no score is
    a fixture. Motivation: the planner hands over a file, not a Python object. Logic: one row per match, scores as whole numbers."""
    reader = csv.DictReader(io.StringIO(text))
    cols = {_CSV_ALIASES.get(c.strip().lower(), c.strip().lower()) for c in (reader.fieldnames or []) if c}
    if not {"home", "away"} <= cols:
        raise ValueError("the CSV needs columns home and away (or team1 and team2, HomeTeam and AwayTeam); date, "
                         "home_goals, away_goals and group are optional")
    out: list[Match] = []
    for line, row in enumerate(reader, start=2):
        r = {_CSV_ALIASES.get(k.strip().lower(), k.strip().lower()): (v or "").strip() for k, v in row.items() if k}
        if r.get("date") and _DMY.match(r["date"]):
            day, month, year = r["date"].split("/")
            r["date"] = f"{'20' + year if len(year) == 2 else year}-{month.zfill(2)}-{day.zfill(2)}"
        if not r.get("home") or not r.get("away"):
            raise ValueError(f"row {line}: home and away are needed")
        if r["home"] == r["away"]:
            raise ValueError(f"row {line}: a team cannot play itself")
        hg = _score(r.get("home_goals", ""), "home_goals", line)
        ag = _score(r.get("away_goals", ""), "away_goals", line)
        if (hg is None) != (ag is None):
            raise ValueError(f"row {line}: give both scores or neither")
        out.append(Match(r.get("date") or None, r["home"], r["away"], hg, ag, r.get("group") or None))
    return out


def parse_football_json(doc: Any) -> list[Match]:
    """Matches from an openfootball football.json document.

    Motivation vs Logic
    Motivation: the open football data is published as one JSON file per league and season, and the planner should not
    re-parse it in the sandbox. Logic: each entry of ``matches`` gives team1 (home), team2 (away), date and, when played,
    ``score.ft`` as [home goals, away goals]. An entry with no full-time score is a fixture.
    """
    if not isinstance(doc, dict) or not isinstance(doc.get("matches"), list):
        raise ValueError("not a football.json document: expected an object with a matches list")
    out: list[Match] = []
    for i, m in enumerate(doc["matches"]):
        if not isinstance(m, dict):
            raise ValueError(f"match {i}: each match must be an object")
        home, away = m.get("team1"), m.get("team2")
        if not isinstance(home, str) or not isinstance(away, str) or not home or not away:
            raise ValueError(f"match {i}: team1 and team2 must be team names")
        score = m.get("score")
        # Most entries give {"ft": [h, a]}; some give the full-time score as a bare [h, a] list (2026-27 en.1.json, live).
        ft = score.get("ft") if isinstance(score, dict) else (score if isinstance(score, list) and score else None)
        if ft is None:
            hg = ag = None
        else:
            if not isinstance(ft, list) or len(ft) != 2 or not all(isinstance(g, int) and g >= 0 for g in ft):
                raise ValueError(f"match {i}: score.ft must be two whole numbers")
            hg, ag = int(ft[0]), int(ft[1])
        date = m.get("date") if isinstance(m.get("date"), str) else None
        out.append(Match(date, home, away, hg, ag, None))
    return out


def parse_matches(text: str, filename: str = "") -> list[Match]:
    """Matches from a CSV or an openfootball football.json, chosen by the file name."""
    if filename.lower().endswith(".json"):
        try:
            doc = json.loads(text)
        except ValueError:
            raise ValueError("the JSON file does not parse")
        return parse_football_json(doc)
    return read_matches_csv(text)


# ---------------------------------------------------------------- standings

def standings(matches: list[Match], points: tuple[int, int, int] = DEFAULT_POINTS) -> list[dict]:
    """League table from played matches, sorted by points, goal difference, goals for, then name.

    Motivation vs Logic: a table scraped from a page can disagree with its own results; deriving it from the results keeps the
    two consistent. Logic: 3, 1 and 0 points by default (win, draw, loss) and the usual tie-breaks; teams are reported
    alphabetically only when everything else is equal.
    """
    win, draw, loss = points
    rows: dict[str, dict] = {}

    def row(team: str) -> dict:
        return rows.setdefault(team, {"team": team, "played": 0, "won": 0, "drawn": 0, "lost": 0,
                                      "goals_for": 0, "goals_against": 0})

    for m in matches:
        if not m.played:
            continue
        h, a = row(m.home), row(m.away)
        h["played"] += 1
        a["played"] += 1
        h["goals_for"] += m.home_goals
        h["goals_against"] += m.away_goals
        a["goals_for"] += m.away_goals
        a["goals_against"] += m.home_goals
        if m.home_goals > m.away_goals:
            h["won"] += 1
            a["lost"] += 1
        elif m.home_goals < m.away_goals:
            a["won"] += 1
            h["lost"] += 1
        else:
            h["drawn"] += 1
            a["drawn"] += 1
    for r in rows.values():
        r["goal_difference"] = r["goals_for"] - r["goals_against"]
        r["points"] = r["won"] * win + r["drawn"] * draw + r["lost"] * loss
    table = sorted(rows.values(), key=lambda r: (-r["points"], -r["goal_difference"], -r["goals_for"], r["team"]))
    for position, r in enumerate(table, start=1):
        r["position"] = position
    return table


# ---------------------------------------------------------------- time handling

def _timeline(matches: list[Match]) -> np.ndarray:
    """Day numbers when every match has an ISO date, otherwise the list position. Motivation: decay needs a clock, and a file
    without dates still has an order. Logic: a mixed file falls back to position so no match is half-dated."""
    days = []
    for m in matches:
        try:
            days.append(float(dt.date.fromisoformat((m.date or "")[:10]).toordinal()))
        except ValueError:
            days = None
            break
    return np.array(days if days is not None else np.arange(len(matches), dtype=float))


def _decay_weights(times: np.ndarray, half_life: float | None) -> np.ndarray:
    if not half_life:
        return np.ones_like(times)
    return 0.5 ** ((times.max() - times) / float(half_life))


def _chronological(matches: list[Match]) -> tuple[list[Match], np.ndarray]:
    times = _timeline(matches)
    order = np.argsort(times, kind="stable")
    return [matches[i] for i in order], times[order]


def _team_list(*groups: list[Match]) -> list[str]:
    return sorted({t for g in groups for m in g for t in (m.home, m.away)})


# ---------------------------------------------------------------- football: Dixon-Coles

def _tau(x, y, lam, mu, rho):
    """Dixon-Coles low-score correction for the scores 0-0, 0-1, 1-0 and 1-1; 1 elsewhere."""
    tau = np.ones(np.broadcast(x, lam).shape)
    tau = np.where((x == 0) & (y == 0), 1 - lam * mu * rho, tau)
    tau = np.where((x == 0) & (y == 1), 1 + lam * rho, tau)
    tau = np.where((x == 1) & (y == 0), 1 + mu * rho, tau)
    tau = np.where((x == 1) & (y == 1), 1 - rho, tau)
    return tau


def _objective(theta, hi, ai, x, y, w, n_teams, prior_var):
    """Penalised negative log-likelihood and its gradient.

    Motivation vs Logic
    Motivation: scipy needs a value and a gradient; numerical gradients on 40 or more ratings are slow and noisy, and the
    Hessian for the uncertainty is built from this gradient.
    Logic: home goals ~ Poisson(lam), away goals ~ Poisson(mu) with log lam = att[home] - dfn[away] + home_adv and
    log mu = att[away] - dfn[home]; the log tau correction is differentiated term by term. A Gaussian ridge with variance
    prior_var pulls attack and defence toward zero, which also removes the shift between them that the likelihood cannot see.
    """
    T = n_teams
    att, dfn = theta[:T], theta[T:2 * T]
    home, rho = theta[2 * T], theta[2 * T + 1]
    eta_l = att[hi] - dfn[ai] + home
    eta_m = att[ai] - dfn[hi]
    lam, mu = np.exp(eta_l), np.exp(eta_m)
    raw = _tau(x, y, lam, mu, rho)
    tau = np.maximum(raw, 1e-8)
    ll = np.sum(w * (np.log(tau) + x * eta_l - lam + y * eta_m - mu))
    f = -ll + 0.5 / prior_var * (att @ att + dfn @ dfn)

    c00 = (x == 0) & (y == 0)
    c01 = (x == 0) & (y == 1)
    c10 = (x == 1) & (y == 0)
    c11 = (x == 1) & (y == 1)
    dtl = np.where(c00, -rho * lam * mu, np.where(c01, rho * lam, 0.0))
    dtm = np.where(c00, -rho * lam * mu, np.where(c10, rho * mu, 0.0))
    dtr = np.where(c00, -lam * mu, np.where(c01, lam, np.where(c10, mu, np.where(c11, -1.0, 0.0))))
    gl = w * ((x - lam) + dtl / tau)
    gm = w * ((y - mu) + dtm / tau)
    gr = w * dtr / tau
    g_att = np.bincount(hi, gl, T) + np.bincount(ai, gm, T)
    g_dfn = -np.bincount(ai, gl, T) - np.bincount(hi, gm, T)
    grad = -np.concatenate([g_att, g_dfn, [gl.sum(), gr.sum()]])
    grad[:2 * T] += np.concatenate([att, dfn]) / prior_var
    return f, grad


def _psd_inverse(h: np.ndarray) -> np.ndarray:
    vals, vecs = np.linalg.eigh((h + h.T) / 2)
    vals = np.maximum(vals, 1e-6)
    return (vecs / vals) @ vecs.T


@dataclass
class DixonColes:
    teams: list[str]
    theta: np.ndarray          # [attack (T), defence (T), home advantage, rho]
    cov: np.ndarray            # Laplace approximation to the posterior covariance of theta
    n_matches: int
    half_life_days: float | None
    prior_sd: float
    converged: bool

    @property
    def index(self) -> dict[str, int]:
        return {t: i for i, t in enumerate(self.teams)}

    @property
    def n_teams(self) -> int:
        return len(self.teams)

    def rates(self, home: str, away: str) -> tuple[float, float]:
        T, idx = self.n_teams, self.index
        att, dfn = self.theta[:T], self.theta[T:2 * T]
        h, a = idx.get(home), idx.get(away)
        ah, dh = (att[h], dfn[h]) if h is not None else (0.0, 0.0)
        aa, da = (att[a], dfn[a]) if a is not None else (0.0, 0.0)
        return float(np.exp(ah - da + self.theta[2 * T])), float(np.exp(aa - dh))

    @property
    def rho(self) -> float:
        return float(self.theta[-1])

    @property
    def home_advantage(self) -> float:
        return float(self.theta[2 * self.n_teams])


def fit_dixon_coles(matches: list[Match], *, teams: list[str] | None = None,
                    half_life_days: float | None = DEFAULT_HALF_LIFE_DAYS,
                    prior_sd: float = DEFAULT_PRIOR_SD) -> DixonColes:
    """Fit the Dixon-Coles model by penalised maximum likelihood on played matches.

    Motivation vs Logic
    Motivation: a team with ten matches must not get an extreme rating from them, and a recent season should count more than an
    old one. Logic: exponential time decay with the given half life (weights relative to the latest match), a Gaussian prior on
    each attack and defence with standard deviation prior_sd, and the Laplace approximation (inverse Hessian at the optimum) for
    the spread of the ratings. Refuses when there are fewer than four played matches per team.
    """
    played, times = _chronological([m for m in matches if m.played])
    teams = teams or _team_list(played)
    if len(played) < 4 * len(teams):
        raise ValueError(f"too few matches to rate the teams: {len(played)} played for {len(teams)} teams; "
                         f"at least {4 * len(teams)} are needed")
    times = _timeline(played)
    idx = {t: i for i, t in enumerate(teams)}
    hi = np.array([idx[m.home] for m in played])
    ai = np.array([idx[m.away] for m in played])
    x = np.array([m.home_goals for m in played], dtype=float)
    y = np.array([m.away_goals for m in played], dtype=float)
    w = _decay_weights(times, half_life_days)
    T = len(teams)
    args = (hi, ai, x, y, w, T, prior_sd ** 2)
    theta0 = np.zeros(2 * T + 2)
    theta0[2 * T] = 0.25
    bounds = [(None, None)] * (2 * T + 1) + [(-0.5, 0.5)]
    res = optimize.minimize(_objective, theta0, args=args, jac=True, method="L-BFGS-B", bounds=bounds,
                            options={"maxiter": 2000})
    theta = res.x
    h = 1e-5
    H = np.zeros((len(theta), len(theta)))
    for i in range(len(theta)):
        e = np.zeros(len(theta))
        e[i] = h
        H[:, i] = (_objective(theta + e, *args)[1] - _objective(theta - e, *args)[1]) / (2 * h)
    cov = _psd_inverse(H)
    return DixonColes(teams, theta, cov, len(played), half_life_days, prior_sd, bool(res.success))


def score_grid(lam: float, mu: float, rho: float) -> np.ndarray:
    """Joint probabilities of home goals (rows) and away goals (columns) up to MAX_GOALS, with the low-score correction."""
    g = np.arange(MAX_GOALS + 1)
    px = np.exp(g * np.log(lam) - lam - gammaln(g + 1))
    py = np.exp(g * np.log(mu) - mu - gammaln(g + 1))
    grid = np.outer(px, py)
    grid[0, 0] *= max(1 - lam * mu * rho, 0.0)
    grid[0, 1] *= max(1 + lam * rho, 0.0)
    grid[1, 0] *= max(1 + mu * rho, 0.0)
    grid[1, 1] *= max(1 - rho, 0.0)
    return grid / grid.sum()


def outcome_probs(grid: np.ndarray) -> tuple[float, float, float]:
    """(home win, draw, away win) from a joint score grid."""
    return float(np.tril(grid, -1).sum()), float(np.trace(grid)), float(np.triu(grid, 1).sum())


def _sample_scores(lam: np.ndarray, mu: np.ndarray, rho: np.ndarray, rng: np.random.Generator):
    """Exact draws of (home, away) goals from the Dixon-Coles joint by rejection from independent Poisson draws.

    Motivation vs Logic
    Motivation: the low-score correction changes the 0-0, 0-1, 1-0 and 1-1 frequencies, and a simulated season must keep them.
    Logic: propose independent Poisson goals and accept with probability tau / M, where M is at least every tau value.
    Accepted draws follow the corrected joint; the acceptance rate is high because tau differs from 1 only on four cells.
    """
    shape = lam.shape
    lf, mf, rf = lam.ravel(), mu.ravel(), rho.ravel()
    M = np.maximum.reduce([np.ones_like(lf), 1 - lf * mf * rf, 1 + lf * rf, 1 + mf * rf, 1 - rf])
    n = lf.size
    x = np.zeros(n, dtype=np.int64)
    y = np.zeros(n, dtype=np.int64)
    done = np.zeros(n, dtype=bool)
    for _ in range(10_000):
        pending = np.flatnonzero(~done)
        if pending.size == 0:
            break
        xs = rng.poisson(lf[pending])
        ys = rng.poisson(mf[pending])
        tau = np.maximum(_tau(xs, ys, lf[pending], mf[pending], rf[pending]), 0.0)
        accept = rng.random(pending.size) * M[pending] < tau
        keep = pending[accept]
        x[keep] = xs[accept]
        y[keep] = ys[accept]
        done[keep] = True
    if not done.all():
        raise RuntimeError("score sampling did not finish; the ratings are out of range")
    return x.reshape(shape), y.reshape(shape)


def _pts(a: np.ndarray, b: np.ndarray, points: tuple[int, int, int]) -> np.ndarray:
    win, draw, loss = points
    return win * (a > b) + draw * (a == b) + loss * (a < b)


def _incidence(idx: np.ndarray, n_teams: int) -> np.ndarray:
    m = np.zeros((idx.size, n_teams))
    m[np.arange(idx.size), idx] = 1.0
    return m


def predict_football(model: DixonColes, home: str, away: str) -> tuple[float, float, float]:
    lam, mu = model.rates(home, away)
    return outcome_probs(score_grid(lam, mu, model.rho))


# ---------------------------------------------------------------- win_loss: score margin model

@dataclass
class MarginModel:
    teams: list[str]
    beta: np.ndarray           # [home advantage, rating of each team]
    cov: np.ndarray            # posterior covariance of beta
    sigma: float               # residual sd of the margin
    n_matches: int
    half_life_days: float | None
    prior_sd: float

    @property
    def index(self) -> dict[str, int]:
        return {t: i for i, t in enumerate(self.teams)}

    def mean_margin(self, home: str, away: str) -> float:
        idx = self.index
        rh = self.beta[1 + idx[home]] if home in idx else 0.0
        ra = self.beta[1 + idx[away]] if away in idx else 0.0
        return float(self.beta[0] + rh - ra)

    def home_win_prob(self, home: str, away: str) -> float:
        return float(ndtr(self.mean_margin(home, away) / self.sigma))


def fit_margin(matches: list[Match], *, teams: list[str] | None = None, half_life_days: float | None = DEFAULT_HALF_LIFE_DAYS,
               prior_sd: float = DEFAULT_PRIOR_SD) -> MarginModel:
    """Fit margin = home advantage + rating(home) - rating(away) + noise by weighted ridge least squares.

    Motivation vs Logic
    Motivation: a sport with no draws is better described by the score margin than by a win count, because a 20-point win and a
    1-point win carry different information. Logic: time-decay weights, a ridge penalty on ratings with variance sigma squared over
    prior_sd squared (so the shrinkage is in the rating scale), sigma estimated from the residuals, and the posterior covariance
    sigma squared times the inverse of the penalised normal matrix. The home advantage is not penalised.
    """
    played, _ = _chronological([m for m in matches if m.played])
    teams = teams or _team_list(played)
    if len(played) < 4 * len(teams):
        raise ValueError(f"too few matches to rate the teams: {len(played)} played for {len(teams)} teams; "
                         f"at least {4 * len(teams)} are needed")
    if any(m.home_goals == m.away_goals for m in played):
        raise ValueError("win_loss mode has no draws, but a played match has equal scores; check the sport or the file")
    times = _timeline(played)
    idx = {t: i for i, t in enumerate(teams)}
    n, T = len(played), len(teams)
    X = np.zeros((n, T + 1))
    X[:, 0] = 1.0
    for i, m in enumerate(played):
        X[i, 1 + idx[m.home]] += 1.0
        X[i, 1 + idx[m.away]] -= 1.0
    yv = np.array([m.home_goals - m.away_goals for m in played], dtype=float)
    w = _decay_weights(times, half_life_days)
    XtW = X.T * w
    G = XtW @ X
    b = XtW @ yv
    P = np.zeros((T + 1, T + 1))
    P[1:, 1:] = np.eye(T)
    ridge0 = np.linalg.solve(G + P + 1e-9 * np.eye(T + 1), b)
    s2 = float(np.sum(w * (yv - X @ ridge0) ** 2) / np.sum(w))
    lam = s2 / prior_sd ** 2
    A = G + lam * P + 1e-9 * np.eye(T + 1)
    beta = np.linalg.solve(A, b)
    s2 = float(np.sum(w * (yv - X @ beta) ** 2) / np.sum(w))
    cov = s2 * np.linalg.inv(A)
    return MarginModel(teams, beta, cov, float(np.sqrt(s2)), n, half_life_days, prior_sd)


# ---------------------------------------------------------------- season simulation

def _remaining_double_round_robin(teams: list[str], played: list[Match]) -> list[Match]:
    """Every ordered pair once (a double round robin), less the pairs already played."""
    count: dict[tuple[str, str], int] = defaultdict(int)
    for m in played:
        count[(m.home, m.away)] += 1
    out = []
    for h in teams:
        for a in teams:
            if h != a and count[(h, a)] == 0:
                out.append(Match(None, h, a, None, None))
    return out


def _fixed_values(played: list[Match], team: str, sport: str, points) -> list[float]:
    """The known per-match values of one team in chronological order (points or a win indicator)."""
    values = []
    for m in played:
        if team not in (m.home, m.away):
            continue
        mine, theirs = (m.home_goals, m.away_goals) if m.home == team else (m.away_goals, m.home_goals)
        values.append(float(_pts(np.array(mine), np.array(theirs), points)) if sport == FOOTBALL
                      else float(mine > theirs))
    return values


def _football_chunk(model: DixonColes, hi, ai, n, rng, points, interest):
    T = model.n_teams
    th = rng.multivariate_normal(model.theta, model.cov, size=n, method="eigh")
    att, dfn = th[:, :T], th[:, T:2 * T]
    home, rho = th[:, 2 * T], np.clip(th[:, 2 * T + 1], -0.5, 0.5)
    lam = np.exp(att[:, hi] - dfn[:, ai] + home[:, None])
    mu = np.exp(att[:, ai] - dfn[:, hi])
    hg, ag = _sample_scores(lam, mu, np.broadcast_to(rho[:, None], lam.shape).copy(), rng)
    ph, pa = _pts(hg, ag, points), _pts(ag, hg, points)
    H, A = _incidence(hi, T), _incidence(ai, T)
    values = {t: np.where((hi == t)[None, :], ph, pa)[:, np.flatnonzero((hi == t) | (ai == t))] for t in interest}
    return {"points": ph @ H + pa @ A, "gd": (hg - ag) @ H + (ag - hg) @ A, "gf": hg @ H + ag @ A,
            "ratings": th[:, :2 * T], "values": values}


def _winloss_fixture_chunk(model: MarginModel, hi, ai, n, rng, interest):
    T = len(model.teams)
    beta = rng.multivariate_normal(model.beta, model.cov, size=n, method="eigh")
    mu = beta[:, :1] + beta[:, 1 + hi] - beta[:, 1 + ai]
    home_win = (rng.random(mu.shape) < ndtr(mu / model.sigma)).astype(float)
    away_win = 1.0 - home_win
    H, A = _incidence(hi, T), _incidence(ai, T)
    values = {t: np.where((hi == t)[None, :], home_win, away_win)[:, np.flatnonzero((hi == t) | (ai == t))] for t in interest}
    return {"points": home_win @ H + away_win @ A, "ratings": beta[:, 1:], "values": values}


def _winloss_schedule_chunk(model: MarginModel, plan: list[dict], n, rng, interest):
    """Remaining games drawn per team from its observed opponent mix (no fixtures file). Each team's games are simulated
    on their own, so one team's game against another is not forced to match the other team's game (a stated approximation)."""
    T = len(model.teams)
    beta = rng.multivariate_normal(model.beta, model.cov, size=n, method="eigh")
    wins = np.zeros((n, T))
    values = {}
    rows = np.arange(n)[:, None]
    for t, item in enumerate(plan):
        r = item["games"]
        if r == 0:
            values[t] = np.zeros((n, 0))
            continue
        opp = rng.choice(T, size=(n, r), p=item["mix"])
        is_home = rng.random((n, r)) < item["home_share"]
        r_t = beta[:, 1 + t][:, None]
        r_o = beta[rows, 1 + opp]
        mu = np.where(is_home, beta[:, :1] + r_t - r_o, -(beta[:, :1] + r_o - r_t))
        won = (rng.random((n, r)) < ndtr(mu / model.sigma)).astype(float)
        wins[:, t] = won.sum(axis=1)
        values[t] = won
    return {"points": wins, "ratings": beta[:, 1:], "values": {t: values[t] for t in interest}}


def _positions(key: np.ndarray) -> np.ndarray:
    order = np.argsort(-key, axis=1, kind="stable")
    pos = np.empty_like(order)
    pos[np.arange(key.shape[0])[:, None], order] = np.arange(key.shape[1])[None, :]
    return pos


def _quantiles(a: np.ndarray) -> dict[str, float]:
    p10, p50, p90 = np.percentile(a, [10, 50, 90])
    return {"median": float(p50), "p10": float(p10), "p90": float(p90)}


def _compare(model_scores: np.ndarray, base_scores: np.ndarray, label: str, rng: np.random.Generator) -> dict:
    """Paired comparison of per-match scores where lower is better: mean difference with a bootstrap 95 per cent interval."""
    d = model_scores - base_scores
    boot = d[rng.integers(0, d.size, size=(1000, d.size))].mean(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    verdict = "model_better" if hi < 0 else "baseline_better" if lo > 0 else "no_clear_difference"
    return {"comparison": label, "mean_difference": float(d.mean()), "ci95": [float(lo), float(hi)], "verdict": verdict}


# ---------------------------------------------------------------- backtests

def _origins(n: int, min_train: int, block: int):
    start = min_train
    while start < n:
        yield start, min(n, start + block)
        start += block


def _elo_probs(train: list[Match], test: list[Match], draw_rate: float, home_adv: float = 60.0, k: float = 20.0):
    """Elo ratings from the training matches; the expected score is split into home, draw and away with a fixed draw share.
    Motivation: the simplest rating baseline every sports model should beat. Logic: E = expected score for the home side; the
    draw share d is taken from the training data, so home = E - d/2 and away = 1 - E - d/2, then normalised."""
    R: dict[str, float] = defaultdict(lambda: 1500.0)
    for m in train:
        e = 1 / (1 + 10 ** ((R[m.away] - R[m.home] - home_adv) / 400))
        s = 1.0 if m.home_goals > m.away_goals else 0.5 if m.home_goals == m.away_goals else 0.0
        R[m.home] += k * (s - e)
        R[m.away] -= k * (s - e)
    out = []
    for m in test:
        e = 1 / (1 + 10 ** ((R[m.away] - R[m.home] - home_adv) / 400))
        ph, pa = max(e - draw_rate / 2, 0.0), max(1 - e - draw_rate / 2, 0.0)
        tot = ph + draw_rate + pa
        out.append((ph / tot, draw_rate / tot, pa / tot))
    return out


def _outcome_index(m: Match) -> int:
    """0 away, 1 draw, 2 home: the ordinal order used by RPS."""
    return 2 if m.home_goals > m.away_goals else 0 if m.home_goals < m.away_goals else 1


def _rps_brier(p: np.ndarray, o: np.ndarray):
    """Ranked Probability Score (ordinal, away < draw < home) and per-outcome Brier, with p rows in the same order."""
    onehot = np.eye(3)[o]
    F, O = np.cumsum(p, axis=1)[:, :2], np.cumsum(onehot, axis=1)[:, :2]
    rps = 0.5 * np.sum((F - O) ** 2, axis=1)
    brier_each = (p - onehot) ** 2
    return rps, brier_each


def backtest_football(matches: list[Match], teams: list[str], *, half_life_days, prior_sd,
                      min_train: int | None = None, block: int | None = None) -> dict:
    """Rolling-origin backtest: fit on matches before each origin, score the next block, against equal probabilities, home
    rate and Elo. Motivation: the model must beat the simple rules on the same matches before any season is simulated.
    Logic: origins step through the chronological list; each block is scored with RPS and Brier; differences are paired
    per match with a bootstrap interval."""
    played, _ = _chronological([m for m in matches if m.played])
    n = len(played)
    min_train = min_train or max(n // 2, 4 * len(teams))
    block = block or max(1, n // 10)
    if min_train >= n:
        return {"skipped": "too few matches for a backtest after the training minimum"}
    scores = {k: [] for k in ("dixon_coles", "equal", "home_rate", "elo")}
    outcomes, briers = [], {k: [] for k in scores}
    origins = 0
    notes = []
    for s, e in _origins(n, min_train, block):
        train, test = played[:s], played[s:e]
        try:
            model = fit_dixon_coles(train, teams=teams, half_life_days=half_life_days, prior_sd=prior_sd)
        except ValueError as exc:
            notes.append(f"origin {s}: {exc}")
            continue
        origins += 1
        rates = np.array([sum(_outcome_index(m) == k for m in train) for k in (2, 1, 0)], float) / len(train)
        home_rate = (rates[0], rates[1], rates[2])
        dc = [predict_football(model, m.home, m.away) for m in test]
        elo = _elo_probs(train, test, draw_rate=float(rates[1]))
        preds = {"dixon_coles": dc, "equal": [(1 / 3, 1 / 3, 1 / 3)] * len(test), "home_rate": [home_rate] * len(test),
                 "elo": elo}
        outcomes.extend(_outcome_index(m) for m in test)
        for name, rows in preds.items():
            p = np.array([[pa, pd, ph] for ph, pd, pa in rows])   # ordinal order: away, draw, home
            rps, be = _rps_brier(p, np.array([_outcome_index(m) for m in test]))
            scores[name].extend(rps.tolist())
            briers[name].append(be)
    if origins == 0:
        return {"skipped": "no origin could be fitted", "notes": notes}
    o = np.array(outcomes)
    rng = np.random.default_rng(0)
    models = {}
    for name in scores:
        be = np.vstack(briers[name])
        models[name] = {"rps": float(np.mean(scores[name])), "brier": float(be.sum(axis=1).mean()),
                        "brier_away": float(be[:, 0].mean()), "brier_draw": float(be[:, 1].mean()),
                        "brier_home": float(be[:, 2].mean())}
    rps_arr = {k: np.array(v) for k, v in scores.items()}
    brier_arr = {k: np.vstack(briers[k]).sum(axis=1) for k in scores}
    comparisons = []
    for base in ("equal", "home_rate", "elo"):
        comparisons.append(_compare(rps_arr["dixon_coles"], rps_arr[base], f"dixon_coles vs {base} (rps)", rng))
        comparisons.append(_compare(brier_arr["dixon_coles"], brier_arr[base], f"dixon_coles vs {base} (brier)", rng))
    comparisons.append(_compare(rps_arr["elo"], rps_arr["equal"], "elo vs equal (rps)", rng))
    return {"origins": origins, "n_test": int(o.size), "min_train": int(min_train), "block": int(block),
            "models": models, "comparisons": comparisons, "notes": notes,
            "reading": "Scores are lower-is-better. Verdicts are measured on these matches only; a verdict of "
                       "model_better means the interval for the mean difference lies below zero."}


def backtest_winloss(matches: list[Match], teams: list[str], *, half_life_days, prior_sd,
                     min_train: int | None = None, block: int | None = None) -> dict:
    """Rolling-origin backtest of the win probability against equal odds and the home-win rate, with Brier and log loss.
    Motivation vs Logic: as football, but the outcome is a home win or a loss and the margin model gives the probability."""
    played, _ = _chronological([m for m in matches if m.played])
    n = len(played)
    min_train = min_train or max(n // 2, 4 * len(teams))
    block = block or max(1, n // 10)
    if min_train >= n:
        return {"skipped": "too few matches for a backtest after the training minimum"}
    p_model, p_home, y_all = [], [], []
    origins, notes = 0, []
    for s, e in _origins(n, min_train, block):
        train, test = played[:s], played[s:e]
        try:
            model = fit_margin(train, teams=teams, half_life_days=half_life_days, prior_sd=prior_sd)
        except ValueError as exc:
            notes.append(f"origin {s}: {exc}")
            continue
        origins += 1
        rate = sum(m.home_goals > m.away_goals for m in train) / len(train)
        p_model.extend(model.home_win_prob(m.home, m.away) for m in test)
        p_home.extend([rate] * len(test))
        y_all.extend(int(m.home_goals > m.away_goals) for m in test)
    if origins == 0:
        return {"skipped": "no origin could be fitted", "notes": notes}
    p = np.array(p_model)
    y = np.array(y_all, dtype=float)
    base = {"margin_model": p, "equal": np.full(y.size, 0.5), "home_rate": np.array(p_home)}
    clip = lambda q: np.clip(q, 1e-6, 1 - 1e-6)  # noqa: E731
    models = {k: {"brier": float(np.mean((v - y) ** 2)),
                  "log_loss": float(np.mean(-(y * np.log(clip(v)) + (1 - y) * np.log(1 - clip(v)))))}
              for k, v in base.items()}
    rng = np.random.default_rng(0)
    comparisons = []
    for b in ("equal", "home_rate"):
        bs_m, bs_b = (p - y) ** 2, (base[b] - y) ** 2
        ll_m = -(y * np.log(clip(p)) + (1 - y) * np.log(1 - clip(p)))
        ll_b = -(y * np.log(clip(base[b])) + (1 - y) * np.log(1 - clip(base[b])))
        comparisons.append(_compare(bs_m, bs_b, f"margin_model vs {b} (brier)", rng))
        comparisons.append(_compare(ll_m, ll_b, f"margin_model vs {b} (log loss)", rng))
    return {"origins": origins, "n_test": int(y.size), "min_train": int(min_train), "block": int(block), "models": models,
            "comparisons": comparisons, "notes": notes,
            "reading": "Scores are lower-is-better. Verdicts are measured on these matches only."}


# ---------------------------------------------------------------- the season

def _team_index(teams: list[str]) -> dict[str, int]:
    return {t: i for i, t in enumerate(teams)}


def _football_season(played: list[Match], rem: list[Match], teams: list[str], *, n_sims: int, seed: int,
                     points, half_life_days, prior_sd, interest: list[str], relegation_places: int, top_places: int,
                     fit_extra: list[Match] = ()) -> dict:
    """Simulate the remaining football matches; ranking by points, goal difference, goals for, then name. ``fit_extra``
    (earlier seasons) is used to fit the ratings only; the table and the simulation start from ``played``."""
    model = fit_dixon_coles(_chronological(list(fit_extra) + list(played))[0], teams=teams, half_life_days=half_life_days,
                            prior_sd=prior_sd)
    T = len(teams)
    idx = _team_index(teams)
    hi = np.array([idx[m.home] for m in rem], int)
    ai = np.array([idx[m.away] for m in rem], int)
    keep = {idx[t] for t in interest}
    rng = np.random.default_rng(seed)
    chunk = max(1, 2_000_000 // max(1, len(rem)))
    parts = [_football_chunk(model, hi, ai, min(chunk, n_sims - s), rng, points, keep)
             for s in range(0, n_sims, chunk)]
    table = {r["team"]: r for r in standings(played, points)}
    base_pts = np.array([table.get(t, {}).get("points", 0) for t in teams], float)
    base_gd = np.array([table.get(t, {}).get("goal_difference", 0) for t in teams], float)
    base_gf = np.array([table.get(t, {}).get("goals_for", 0) for t in teams], float)
    total = base_pts + np.vstack([p["points"] for p in parts])
    gd = base_gd + np.vstack([p["gd"] for p in parts])
    gf = base_gf + np.vstack([p["gf"] for p in parts])
    key = total * 1e8 + (gd + 1e4) * 1e4 + gf
    pos = _positions(key)
    prob = {"title": (pos == 0).mean(0), "top_places": (pos < top_places).mean(0),
            "relegation": (pos >= T - relegation_places).mean(0), "expected_position": pos.mean(0) + 1}
    values = {t: np.vstack([p["values"][idx[t]] for p in parts]) for t in interest}
    ratings = [{"team": t, "attack": float(model.theta[i]), "defence": float(model.theta[T + i]),
                "attack_sd": float(np.sqrt(model.cov[i, i])), "defence_sd": float(np.sqrt(model.cov[T + i, T + i]))}
               for i, t in enumerate(teams)]
    settings = {"sport": FOOTBALL, "model": "Dixon-Coles Poisson with low-score correction and time decay",
                "home_advantage": model.home_advantage, "rho": model.rho, "half_life_days": half_life_days,
                "prior_sd": prior_sd, "points_rule": list(points), "converged": model.converged,
                "n_played": len(played), "n_remaining": len(rem), "top_places": top_places,
                "relegation_places": relegation_places, "n_sims": n_sims, "seed": seed}
    return {"totals": total, "prob": prob, "values": values, "ratings": ratings, "settings": settings,
            "standing": standings(played, points), "groups": None, "metric": "points"}


def _winloss_season(played: list[Match], rem: list[Match], teams: list[str], groups: dict[str, str], *, n_sims: int,
                    seed: int, half_life_days, prior_sd, interest: list[str], playoff_places: int,
                    games_per_team: int | None, fit_extra: list[Match] = ()) -> dict:
    """Simulate a win_loss season: remaining games from the fixtures, or from each team's opponent mix."""
    model = fit_margin(_chronological(list(fit_extra) + list(played))[0], teams=teams, half_life_days=half_life_days,
                       prior_sd=prior_sd)
    T = len(teams)
    idx = _team_index(teams)
    keep = {idx[t] for t in interest}
    wins_played = np.zeros(T)
    games_played = np.zeros(T)
    for m in played:
        games_played[idx[m.home]] += 1
        games_played[idx[m.away]] += 1
        wins_played[idx[m.home] if m.home_goals > m.away_goals else idx[m.away]] += 1
    rng = np.random.default_rng(seed)
    notes = []
    if rem:
        hi = np.array([idx[m.home] for m in rem], int)
        ai = np.array([idx[m.away] for m in rem], int)
        chunk = max(1, 2_000_000 // max(1, len(rem)))
        parts = [_winloss_fixture_chunk(model, hi, ai, min(chunk, n_sims - s), rng, keep)
                 for s in range(0, n_sims, chunk)]
        n_remaining = len(rem)
    else:
        plan, home_games = [], np.zeros(T)
        opp_counts = np.zeros((T, T))
        for m in played:
            opp_counts[idx[m.home], idx[m.away]] += 1
            opp_counts[idx[m.away], idx[m.home]] += 1
            home_games[idx[m.home]] += 1
        for t in range(T):
            counts = opp_counts[t].copy()
            counts[t] = 0
            mix = counts / counts.sum() if counts.sum() > 0 else np.where(np.arange(T) == t, 0.0, 1.0 / max(T - 1, 1))
            left = max(0, int(games_per_team) - int(games_played[t]))
            share = home_games[t] / games_played[t] if games_played[t] else 0.5
            plan.append({"games": left, "mix": mix, "home_share": share})
        total_rem = sum(p["games"] for p in plan)
        chunk = max(1, 2_000_000 // max(1, total_rem))
        parts = [_winloss_schedule_chunk(model, plan, min(chunk, n_sims - s), rng, keep)
                 for s in range(0, n_sims, chunk)]
        n_remaining = int(total_rem)
        notes.append("no fixtures given: each team's remaining games are drawn from its observed opponent mix, so one "
                     "team's schedule is not paired with the other team's (an approximation)")
    total = wins_played + np.vstack([p["points"] for p in parts])
    draws = np.vstack([p["ratings"] for p in parts])
    key = total * 1000 + np.clip(draws * 10, -499, 499)
    if groups:
        grp = [groups.get(t) for t in teams]
        same = np.array([[a == b for b in grp] for a in grp])
    else:
        same = np.ones((T, T), bool)
    greater = key[:, None, :] > key[:, :, None]
    rank = (greater & same[None, :, :]).sum(axis=2)
    prob = {"playoff": (rank < playoff_places).mean(0), "expected_rank_in_group": rank.mean(0) + 1}
    values = {t: np.vstack([p["values"][idx[t]] for p in parts]) for t in interest}
    ratings = [{"team": t, "rating": float(model.beta[1 + i]), "rating_sd": float(np.sqrt(model.cov[1 + i, 1 + i]))}
               for i, t in enumerate(teams)]
    settings = {"sport": WIN_LOSS, "model": "normal score-margin model, ridge-shrunk ratings, time decay",
                "home_advantage": float(model.beta[0]), "margin_sd": model.sigma, "half_life_days": half_life_days,
                "prior_sd": prior_sd, "games_per_team": games_per_team, "playoff_places": playoff_places,
                "grouped": bool(groups), "converged": True, "n_played": len(played), "n_remaining": n_remaining,
                "n_sims": n_sims, "seed": seed}
    standing = sorted(({"team": t, "wins": int(wins_played[idx[t]]), "played": int(games_played[idx[t]]),
                        "group": groups.get(t)} for t in teams), key=lambda r: (-r["wins"], r["team"]))
    return {"totals": total, "prob": prob, "values": values, "ratings": ratings, "settings": settings,
            "standing": standing, "groups": groups or None, "metric": "wins", "notes": notes}


def simulate_season(matches: list[Match], fixtures: list[Match] | None = None, *, sport: str = FOOTBALL,
                    n_sims: int = 10_000, seed: int = 0, points: tuple[int, int, int] = DEFAULT_POINTS,
                    half_life_days: float | None = DEFAULT_HALF_LIFE_DAYS, prior_sd: float = DEFAULT_PRIOR_SD,
                    teams_of_interest: list[str] | None = None, relegation_places: int = 3, top_places: int = 4,
                    playoff_places: int = 8, games_per_team: int | None = None, backtest: bool = True,
                    history: list[Match] | None = None) -> dict[str, Any]:
    """Fit the model on the played matches and simulate the rest of the season n_sims times.

    Motivation vs Logic
    Motivation: the owner asked for final-table ranges and probabilities (title, top places, relegation, or the playoff line)
    for named teams, from fitted ratings, with the backtest beside them. Logic: remaining matches come from the fixtures list,
    else from unplayed rows in the results, else (football) a double round robin completed from the results, else (win_loss)
    games_per_team per team drawn from each team's observed opponent mix. Each simulated season draws one set of ratings from
    the approximate posterior, simulates every remaining game, and ranks the teams with the sport's tie-breaks. Each team's
    range is its own and is never added across teams. Refuses an unknown sport, a small sample, or a win_loss run with no
    schedule.
    """
    if sport not in (FOOTBALL, WIN_LOSS):
        raise ValueError(f"sport must be {FOOTBALL!r} or {WIN_LOSS!r}")
    if not 10 <= n_sims <= 100_000:
        raise ValueError("simulations must be from 10 to 100000")
    everything = list(matches) + list(fixtures or [])
    played, _ = _chronological([m for m in everything if m.played])
    unplayed = [m for m in everything if not m.played]
    teams = _team_list(everything)
    # A row's group is the home team's group. A team takes its group from its home rows, else from its away rows.
    groups: dict[str, str] = {}
    for m in everything:
        if m.group and m.home not in groups:
            groups[m.home] = m.group
    for m in everything:
        if m.group and m.away not in groups:
            groups[m.away] = m.group
    notes: list[str] = []
    # Earlier seasons fit the ratings only. Early in a season (live 2026-27 en.1.json held 50 matches for 20 clubs) the
    # current season alone is too small to rate the teams. Matches with a club outside this season are left out, so a
    # promoted club is rated from this season and the shrinkage toward the mean.
    in_season = set(teams)
    fit_extra = [m for m in (history or []) if m.played and m.home in in_season and m.away in in_season]
    if history:
        notes.append(f"{len(fit_extra)} earlier matches between this season's teams were used to fit the ratings only; "
                     "the table and the simulation start from this season's results")
    rem, _ = _chronological(unplayed) if unplayed else ([], None)
    if sport == FOOTBALL and not rem:
        rem = _remaining_double_round_robin(teams, played)
        notes.append("no fixtures given: the remaining schedule is a double round robin completed from the results")
    if sport == WIN_LOSS and not rem and games_per_team is None:
        raise ValueError("win_loss with no fixtures needs games_per_team (for example 82) to schedule the rest of the season")
    current = {t: sum(_fixed_values(played, t, sport, points)) for t in teams}
    if teams_of_interest:
        interest = [t for t in dict.fromkeys(teams_of_interest) if t in current]
        unknown = sorted(set(teams_of_interest) - set(teams))
        if unknown:
            notes.append("teams not in the file were ignored: " + ", ".join(unknown))
    else:
        interest = sorted(teams, key=lambda t: (-current[t], t))[:4]

    if sport == FOOTBALL:
        core = _football_season(played, rem, teams, n_sims=n_sims, seed=seed, points=points, half_life_days=half_life_days,
                                prior_sd=prior_sd, interest=interest, relegation_places=relegation_places, top_places=top_places,
                                fit_extra=fit_extra)
    else:
        core = _winloss_season(played, rem, teams, groups, n_sims=n_sims, seed=seed, half_life_days=half_life_days,
                               prior_sd=prior_sd, interest=interest, playoff_places=playoff_places, games_per_team=games_per_team,
                               fit_extra=fit_extra)
    notes.extend(core.get("notes", []))
    totals, prob = core["totals"], core["prob"]
    team_rows = []
    for i, t in enumerate(teams):
        row = {"team": t, "current": float(current[t]), "final": _quantiles(totals[:, i]),
               "expected_final": float(totals[:, i].mean())}
        row.update({name: float(arr[i]) for name, arr in prob.items()})
        if groups.get(t):
            row["group"] = groups[t]
        team_rows.append(row)
    team_rows.sort(key=lambda r: (-r["expected_final"], r["team"]))

    paths = {}
    for t in interest:
        fixed = _fixed_values(played, t, sport, points)
        entries = [{"game": k, "median": v, "p10": v, "p90": v, "measured": True}
                   for k, v in enumerate(np.cumsum(fixed).tolist(), start=1)]
        rem_vals = core["values"].get(t)
        if rem_vals is not None and rem_vals.shape[1]:
            cum = float(sum(fixed)) + np.cumsum(rem_vals, axis=1)
            for j in range(cum.shape[1]):
                entries.append({"game": len(fixed) + j + 1, **_quantiles(cum[:, j]), "measured": False})
        paths[t] = entries

    if backtest:
        if sport == FOOTBALL:
            bt = backtest_football(fit_extra + everything, teams, half_life_days=half_life_days, prior_sd=prior_sd)
        else:
            bt = backtest_winloss(fit_extra + everything, teams, half_life_days=half_life_days, prior_sd=prior_sd)
    else:
        bt = None
    return {"sport": sport, "metric": core["metric"], "figure_label": FIGURE_LABEL,
            "settings": core["settings"], "ratings": core["ratings"], "standings": core["standing"],
            "teams": team_rows, "points_paths": paths, "backtest": bt, "notes": notes,
            "reading": ("Each team's range is its own; do not add ranges across teams. Probabilities and paths are simulated "
                        f"from fitted ratings ({FIGURE_LABEL}), not observed results.")}
