"""Two blocking checks on a prediction before the lead's answer leaves: it is drawn, and it is not a flat line without a search.

Motivation vs Logic
Motivation (owner review, 2026-11-27): a flat horizontal centre line is "unacceptable for most of the time", and a prediction
that arrives as a table with no chart (the EPL mission) is not delivered. The earlier gates stopped a flat line only until a
driver test ran, and only a forecast_report pack made a chart; a league simulation made none.
Logic:
1. ``check_prediction_chart``: once a forecast, a pack or a season simulation exists, an image file (PNG or SVG) written at or
   after it must exist. A league gets a club-compared chart (expected points with the 10 to 90 range per club, and the
   position it is expected to finish in), made in sandbox_python from the simulation's own figures.
2. ``check_flat_centre``: when the delivered single-series centre line does not move visibly against its band, the finish is
   held until at least FLAT_STRATEGIES different ways of finding the shape were tried (trend rules, drivers, seasonality,
   steps and regimes, a blend, cleaning, outside research). The check names the ones not yet tried. A line that stays flat
   after that is allowed with the level sentence the earlier gate already requires.
Both run for the lead only (goal.py passes them with the horizon checks). They add no tool and never ask the model."""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

from prophet_agent.runtime import contracts
from prophet_agent.runtime.contracts import _check, _ok_result

FLAT_STRATEGIES = 4
CHART_SUFFIXES = (".png", ".svg")
# Owner choice (2026-11-27): the fundamentals check warns on the first live rounds, so what missions reach on their own is
# measured before it blocks. PROPHET_FUNDAMENTALS_GATE=block makes it a hard block.
# Measured (round 20261128-063203, warn mode): SPX, EPL and GOLD never called driver_map, and SPX delivered a drift line that
# the evaluator judged met. The check blocks by default now; PROPHET_FUNDAMENTALS_GATE=warn turns it back into a warning.
FUNDAMENTALS_SEVERITY = "warn" if os.environ.get("PROPHET_FUNDAMENTALS_GATE", "block").strip().lower() == "warn" else "block"

PREDICTION_TOOLS = ("forecast_report", "timesfm_forecast", "simulate_season", "scenario_forecast")

# Each strategy: (key, what to do). Whether it was tried is read from successful calls only (``_tried``): a forecasting
# attempt, not a word in a scan, a brief or a default argument. Round 22 (AURATE, AUUNEMP): a first version matched words
# ("trend" in a scan cell, the backtest's default "combine", open-data searches) and passed every flat line.
_STRATEGIES: tuple[tuple[str, str], ...] = (
    ("trend", "a trend rule on the cleaned history (damped trend, recent drift, momentum or a pull to a long-run level) scored "
              "on rolling origins"),
    ("steps", "a step path where the series moves in decisions or regimes (policy rates, prices set in rounds, level shifts): "
              "when the next steps are, how large, and the probability of each"),
    ("seasonal", "the cycle or calendar (season profile, day of week, holidays, fixtures) added back to the level"),
    ("drivers", "outside drivers: a related series, a leading indicator, rates, weather, fixtures, saved as covariates and "
                "scored with control_lab (with_covariates)"),
    ("blend", "a combination of several forecasters (TimesFM, a trend rule, a seasonal rule, a gradient-boosted model) weighted "
              "by rolling-origin error, delivered with forecast_report centre 'blend'"),
    ("cleaning", "cleaning before fitting: outliers (Hampel or winsorising), recording breaks, a smoothed or filtered level "
                 "(local level or Kalman), revisions and missing months, scored against the raw series"),
    ("research", "outside research for what the history cannot show (scheduled events, policy path, consensus, fixtures)"),
)
TRENDING = ("drift", "damped_trend", "recent_drift", "momentum", "mean_reversion")
_STEP_CODE = re.compile(r"step_path|stepwise|piecewise|markov|MarkovRegression|\bhmm\b|regime_switch|meeting|decision_dates", re.I)
_SEASON_CODE = re.compile(r"\bSTL\(|seasonal_decompose|fourier|holiday|season_profile|seasonal_naive", re.I)
_BLEND_CODE = re.compile(r"GradientBoosting|HistGradient|lightgbm|LGBM|xgboost|ensemble|stacking|weighted_average", re.I)
_CLEAN_CODE = re.compile(r"hampel|winsor|\.clip\(|kalman|UnobservedComponents|savgol|rolling\([^)]*\)\.median|interpolate\(", re.I)


def _flatten(history: Any) -> list[dict]:
    return [h for h in contracts.with_delegated(history) if isinstance(h, dict)]


def _lists_chart(h: dict) -> bool:
    res = h.get("result") if h.get("kind") == "tool" else None
    if not isinstance(res, dict):
        return False
    for key in ("outputs", "files"):
        for o in res.get(key) or []:
            path = o.split(" (")[0] if isinstance(o, str) else o.get("path") if isinstance(o, dict) else None
            if isinstance(path, str) and Path(path).suffix.lower() in CHART_SUFFIXES:
                return True
    return False


def check_prediction_chart(history: Any) -> dict | None:
    """Blocks while a prediction exists and no chart was written at or after it. None when there is no prediction yet."""
    flat = _flatten(history)
    last = None
    league = False
    for i, h in enumerate(flat):
        for tool in PREDICTION_TOOLS:
            if _ok_result(h, tool) is not None:
                last, league = i, tool == "simulate_season"
    if last is None:
        return None
    if any(_lists_chart(h) for h in flat[last:]):
        return _check("prediction_charted", True, "a chart of the prediction was written")
    if league:
        how = ("the league simulation gave a table and no picture. Draw a club-compared chart in sandbox_python from the "
               "simulation's figures (matplotlib, save to out/): for every club the expected final points with its 10 to 90 "
               "range, sorted by expected position, with the current points marked, and a second panel of the chance of each "
               "finishing place or of the title, top four and relegation. Then name the chart in the answer")
    else:
        how = ("the forecast was given as numbers only. Build the pack with forecast_report (it draws the history, the centre "
               "line and the 10 to 90 band), or draw the chart in sandbox_python and save it to out/. Then name the chart in "
               "the answer")
    return _check("prediction_charted", False, how)


def _flat_centre(history: Any) -> bool:
    """True when the newest single-series pack's drawn centre line does not move visibly against its band."""
    moves = contracts._drawn_line_moves([h for h in history if isinstance(h, dict)] if isinstance(history, list) else [])
    return moves is not None and not moves[0]


def _ok_call(h: dict) -> bool:
    if h.get("kind") != "tool" or h.get("error") or "result" not in h:
        return False
    res = h["result"]
    return not (isinstance(res, dict) and res.get("ok") is False)


def _tried(key: str, h: dict) -> bool:
    tool, args = h.get("tool"), h.get("args") if isinstance(h.get("args"), dict) else {}
    code = str(args.get("code") or "") if tool == "sandbox_python" else ""
    if key == "trend":
        tc = (h.get("result") or {}).get("tilt_check") if tool == "forecast_report" and isinstance(h.get("result"), dict) else None
        centre = args.get("centre")
        return bool(tc) or (tool == "forecast_report" and isinstance(centre, str) and centre in TRENDING)
    if key == "steps":
        return bool(_STEP_CODE.search(code))
    if key == "seasonal":
        return (tool in ("backtest_series", "forecast_report", "control_lab", "timesfm_forecast")
                and bool(args.get("season") or args.get("period"))) or bool(_SEASON_CODE.search(code))
    if key == "drivers":
        return contracts._signal_searched([h])
    if key == "blend":
        return (tool == "forecast_report" and (args.get("centre") == "blend" or isinstance(args.get("centre"), dict))) \
            or bool(_BLEND_CODE.search(code))
    if key == "cleaning":
        return bool(args.get("ignore_ranges")) or bool(_CLEAN_CODE.search(code))
    if key == "research":
        return tool in contracts.RESEARCH_TOOLS
    return False


def strategies_tried(history: Any) -> list[str]:
    """The keys of the strategies successful calls show (sub-agents' calls included)."""
    calls = [h for h in _flatten(history) if _ok_call(h)]
    return [key for key, _what in _STRATEGIES if any(_tried(key, h) for h in calls)]


def check_flat_centre(history: Any) -> dict | None:
    """Blocks a flat delivered centre line until FLAT_STRATEGIES different strategies were tried. None when the line moves or
    no single-series pack exists."""
    if not _flat_centre(history):
        return None
    tried = strategies_tried(history)
    if len(tried) >= FLAT_STRATEGIES:
        return _check("centre_not_flat", True, f"the centre line is flat after {len(tried)} strategies: {', '.join(tried)}")
    todo = [(k, what) for k, what in _STRATEGIES if k not in tried]
    menu = "; ".join(f"{k}: {what}" for k, what in todo[:5])
    return _check("centre_not_flat", False,
                  f"the centre line is a flat horizontal line, which is not an acceptable prediction ({len(tried)} of "
                  f"{FLAT_STRATEGIES} strategies tried: {', '.join(tried) or 'none'}). Keep going until the line has a shape "
                  f"the evidence supports (a trend, steps, a cycle). Try: {menu}. Backtest each on rolling origins, deliver the "
                  "best, and rebuild the pack with it as the centre")


DRIVER_GOAL = re.compile(r"\bdrivers?\b|\bsignals?\b|what moves|\bfactors?\b|\bleading indicators?\b", re.I)


DRIVEN_DOMAINS = ("policy_rate", "labour_market", "precious_metal", "equity_index", "currency", "commodity")


def check_signals_gathered(history: Any, goal: str = "") -> dict | None:
    """Outside signals were gathered for a single-series prediction. A goal that asks for the drivers or signals ("Search for
    the signals that move it") needs a driver tested as a covariate (a successful control_lab with_covariates or a call that
    passed covariates), and blocks without one; round 22: GOLD searched only for the price series and AURATE downloaded the
    Australian 10-year yield and never tested it. Any other prediction made from the series' own history only, with no
    driver tested and no outside research, gets a warning. None when there is no prediction or the condition holds."""
    flat = _flatten(history)
    if not any(_ok_result(h, t) is not None for h in flat for t in ("forecast_report", "timesfm_forecast")):
        return None
    tried = set(strategies_tried(history))
    from prophet_agent.research import driver_families as dfam
    # A macro or market target is driven by outside forces whatever the request's wording: live SPX (round 20261128-063203)
    # asked only to "compare against the simple rules" and delivered a drift line with no driver tested.
    if DRIVER_GOAL.search(goal or "") or dfam.detect(goal or "") in DRIVEN_DOMAINS:
        if "drivers" in tried:
            return _check("signals_gathered", True, "a driver was tested as a covariate")
        return _check("signals_gathered", False,
                      "the target is moved by outside drivers (or the request asks for them), and none was tested. Download the "
                      "candidates (related and leading series, rates, the dollar, prices, survey or consensus data), align "
                      "them to the target's dates, save them as covariates with the series, and score them with control_lab "
                      "(with_covariates); keep those that win, and say which were tested and dropped")
    if tried & {"drivers", "research"}:
        return None
    return _check("signals_gathered", False,
                  "the prediction uses the series' own history only. Gather outside signals (related and leading series, "
                  "calendar and weather, scheduled events, consensus), keep those that win on rolling origins, and combine them",
                  "warn")


# The grain a goal asks for, and the step lengths in days that deliver it.
_GRAINS: tuple[tuple[str, re.Pattern[str], tuple[float, float]], ...] = (
    ("monthly", re.compile(r"\bmonth[- ]by[- ]month\b|\bmonthly\b|\b(?:per|each|every) month\b", re.I), (28.0, 31.5)),
    ("quarterly", re.compile(r"\bquarter[- ]by[- ]quarter\b|\bquarterly\b|\b(?:per|each|every) quarter\b", re.I), (89.0, 92.5)),
    ("weekly", re.compile(r"\bweek[- ]by[- ]week\b|\bweekly\b|\b(?:per|each|every) week\b", re.I), (6.5, 7.5)),
    ("yearly", re.compile(r"\byear[- ]by[- ]year\b|\byearly\b|\bannual(?:ly)?\b|\b(?:per|each|every) year\b", re.I), (364.0, 366.5)),
)


def check_grain(goal: str, history: Any) -> dict | None:
    """The standing single-series forecast runs at the grain the goal asks for ("month by month", "quarterly", "weekly").

    Panel round 22 (SPX, 2026-11-27): "forecast it month by month over the next 2 years" was delivered as 504 business-day
    points, and the answer called them 24 months; the horizon check converted the span and passed. A step finer than the grain
    asked blocks (aggregate first: the month-end close for a price or index, the mean for a rate, the sum for a flow, then
    forecast in that grain). A step coarser than asked is a warning (derive the finer path, or say only the coarser grain is
    published). None when the goal names no grain, names several, or no forecast with a recorded step stands."""
    from prophet_agent.runtime import goal as goal_mod
    asked = [(name, span) for name, pattern, span in _GRAINS if pattern.search(goal or "")]
    if len(asked) != 1:
        return None
    name, (lo, hi) = asked[0]
    plan = goal_mod._plan_horizon([h for h in history if isinstance(h, dict)] if isinstance(history, list) else [])
    if plan is None:
        return None
    step = goal_mod._series_step(plan[1])
    if step is None:
        return None
    days = float(step[0])
    if lo <= days <= hi:
        return _check("grain_matches_request", True, f"the forecast is {name}, as asked")
    if days < lo:
        return _check("grain_matches_request", False,
                      f"the request asks for a {name} forecast, but the standing forecast steps every {days:g} days "
                      f"({plan[0]} periods). Aggregate the series to {name} first (the period-end close for a price or "
                      f"index, the mean for a rate, the sum for a flow), save it, and forecast it in {name} periods; then "
                      "rebuild the pack and describe the periods as they are")
    return _check("grain_matches_request", False,
                  f"the request asks for a {name} forecast, but the series steps every {days:g} days. Derive the {name} "
                  "path (temporal disaggregation such as Denton, or a related series published at that grain) or say "
                  f"plainly that only a coarser grain is published", "warn")


_NUMBER = re.compile(r"(?<![\w.])-?\d{1,3}(?:,\d{3})+(?:\.\d+)?|(?<![\w.])-?\d+(?:\.\d+)?")
_RANGE_PAIR = re.compile(r"(\d[\d,]*(?:\.\d+)?)\s*%?\s*(?:to|–|—|and)\s*\$?(\d[\d,]*(?:\.\d+)?)", re.I)
_RANGE_WORD = re.compile(r"\brange\b|\bband\b|\binterval\b|\b10(?:th)?\s*(?:–|-|to)\s*90|\bp10\b", re.I)


def _delivered_band(history: Any) -> tuple[float, float] | None:
    """(q10, q90) at the last step of the newest single-series pack, or of a newer scenario_forecast built on it, or None."""
    for h in reversed(history if isinstance(history, list) else []):
        sc = _ok_result(h, "scenario_forecast") if isinstance(h, dict) else None
        if isinstance(sc, dict) and isinstance(sc.get("end"), dict):
            try:
                return float(sc["end"]["q10"]), float(sc["end"]["q90"])
            except (KeyError, TypeError, ValueError):
                return None
        if isinstance(h, dict) and isinstance(_ok_result(h, "forecast_report"), dict):
            full = h.get("full") if isinstance(h.get("full"), dict) else {}
            q = ((full.get("forecast") or {}).get("quantiles") or {}) if isinstance(full.get("forecast"), dict) else {}
            lo, hi = q.get("0.1"), q.get("0.9")
            if isinstance(lo, list) and isinstance(hi, list) and lo and hi:
                return float(lo[-1]), float(hi[-1])
            return None
    return None


def _stated(text: str, value: float) -> bool:
    """True when the answer states ``value`` at some precision it uses (rounded to the number's own decimals)."""
    for m in _NUMBER.finditer(text):
        raw = m.group(0).replace(",", "")
        try:
            n = float(raw)
        except ValueError:
            continue
        decimals = len(raw.split(".")[1]) if "." in raw else 0
        if abs(n - value) <= 0.5 * 10 ** (-decimals) + 1e-9 or abs(n - value) <= 0.005 * abs(value):
            return True
    return False


def check_answer_band(answer: str, history: Any) -> dict | None:
    """The answer that states a range must state the range the delivered pack ends with.

    Panel round 22 (AUUNEMP, 2026-11-27): the pack and its chart end at 2.63 to 8.23, and the answer said "a likely range of
    4.3% to 5.5% by October 2028", figures from an earlier draft with no tool behind them. Blocks when the answer states a
    range (a range word and an "A to B" pair) and neither the delivered end-of-horizon 10th nor 90th percentile appears in it.
    None when no single-series pack or no range is stated."""
    text = answer if isinstance(answer, str) else ""
    band = _delivered_band(history)
    if band is None or not (_RANGE_WORD.search(text) and _RANGE_PAIR.search(text)):
        return None
    lo, hi = band
    if _stated(text, lo) and _stated(text, hi):
        return _check("answer_band_matches_pack", True, "the answer states the delivered end-of-horizon range")
    return _check("answer_band_matches_pack", False,
                  f"the answer states a range, but the pack it delivers ends at {lo:.4g} to {hi:.4g} (10th to 90th "
                  "percentile at the last step) and the answer does not state those figures. State the range the pack and "
                  "its chart show, or rebuild the pack that produced the range the answer gives")


def check_league_signals(history: Any) -> dict | None:
    """A football season simulation rests on more than this season's goals.

    Owner review (EPL 6e19a3b6, 2026-11-27): after 11 matchdays the simulation rated Brighton the title favourite at 45 per
    cent from a +17 goal difference that was 10.9 goals ahead of its expected-goal difference, with no earlier season and no
    bookmaker prices in the fit; with two earlier seasons, xG and closing prices the same model gives Brighton 3 per cent.
    Blocks until the newest football simulation fitted earlier seasons and at least one of expected goals or market prices.
    None when no football simulation stands."""
    flat = _flatten(history)
    for h in reversed(flat):
        res = _ok_result(h, "simulate_season")
        if not isinstance(res, dict):
            continue
        full = h.get("full") if isinstance(h.get("full"), dict) else res
        if (full.get("sport") or res.get("sport") or "football") != "football":
            return None
        settings = full.get("settings") if isinstance(full.get("settings"), dict) else (res.get("settings") or {})
        sig = settings.get("signals") if isinstance(settings.get("signals"), dict) else {}
        hist_n = int(sig.get("history_matches") or 0)
        xg = int(sig.get("xg") or 0)
        mk = int(sig.get("market") or 0) + int(sig.get("market_fixtures") or 0)
        if hist_n > 0 and (xg > 0 or mk > 0):
            return _check("league_signals", True, f"the ratings fit {hist_n} earlier-season matches, {xg} xG pairs and "
                                                  f"{mk} priced matches beside this season's results")
        missing = ([] if hist_n else ["earlier seasons (history)"]) + \
                  ([] if (xg or mk) else ["expected goals or bookmaker prices"])
        return _check("league_signals", False,
                      "the season simulation rates the clubs from " + ("this season's goals only" if not hist_n and not
                                                                        (xg or mk) else "too few signals")
                      + f"; missing: {', '.join(missing)}. A dozen matches of goals mostly measure finishing luck. Download "
                      "football-data.co.uk/mmz4281/<YYZZ>/E0.csv for this season (it carries HxG, AxG and closing prices) "
                      "and for the two seasons before (for example 2627, 2526, 2425), call simulate_season with results set "
                      "to this season's file, history set to the earlier ones and fixtures set to a file of the remaining "
                      "matches; read 'finishing' (clubs whose goal difference runs ahead of their xG) and the backtest "
                      "against goals_only and the market before quoting any chance")
    return None


_SIGNAL_WORDS = {"xg": re.compile(r"\bx ?g\b|expected[- ]goals?", re.I),
                 "market": re.compile(r"\b(?:bookmaker|betting|closing)\s+(?:odds|prices?)\b|\bmarket\s+(?:weights?|prices?|odds)\b",
                                      re.I)}
_USED = re.compile(r"\b(?:us(?:e|es|ed|ing)|weight(?:s|ed|ing)?|fit(?:s|ted)?|blend(?:s|ed)?|combin(?:es|ed)|based on|"
                   r"draws? on|includ(?:es|ed|ing)|incorporat(?:es|ed|ing)|with)\b", re.I)
_NOT = re.compile(r"\b(?:no|not|without|unavailable|missing|lack(?:s|ed)?|absent|could not|couldn't|none)\b", re.I)


def check_league_claims(answer: str, history: Any) -> dict | None:
    """The answer does not say the season model used expected goals or market prices that its fit did not have.

    Live EPL (round 20261128-063203): the answer said the ratings used xG and market weights; the simulation's settings show
    0 xG pairs and 0 priced fixtures, and the files had no HxG or AxG column. Logic: a sentence that names xG (or bookmaker
    or market prices) with a word of use (uses, weighted, fitted, based on...) and no negation is a claim of use; it blocks
    when the newest football simulation counted none of that signal. None without a simulation or such a sentence."""
    if not answer:
        return None
    for h in reversed(_flatten(history)):
        res = _ok_result(h, "simulate_season")
        if not isinstance(res, dict):
            continue
        full = h.get("full") if isinstance(h.get("full"), dict) else res
        settings = full.get("settings") if isinstance(full.get("settings"), dict) else (res.get("settings") or {})
        sig = settings.get("signals") if isinstance(settings.get("signals"), dict) else {}
        counts = {"xg": int(sig.get("xg") or 0), "market": int(sig.get("market") or 0) + int(sig.get("market_fixtures") or 0)}
        wrong = []
        for sentence in re.split(r"(?<=[.!?])\s+|\n+", answer):
            for key, rx in _SIGNAL_WORDS.items():
                if counts[key] == 0 and rx.search(sentence) and _USED.search(sentence) and not _NOT.search(sentence):
                    wrong.append(f"{key}: \"{sentence.strip()[:140]}\"")
        if not wrong:
            return None
        return _check("league_claims_match_fit", False,
                      "the answer says the model used a signal its fit did not have (the simulation counted "
                      f"{counts['xg']} xG pairs and {counts['market']} priced matches): " + "; ".join(wrong[:3])
                      + ". Say which signals the ratings were fitted on, as the simulation's settings list them")
    return None


# Calls whose result can show a driver measured against the target, and the code words of a measurement in a cell.
_TEST_TOOLS = ("control_lab", "backtest_series", "panel_backtest", "build_drivers", "timesfm_forecast", "forecast_report",
               "simulate_season")
_TEST_CODE = re.compile(r"grangercausality|\bcorr\(|corrcoef|\bccf\(|\bOLS\(|\bols\(|LinearRegression|\bRidge\(|"
                        r"\bLasso|\bVAR\(|\bARDL|SARIMAX\(.*exog|statsmodels\.api|\.fit\(\).*params|elasticit|"
                        r"lead_lag|lagged|shift\(-?\d+\).*corr|\bbeta\b", re.I | re.S)
_SELF = ("driver_map", "scenario_forecast", "notes", "board", "read_observation")


def _norm(text: str) -> str:
    t = text.strip().lower()
    t = re.sub(r"^https?://", "", t)
    t = re.sub(r"^www\.", "", t)
    return t.rstrip("/")


def _call_text(h: dict) -> str:
    body = h.get("full") if isinstance(h.get("full"), dict) else h.get("result")
    raw = json.dumps(h.get("args") or {}, default=str) + " " + json.dumps(body, default=str)
    return raw.lower().replace("\\/", "/").replace("https://", "").replace("http://", "").replace("www.", "")


def _is_test(h: dict) -> bool:
    tool = h.get("tool")
    args = h.get("args") if isinstance(h.get("args"), dict) else {}
    if tool in ("build_drivers", "simulate_season", "panel_backtest"):
        return True
    if tool in _TEST_TOOLS:
        return contracts._signal_searched([h])
    return tool == "sandbox_python" and bool(_TEST_CODE.search(str(args.get("code") or "")))


def _traced(evidence: str, texts: list[str]) -> bool:
    """True when the evidence, or one of its parts of six characters or more, appears in one of the calls' texts."""
    parts = [_norm(evidence)] + [_norm(p) for p in re.split(r"[\s,;|]+", evidence)]
    parts = [p for p in dict.fromkeys(parts) if len(p) >= 6]
    return any(p in t for p in parts for t in texts)


def check_fundamentals(history: Any, goal: str = "") -> dict | None:
    """The forecast accounts for the fundamentals that drive the target, each traced to a call in the mission.

    Owner review (2026-11-27): "the predictions on most missions haven't accounted for other fundamental factors, such as
    political and geopolitical, economical, and other factors that could drive the final decision". Logic: the goal's domain
    (research.driver_families) sets the minimum: driver families researched, families tested against the target, an
    official or market benchmark, and scenarios applied with scenario_forecast. Findings come from driver_map record calls
    (their arguments, newest per family and signal). A finding counts only when its evidence (URL, file or run id) appears
    in a successful call of this mission; a tested finding only when that call measured a driver against the target
    (covariates scored, a driver file built, a regression or lead-lag cell, or a season fitted on its signals). Blocks with
    what is missing and the uncovered families' queries. None when no prediction stands."""
    from prophet_agent.research import driver_families as dfam
    from prophet_agent.tools.fundamentals_tool import _benchmarks, _findings
    flat = _flatten(history)
    if not any(_ok_result(h, t) is not None for h in flat for t in PREDICTION_TOOLS):
        return None
    domain = dfam.detect(goal or "")
    minimum = dfam.DOMAINS[domain]["minimum"]
    calls = [h for h in flat if h.get("kind") == "tool" and h.get("tool") not in _SELF
             and _ok_result(h, str(h.get("tool"))) is not None]
    texts = [_call_text(h) for h in calls]
    test_texts = [t for h, t in zip(calls, texts) if _is_test(h)]
    findings: dict[tuple, dict] = {}
    benches: dict[str, dict] = {}
    for h in flat:
        if _ok_result(h, "driver_map") is None or not isinstance(h.get("args"), dict):
            continue
        a = h["args"]
        if a.get("action") != "record":
            continue
        dom = a.get("domain") if a.get("domain") in dfam.DOMAINS else dfam.detect(str(a.get("target") or goal))
        try:
            for f in _findings(a.get("findings"), dom):
                findings[(f["family"], f["signal"].lower())] = f
            for b in _benchmarks(a.get("benchmarks")):
                benches[b["source"].lower()] = b
        except ValueError:
            continue
    untraced = [f"{f['family']}: {f['evidence'][:60]}" for f in findings.values() if not _traced(f["evidence"], texts)]
    good = [f for f in findings.values() if _traced(f["evidence"], texts)]
    researched = sorted({f["family"] for f in good})
    tested = sorted({f["family"] for f in good if f["kind"] == "tested" and _traced(f["evidence"], test_texts)})
    n_bench = sum(1 for b in benches.values() if _traced(b["evidence"], texts)) + \
        sum(1 for f in good if f["kind"] == "benchmark")
    applied = 0
    for h in flat:
        if isinstance(_ok_result(h, "scenario_forecast"), dict) and isinstance(h.get("args"), dict):
            applied = max(applied, len(h["args"].get("scenarios") or []))
    missing = []
    if not findings:
        missing.append("no driver map: call driver_map action plan with the target, work through its families, then "
                       "driver_map action record with each finding and its evidence")
    if len(researched) < minimum["researched"]:
        missing.append(f"driver families researched with traced evidence: {len(researched)} of {minimum['researched']}")
    if len(tested) < minimum["tested"]:
        missing.append(f"families tested against the target: {len(tested)} of {minimum['tested']} (score the driver "
                       "series as covariates with control_lab with_covariates or backtest_series, or a lead-lag or "
                       "regression cell, and give that call's run id as the evidence)")
    if n_bench < minimum["benchmarks"]:
        missing.append("an official or market benchmark forecast of the target, with its value and the URL it was read "
                       "from (" + "; ".join(dfam.DOMAINS[domain]["benchmarks"][:2]) + ")")
    if applied < minimum["scenarios"]:
        missing.append(f"scenarios applied to the forecast: {applied} of {minimum['scenarios']} (state the political, "
                       "policy and geopolitical risks with a probability and a shift, and run scenario_forecast on the "
                       "pack)")
    if not missing:
        return _check("fundamentals_covered", True,
                      f"{len(researched)} driver families researched, {len(tested)} tested, {n_bench} benchmarks, "
                      f"{applied} scenarios applied ({domain})", FUNDAMENTALS_SEVERITY)
    plan = dfam.plan(goal or "", domain)
    todo = [f for f in plan["families"] if f["family"] not in researched][:5]
    hint = "; ".join(f"{f['family']} ({f['data'][0]}; search '{f['queries'][0]}')" for f in todo if f["data"] and f["queries"])
    detail = (f"the forecast does not yet account for the fundamentals that drive it ({domain}). Missing: "
              + "; ".join(missing) + ". " + (f"Families not covered yet: {hint}. " if hint else "")
              + (f"Evidence not found in any call of this mission (dropped): {', '.join(untraced[:4])}. " if untraced else "")
              + "Search broadly (research lanes per family), then measure, then record.")
    return _check("fundamentals_covered", False, detail, FUNDAMENTALS_SEVERITY)


# ---------------------------------------------------------------------------------------------- newest published level
# The target's words per domain, the verbs of a level that has been set or released, and words of a forecast (skipped).
_LEVEL_TARGET = {
    "policy_rate": re.compile(r"\b(?:cash|policy|official cash|bank|refinancing|federal funds|fed funds|repo|base|"
                              r"benchmark|interest) rate\b|\bOCR\b", re.I),
    "labour_market": re.compile(r"\b(?:unemployment|jobless) rate\b", re.I),
}
_LEVEL_SET = re.compile(r"\b(?:rais\w*|lift\w*|hik\w*|increas\w*|push\w*|cut|cuts|cutting|lower\w*|reduc\w*|rose|fell|"
                        r"climb\w*|jump\w*|eas\w*|dropp?\w*|ticked (?:up|down)|edged (?:up|down))\b[^.%\n]{0,90}?"
                        r"\bto (?:an? )?(?:[\w-]+[ -])?(?:high|low)?\s?(?:of )?(?P<v>\d{1,2}(?:\.\d{1,2})?)\s*(?:per ?cent|%)"
                        r"|\b(?:held|kept|steady|unchanged|remain\w*)\b[^.%\n]{0,60}?\bat (?P<w>\d{1,2}(?:\.\d{1,2})?)\s*"
                        r"(?:per ?cent|%)", re.I)
_LEVEL_FUTURE = re.compile(r"\b(?:expect\w*|forecast\w*|predict\w*|project\w*|will|would|could|may|might|likely|"
                           r"should|tip\w*|bet\w*|price[sd]? in|anticipat\w*|see[sn]?|if)\b", re.I)
_LEVEL_SUBGROUP = re.compile(r"\byouth|urban|rural|women|men\b|female|male|underemployment|participation", re.I)
LEVEL_SEVERITY = {"policy_rate": "block", "labour_market": "warn"}


def _dated_texts(x: Any, out: list[tuple[str, str, str]], depth: int = 0) -> None:
    """(date, url, text) for every dated source in a research result: a dict with a date and passages or a text."""
    if depth > 6:
        return
    if isinstance(x, dict):
        date = x.get("date") or x.get("published")
        texts = [p.get("text") for p in x.get("passages") or [] if isinstance(p, dict)] + [x.get("text")]
        if isinstance(date, str) and re.match(r"\d{4}-\d{2}-\d{2}", date):
            for t in texts:
                if isinstance(t, str) and t:
                    out.append((date[:10], str(x.get("final_url") or x.get("url") or ""), t))
        for v in x.values():
            if isinstance(v, (dict, list)):
                _dated_texts(v, out, depth + 1)
    elif isinstance(x, list):
        for v in x[:200]:
            _dated_texts(v, out, depth + 1)


def newest_levels(history: Any, domain: str, after: str) -> list[dict]:
    """Levels of the target that sources dated after ``after`` report as set or released (not forecast), newest first:
    {value, date, url, quote}."""
    target = _LEVEL_TARGET.get(domain)
    if target is None:
        return []
    found: list[dict] = []
    for h in _flatten(history):
        if h.get("kind") != "tool" or h.get("tool") not in contracts.RESEARCH_TOOLS:
            continue
        rows: list[tuple[str, str, str]] = []
        _dated_texts(h.get("full") if isinstance(h.get("full"), dict) else h.get("result"), rows)
        for date, url, text in rows:
            if date <= after:
                continue
            for sentence in re.split(r"(?<=[.!?])\s+|\n+", text):
                if not target.search(sentence) or _LEVEL_FUTURE.search(sentence) or _LEVEL_SUBGROUP.search(sentence):
                    continue
                m = _LEVEL_SET.search(sentence)
                if m:
                    found.append({"value": float(m.group("v") or m.group("w")), "date": date, "url": url,
                                  "quote": sentence.strip()[:200]})
    uniq = {(f["value"], f["url"]): f for f in found}
    return sorted(uniq.values(), key=lambda f: f["date"], reverse=True)


def check_newest_level(history: Any, goal: str = "") -> dict | None:
    """The forecast starts from a level published after its data ends.

    Panel round 22 (AURATE, 2026-11-27): the series ended at 4.60% on 2026-10-31 and the forecast held 4.60% for five years,
    while sources the mission itself fetched, dated 24 and 26 November, said the RBA "lifted the cash rate to an 18-year
    high of 4.85 per cent" earlier in the month. Logic: for a policy rate or an unemployment rate, sentences in research
    sources dated after the series' last observation that report the target set or released at a level (a decision or a
    release verb with "to X per cent", or "held at X per cent"; sentences of expectation or forecast are skipped) give the
    newest level, the one most sources agree on and then the newest. When it differs from the last observation and the
    forecast's first step is not near it, the finish is held (a policy rate) or warned (an unemployment rate, where a
    subgroup or a different adjustment can read differently). None when no single-series forecast or no newer level."""
    from prophet_agent.research import driver_families as dfam
    domain = dfam.detect(goal or "")
    if domain not in _LEVEL_TARGET:
        return None
    fc = None
    for h in reversed(_flatten(history)):
        if isinstance(_ok_result(h, "forecast_report"), dict):
            full = h.get("full") if isinstance(h.get("full"), dict) else h.get("result")
            fc = full.get("forecast") if isinstance(full, dict) else None
            break
    if not isinstance(fc, dict) or not isinstance(fc.get("last_observed"), dict) or not fc.get("point"):
        return None
    last = fc["last_observed"]
    try:
        end, last_v, first = str(last["date"])[:10], float(last["value"]), float(fc["point"][0])
    except (KeyError, TypeError, ValueError, IndexError):
        return None
    levels = newest_levels(history, domain, end)
    if not levels:
        return None
    counts: dict[float, int] = {}
    for f in levels:
        counts[f["value"]] = counts.get(f["value"], 0) + 1
    best = max(counts, key=lambda v: (counts[v], max(f["date"] for f in levels if f["value"] == v)))
    src = next(f for f in levels if f["value"] == best)
    tol = 0.05 if domain == "policy_rate" else 0.15
    if abs(best - last_v) <= tol:
        return None
    if abs(first - best) <= max(tol, 0.5 * abs(best - last_v)):
        return _check("newest_level_used", True, f"the forecast starts near {best:g}, the level published on {src['date']}",
                      LEVEL_SEVERITY[domain])
    return _check("newest_level_used", False,
                  f"a source dated {src['date']} ({src['url'][:90]}) reports the level at {best:g}: \"{src['quote'][:160]}\". "
                  f"The series ends at {last_v:g} on {end} and the forecast's first step is {first:g}. Add the new level to "
                  "the series as an observation at its date (the month it took effect), forecast again from it, and say in "
                  "the answer where the level came from. If the source is wrong or about another measure, say why in the "
                  "answer", LEVEL_SEVERITY[domain])


# -------------------------------------------------------------------------------------------------------- band width
BAND_NARROW, BAND_WIDE = 0.35, 2.5     # delivered 10-90 width as a share of the series' own h-step change spread
BAND_SEVERITY = "warn"                 # measured on the first live rounds before it blocks (owner choice, 2026-11-27)
_PRICE_DOMAINS = ("precious_metal", "equity_index", "commodity", "currency")


def _pack_input(run_id: Any) -> dict | None:
    from prophet_agent.sandbox.runner import SANDBOX_ROOT
    if not isinstance(run_id, str) or not re.fullmatch(r"[0-9a-f]{32}", run_id):
        return None
    try:
        return json.loads((SANDBOX_ROOT / run_id / "data" / "input.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def change_spread(history: list[float], dates: list[str] | None, h: int, log: bool, years: float = 25.0) -> float | None:
    """The 10-90 spread of the series' own h-step changes (log changes for a price) over origins in the last ``years``,
    or None with fewer than 12 origins."""
    import math
    y = [float(v) for v in history]
    if log and any(v <= 0 for v in y):
        return None
    start = 0
    if dates and len(dates) == len(y):
        try:
            import datetime as dt
            last = dt.date.fromisoformat(str(dates[-1])[:10])
            cut = last.replace(year=last.year - int(years))
            start = next((i for i, d in enumerate(dates) if dt.date.fromisoformat(str(d)[:10]) >= cut), 0)
        except ValueError:
            start = 0
    ch = [(math.log(y[t + h] / y[t]) if log else y[t + h] - y[t]) for t in range(start, len(y) - h)]
    if len(ch) < 12:
        return None
    ch.sort()
    q = lambda p: ch[min(len(ch) - 1, max(0, int(round(p * (len(ch) - 1)))))]
    return q(0.9) - q(0.1)


def check_band_width(history: Any, goal: str = "") -> dict | None:
    """The delivered end-of-horizon range is neither far narrower nor far wider than the series has moved over the horizon.

    Round 22 audit (2026-11-27): ranges were taken on trust. A band far narrower than the series' own h-step moves claims a
    certainty its history never showed; one far wider says little. Logic: the 10-90 width at the last step of the delivered
    forecast (a scenario_forecast's if one was built) against the 10-90 spread of the series' own changes over the same
    number of steps, from origins in the last 25 years (log changes for a price). Outside BAND_NARROW to BAND_WIDE times
    that spread it is flagged with both figures. None without a single-series pack and its stored history."""
    from prophet_agent.research import driver_families as dfam
    band = _delivered_band(history)
    if band is None:
        return None
    run_id = None
    for h in reversed(_flatten(history)):
        if isinstance(_ok_result(h, "forecast_report"), dict):
            full = h.get("full") if isinstance(h.get("full"), dict) else h.get("result")
            run_id = full.get("run_id") if isinstance(full, dict) else None
            break
    d = _pack_input(run_id)
    if not d or not isinstance(d.get("history"), list) or not isinstance(d.get("point"), list) or not d["point"]:
        return None
    h_steps = len(d["point"])
    log = dfam.detect(goal or "") in _PRICE_DOMAINS
    spread = change_spread(d["history"], d.get("dates"), h_steps, log)
    lo, hi = band
    if spread is None or spread <= 0 or (log and lo <= 0):
        return None
    import math
    width = math.log(hi / lo) if log else hi - lo
    ratio = width / spread
    unit = "log points" if log else "units"
    facts = (f"the delivered 10 to 90 range at the last step is {lo:.4g} to {hi:.4g} (width {width:.3g} {unit}); the "
             f"series' own {h_steps}-step changes over the last 25 years span {spread:.3g} {unit} between their 10th and "
             f"90th percentiles (ratio {ratio:.2f})")
    if ratio < BAND_NARROW:
        return _check("band_width_sane", False, facts + ". The range is far narrower than the series has moved over this "
                      "horizon: check the interval calibration on rolling origins, widen the band (forecast_report "
                      "interval calibration, or scenarios with their own uncertainty), or say why this period is calmer",
                      BAND_SEVERITY)
    if ratio > BAND_WIDE:
        return _check("band_width_sane", False, facts + ". The range is far wider than the series has moved over this "
                      "horizon: check the band was not stretched by an outlier period or a wrong scale, and narrow it with "
                      "calibration on rolling origins or tested drivers", BAND_SEVERITY)
    return _check("band_width_sane", True, facts, BAND_SEVERITY)


MARKET_NARROW = 0.6      # delivered 10-90 log width as a share of the options-implied width at the same horizon


def _horizon_months(history: Any) -> float | None:
    """Months from the last observation to the last forecast date of the newest single-series pack, or None."""
    import datetime as dt
    for h in reversed(_flatten(history)):
        if isinstance(_ok_result(h, "forecast_report"), dict):
            full = h.get("full") if isinstance(h.get("full"), dict) else {}
            fc = full.get("forecast") if isinstance(full.get("forecast"), dict) else {}
            last = (fc.get("last_observed") or {}).get("date") if isinstance(fc.get("last_observed"), dict) else None
            fut = fc.get("future_dates") or []
            try:
                days = (dt.date.fromisoformat(str(fut[-1])[:10]) - dt.date.fromisoformat(str(last)[:10])).days
            except (IndexError, TypeError, ValueError):
                return None
            return days / 30.4375 if days > 0 else None
    return None


def check_market_band(history: Any, goal: str = "") -> dict | None:
    """The delivered range is not far narrower than the options market prices at the same horizon.

    Motivation: round 20261128-063203, SPX delivered a 24-month 10-90 range of about -12.6% to +11.4% (its backtest coverage
    was 56% against 80%) while its own sources quoted a VIX of 26.6. Logic: when the mission read an options-implied range
    (market_benchmark vix or gvz), the delivered log width at the last step is compared with the implied width at the same
    number of months (implied volatility reverting to its long-run median, opendata/market_benchmark.py). Below MARKET_NARROW
    times it blocks: implied volatility usually runs a few points above realised, which the margin allows, but a range half
    as wide as the market's claims a certainty the market does not have. None without both."""
    import math
    from prophet_agent.opendata import market_benchmark as mb
    bench = None
    for h in reversed(_flatten(history)):
        r = _ok_result(h, "market_benchmark")
        if isinstance(r, dict) and r.get("kind") in ("vix", "gvz") and isinstance(r.get("implied_vol_pct"), (int, float)):
            bench = r
            break
    band, months = _delivered_band(history), _horizon_months(history)
    if bench is None or band is None or months is None or band[0] <= 0 or band[1] <= band[0]:
        return None
    vol = mb.term_vol(float(bench["implied_vol_pct"]), float(bench.get("long_run_median_pct") or bench["implied_vol_pct"]),
                      months)
    implied = 2 * mb.Z90 * vol / 100.0 * math.sqrt(months / 12.0)
    width = math.log(band[1] / band[0])
    ratio = width / implied
    facts = (f"the delivered 10 to 90 range at {months:.0f} months is {band[0]:.5g} to {band[1]:.5g} ({width:.3f} log points); "
             f"{bench['kind'].upper()} {bench['implied_vol_pct']} on {bench.get('as_of')} implies {implied:.3f} log points at that "
             f"horizon (average volatility {vol:.1f}%; ratio {ratio:.2f}; source {bench.get('source_url')})")
    if ratio < MARKET_NARROW:
        return _check("band_vs_market", False, facts + ". The range is far narrower than the options market prices: widen it "
                      "(interval calibration on rolling origins, a volatility-scaled band, or scenarios with their own "
                      "uncertainty), or give the evidence that this horizon is calmer than the market expects")
    return _check("band_vs_market", True, facts)


def check_market_stated(answer: str, history: Any) -> dict | None:
    """The answer states the market's own forecast when the mission read it, so the reader can compare it with the model's.

    Motivation: round 20261128-063203, AURATE's lanes read that the market prices cuts to about 4.34% by the end of 2027 and
    the answer left it out while its centre rose to 5.44%; SPX's sources quoted a VIX of 26.6 and futures pricing, and the
    answer named neither. Logic: with a market_benchmark result, the answer must state one of its figures: an implied
    cash rate (average or forward) for au_cash_rate, or the implied volatility for vix and gvz. None without one."""
    if not answer:
        return None
    for h in reversed(_flatten(history)):
        r = _ok_result(h, "market_benchmark")
        if not isinstance(r, dict):
            continue
        if r.get("kind") == "au_cash_rate":
            figures = [float(p["rate"]) for p in r.get("implied_forwards") or [] if isinstance(p, dict)] + \
                      [float(v) for v in (r.get("implied_average") or {}).values() if isinstance(v, (int, float))]
            what = "the cash rate the bank bill curve implies (implied_average or implied_forwards)"
        else:
            figures = [float(r["implied_vol_pct"])] if isinstance(r.get("implied_vol_pct"), (int, float)) else []
            what = f"the {str(r.get('kind')).upper()} implied volatility and the range it implies"
        if not figures:
            return None
        if any(_stated(answer, f) for f in figures):
            return _check("market_stated", True, "the answer states the market's own forecast beside the model's")
        return _check("market_stated", False,
                      f"the mission read the market's own forecast ({what}, {r.get('source_url')}, {r.get('as_of')}) and the "
                      "answer does not state it. Say what the market prices, compare it with the forecast, and say why the "
                      "forecast differs where it does")
    return None


def prediction_checks(history: Any, goal: str = "", answer: str = "") -> list[dict]:
    """The checks above that apply, in order. Never raises: a malformed observation skips its check."""
    out: list[dict] = []
    calls = (lambda: check_prediction_chart(history), lambda: check_flat_centre(history),
             lambda: check_grain(goal, history), lambda: check_answer_band(answer, history),
             lambda: check_signals_gathered(history, goal), lambda: check_league_signals(history),
             lambda: check_fundamentals(history, goal), lambda: check_newest_level(history, goal),
             lambda: check_band_width(history, goal), lambda: check_market_band(history, goal),
             lambda: check_league_claims(answer, history), lambda: check_market_stated(answer, history))
    for fn in calls:
        try:
            c = fn()
        except Exception:                      # noqa: BLE001 - a quality gate must not break a finish
            c = None
        if c is not None:
            out.append(c)
    return out
