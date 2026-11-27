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
import re
from pathlib import Path
from typing import Any

from prophet_agent.runtime import contracts
from prophet_agent.runtime.contracts import _check, _ok_result

FLAT_STRATEGIES = 4
CHART_SUFFIXES = (".png", ".svg")
PREDICTION_TOOLS = ("forecast_report", "timesfm_forecast", "simulate_season")

# Each strategy: (key, what to do, pattern over the tool names, arguments and code a call used).
_STRATEGIES: tuple[tuple[str, str, re.Pattern[str]], ...] = (
    ("trend", "a trend rule on the cleaned history (damped trend, recent drift, momentum or a pull to a long-run level) scored "
              "on rolling origins",
     re.compile(r"damped|drift|momentum|mean_reversion|holt|trend|\"tilt_check\"", re.I)),
    ("steps", "a step path where the series moves in decisions or regimes (policy rates, prices set in rounds, level shifts): "
              "when the next steps are, how large, and the probability of each",
     re.compile(r"level[_ ]shift|regime|changepoint|change[_ ]point|piecewise|breakpoint|step[_ ]path|stepwise|hmm|markov", re.I)),
    ("seasonal", "the cycle or calendar (season profile, day of week, holidays, fixtures) added back to the level",
     re.compile(r"seasonal|season_profile|\"period\"|stl|calendar|holiday|fourier", re.I)),
    ("drivers", "outside drivers: a related series, a leading indicator, rates, weather, fixtures, saved as covariates and "
                "scored with control_lab (with_covariates)",
     re.compile(r"with_covariates|past_covariates|future_covariates|covariate|exog|driver", re.I)),
    ("blend", "a combination of several forecasters (TimesFM, a trend rule, a seasonal rule, a gradient-boosted model) weighted "
              "by rolling-origin error",
     re.compile(r"\"centre\":\s*\"blend\"|ensemble|combine|stack|global_gbm|\bgbm\b|\bblend\b", re.I)),
    ("cleaning", "cleaning before fitting: outliers (Hampel or winsorising), recording breaks, a smoothed or filtered level "
                 "(local level or Kalman), revisions and missing months",
     re.compile(r"hampel|winsor|outlier|denois|kalman|savgol|rolling_median|interpolat|reindex|deseason", re.I)),
    ("research", "outside research for what the history cannot show (scheduled events, policy path, consensus, fixtures)",
     re.compile(r"web_search|fetch_page|researcher|read_source|open_data|opendata", re.I)),
)


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


def strategies_tried(history: Any) -> list[str]:
    """The keys of the strategies the observations show: tool names, arguments and code of every call, and the tilt the pack
    itself tried when its centre was flat."""
    keys: list[str] = []
    flat = _flatten(history)
    for key, _what, pattern in _STRATEGIES:
        for h in flat:
            if h.get("kind") not in ("tool", "subagent"):
                continue
            blob = json.dumps({"tool": h.get("tool"), "args": h.get("args"), "answer": h.get("answer")}, default=str)[:20000]
            if h.get("kind") == "tool" and _ok_result(h, "forecast_report") is not None:
                blob += json.dumps((_ok_result(h, "forecast_report") or {}).get("tilt_check"), default=str)
            if pattern.search(blob) or (key == "research" and h.get("tool") in contracts.RESEARCH_TOOLS):
                keys.append(key)
                break
    return keys


def check_flat_centre(history: Any) -> dict | None:
    """Blocks a flat delivered centre line until FLAT_STRATEGIES different strategies were tried. None when the line moves or
    no single-series pack exists."""
    if not _flat_centre(history):
        return None
    tried = strategies_tried(history)
    if len(tried) >= FLAT_STRATEGIES:
        return _check("centre_not_flat", True, f"the centre line is flat after {len(tried)} strategies: {', '.join(tried)}")
    todo = [(k, what) for k, what, _ in _STRATEGIES if k not in tried]
    menu = "; ".join(f"{k}: {what}" for k, what in todo[:5])
    return _check("centre_not_flat", False,
                  f"the centre line is a flat horizontal line, which is not an acceptable prediction ({len(tried)} of "
                  f"{FLAT_STRATEGIES} strategies tried: {', '.join(tried) or 'none'}). Keep going until the line has a shape "
                  f"the evidence supports (a trend, steps, a cycle). Try: {menu}. Backtest each on rolling origins, deliver the "
                  "best, and rebuild the pack with it as the centre")


def check_signals_gathered(history: Any) -> dict | None:
    """A warning (it does not block) when a single-series prediction was delivered with no outside signal gathered: no driver
    tested as a covariate and no outside research. The planner sees it beside the blocking checks. None when there is no
    prediction or a signal was gathered."""
    flat = _flatten(history)
    if not any(_ok_result(h, t) is not None for h in flat for t in ("forecast_report", "timesfm_forecast")):
        return None
    tried = set(strategies_tried(history))
    if tried & {"drivers", "research"}:
        return None
    return _check("signals_gathered", False,
                  "the prediction uses the series' own history only. Gather outside signals (related and leading series, "
                  "calendar and weather, scheduled events, consensus), keep those that win on rolling origins, and combine them",
                  "warn")


def prediction_checks(history: Any) -> list[dict]:
    """The checks above that apply, in order. Never raises: a malformed observation skips its check."""
    out: list[dict] = []
    for fn in (check_prediction_chart, check_flat_centre, check_signals_gathered):
        try:
            c = fn(history)
        except Exception:                      # noqa: BLE001 - a quality gate must not break a finish
            c = None
        if c is not None:
            out.append(c)
    return out
