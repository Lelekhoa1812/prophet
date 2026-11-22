"""panel_backtest: for a panel of related series, compare one global gradient-boosted model with TimesFM on the same origins.

Motivation vs Logic
Motivation: when many entities share one calendar (products, sites, branches), the question is not only "what does this
series do" but "does one model learned across all of them beat each series' own rules and the foundation model". A
single-series backtest cannot answer that.
Logic: the series come from a series.json written by ``pl.save_panel`` or ``pl.save_series`` (through series_io, so the
same checks apply). The tool runs ``forecast.global_gbm.panel_backtest`` with TimesFM as the foundation model and returns
the per-entity MASE table, the overall means, how often each model wins, the 80 percent band coverage and a one-sentence
reading. The engine can be injected for tests. The result is a measured backtest of the saved history, not a forecast.
"""
from __future__ import annotations

from typing import Any

import numpy as np
from prophet_agent.forecast import global_gbm
from prophet_agent.forecast.timesfm import TimesFMEngine
from prophet_agent.runtime.runner import Tool
from prophet_agent.tools.series_io import resolve_series

ALLOWED = {"source", "horizon", "n_origins", "season", "entities"}
MAX_PICK = 200          # most entities the person can name in one call
OBSERVE_ROWS = 30       # per-entity rows the planner sees; the ledger keeps all of them
PLAIN = {"naive": "the last value", "seasonal_naive": "the same step last season", "global_gbm": "the global model",
         "timesfm": "TimesFM", "blend": "the blend of the global model and TimesFM"}


def _int(args: dict, key: str, lo: int, hi: int, default: int | None) -> int | None:
    value = args.get(key, default)
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool) or not lo <= value <= hi:
        raise ValueError(f"{key} must be an integer from {lo} to {hi}")
    return value


def _run(args: dict, _key: str, engine_factory=None) -> dict[str, Any]:
    if args.get("source") is None or "values" in args:
        raise ValueError("panel_backtest needs source: {run_id, file} pointing at a series.json with several named series "
                         "(write them with pl.save_panel)")
    extra = sorted(set(args) - ALLOWED)
    if extra:
        raise ValueError(f"unknown arguments {extra}; allowed: source, horizon, n_origins, season, entities")
    horizon = _int(args, "horizon", 1, 512, None)
    if horizon is None:
        raise ValueError("horizon must be an integer from 1 to 512")
    n_origins = _int(args, "n_origins", 3, 12, 6)
    season = _int(args, "season", 2, 366, None)
    entities = args.get("entities")
    if entities is not None:
        if not isinstance(entities, list) or not entities or len(entities) > MAX_PICK or \
                not all(isinstance(e, str) and e for e in entities):
            raise ValueError(f"entities must be a list of 1 to {MAX_PICK} series names")
        if len(set(entities)) != len(entities):
            raise ValueError("entities must not repeat a name")
    panel = dict(resolve_series(args))
    if entities is not None:
        missing = [e for e in entities if e not in panel]
        if missing:
            raise ValueError(f"entities not in the file: {missing}; available: {sorted(panel)[:20]}")
        panel = {e: panel[e] for e in entities}
    if len(panel) < 2:
        raise ValueError("a panel needs at least 2 named series; one series belongs in backtest_series or calibrate_intervals")
    # validate everything before the model is loaded, since loading TimesFM is the slow step
    engine = engine_factory() if engine_factory else TimesFMEngine.get()
    result = global_gbm.panel_backtest(panel, horizon, n_origins=n_origins, season=season, engine=engine)
    result["reading"] = reading(result)
    result["basis"] = "measured: rolling-origin backtest on the saved history; not a forecast of the future"
    return result


def reading(res: dict[str, Any]) -> str:
    """One plain-English sentence on which model did best on this panel and how far the bands can be trusted."""
    overall, wins, best = res["overall"], res["wins"], res["best_overall"]
    n = len(res["per_entity"])
    text = (f"Over {res['origins']} rolling origins at a {res['horizon']}-step horizon across {n} entities, "
            f"{PLAIN.get(best, best)} had the lowest mean MASE ({overall[best]:.2f}) and was best on {wins[best]} of {n}")
    if best in ("naive", "seasonal_naive"):
        text += "; neither model beat the simple rule, so the extra modelling is not earning its place on this panel"
    elif "timesfm" in overall and "global_gbm" in overall:
        text += f"; the global model scored {overall['global_gbm']:.2f} and TimesFM {overall['timesfm']:.2f}"
    cov = res["coverage_80"]
    if cov:
        parts = ", ".join(f"{PLAIN.get(m, m)} {v * 100:.0f} percent" for m, v in cov.items())
        text += f". Inside the 10 to 90 percent band, the share of actual values was {parts} (target 80 percent)"
    return text + "."


def observe(result: dict) -> dict:
    keep = ("horizon", "origins", "season", "overall", "wins", "best_overall", "coverage_80", "notes", "reading", "basis")
    out = {k: result[k] for k in keep if k in result}
    rows = list(result["per_entity"].items())
    out["entities"] = len(rows)
    out["per_entity"] = [{"entity": name, **models} for name, models in rows[:OBSERVE_ROWS]]
    if len(rows) > OBSERVE_ROWS:
        # Live (electricity panel, 2026-11-22): the planner re-ran the identical call six times to see the rows left out. Say
        # how to see named entities, that a repeat changes nothing, and give a median over every entity.
        out["per_entity_note"] = (f"first {OBSERVE_ROWS} of {len(rows)} entities shown; overall, wins and the medians below cover "
                                  "all of them. To see named entities, call again with entities set to their names (up to "
                                  f"{MAX_PICK}); the same call again returns the same result")
        out["median_mase_all_entities"] = {m: round(float(np.median(vals)), 4) for m in _model_names(rows)
                                           if (vals := [float(r[1][m]) for r in rows if _finite_num(r[1].get(m))])}
    return out


def _model_names(rows) -> list[str]:
    names: list[str] = []
    for _, models in rows:
        for m in models:
            if m not in names:
                names.append(m)
    return names


def _finite_num(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and np.isfinite(v)


def panel_backtest_tool(engine_factory=None) -> Tool:
    return Tool("panel_backtest", lambda a, k: _run(a, k, engine_factory), idempotent=True, observe=observe)


PANEL_BACKTEST_SPEC = {
    "name": "panel_backtest",
    "title": "Panel backtest",
    "activity": "Comparing a global model with TimesFM across entities",
    "description": "For several related series on one calendar (a panel), test whether one global gradient-boosted model "
                   "across all of them forecasts better than each series' own naive rules and than TimesFM. It refits at "
                   "the last few origins using only earlier history and returns mean MASE per entity and overall, how often "
                   "each model wins, and 80 percent band coverage. Use it when the person has many entities that share dates "
                   "and wants to know whether pooling helps. Reads a series.json with named series written by pl.save_panel.",
    "args": {"source": "object {run_id: id of the analysis run that wrote the file, file: 'out/series.json'} with several "
                       "named series (required)",
             "horizon": "integer, 1 to 512, steps ahead to score",
             "n_origins": "optional integer, 3 to 12, rolling test windows (default 6)",
             "season": "optional integer, 2 to 366, the seasonal period (for example 7 for weekly data)",
             "entities": f"optional list of series names to compare, 1 to {MAX_PICK} names (default: all in the file)"},
}
