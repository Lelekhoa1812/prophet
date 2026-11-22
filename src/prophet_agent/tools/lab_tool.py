"""The control lab as a runtime tool: one call runs a bounded, leak-free experiment over candidate input treatments for
TimesFM 3 and returns a decision table.

Motivation vs Logic
Motivation: the planner can set the history window, ignore points, add drivers, z-normalise and average symmetrically,
but it has no measured evidence that any of them helps a given series. Hand-built trials are slow and easy to read by eye.
Logic: arguments are validated with readable messages before anything else. TimesFM is loaded only when the experiment
is really going to run, through a wrapper that defers the load to the first engine call. The tool is pure for a given
seed, so it is marked idempotent: a crashed call can be re-run safely.
"""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

from prophet_agent.forecast.control import build_control
from prophet_agent.forecast.lab import CANDIDATE_NAMES, MAX_HORIZON, MAX_PERIOD, run_lab
from prophet_agent.forecast.timesfm import TimesFMEngine
from prophet_agent.runtime.runner import Tool
from prophet_agent.tools.series_io import SOURCE_HELP, VALUES_HELP, resolve_series, source_covariates

ALLOWED_ARGS = {"values", "source", "horizon", "period", "candidates", "past_covariates", "future_covariates"}
MAX_OBSERVED_ROWS = 14      # the table the planner sees: the reference, the treatments and the baselines
MAX_OBSERVED_NOTES = 3      # caveats shown to the planner; the full list is kept in the ledger


def _timesfm() -> Any:
    return TimesFMEngine.get()


class _LazyEngine:
    """Defers loading TimesFM until the first forecast call, so a refused experiment never loads the model."""

    def __init__(self, factory: Callable[[], Any]):
        self._factory = factory
        self._engine: Any = None

    def forecast_controlled(self, series, horizon: int, control):
        if self._engine is None:
            self._engine = self._factory()
        return self._engine.forecast_controlled(series, horizon, control)


def _run(args: dict, _key: str, engine_factory: Callable[[], Any] | None = None) -> dict[str, Any]:
    extra = sorted(set(args) - ALLOWED_ARGS)
    if extra:
        raise ValueError(f"unknown arguments {extra}; the model is fixed to TimesFM 3; allowed: {sorted(ALLOWED_ARGS)}")
    horizon = args.get("horizon")
    if not isinstance(horizon, int) or isinstance(horizon, bool) or not 1 <= horizon <= MAX_HORIZON:
        raise ValueError(f"horizon must be an integer from 1 to {MAX_HORIZON}, in the series' own periods")
    period = args.get("period")
    if period is not None and (not isinstance(period, int) or isinstance(period, bool) or not 1 <= period <= MAX_PERIOD):
        raise ValueError(f"period must be a whole number from 1 to {MAX_PERIOD} (periods per cycle; leave it out, or 1, "
                         "for no cycle)")
    candidates = args.get("candidates")
    if candidates is not None and (not isinstance(candidates, list) or not all(isinstance(c, str) for c in candidates)):
        raise ValueError(f"candidates must be a list of names from {list(CANDIDATE_NAMES)}, or left out for the full grid")
    named = resolve_series(args)
    if len(named) != 1:
        raise ValueError(f"run the lab on one series at a time; choose with source.name from {[n for n, _ in named][:20]}")
    series = named[0][1]
    cov = build_control({"past_covariates": args.get("past_covariates"),
                         "future_covariates": args.get("future_covariates")}, len(series), horizon,
                        source_covariates(args))
    engine = _LazyEngine(engine_factory or _timesfm)
    return run_lab(series, horizon, engine, candidates, period=period,
                   past_covariates=cov.past, future_covariates=cov.future)


def observe(result: dict) -> dict:
    """Compact view for the planner: one line per row of the table, the recommendation, and the first caveats."""
    lines = []
    for r in result["table"][:MAX_OBSERVED_ROWS]:
        gain = "n/a" if r["gain_pct"] is None else f"{r['gain_pct']:+.1f}%"
        pin = "n/a" if r["pinball_gain_pct"] is None else f"{r['pinball_gain_pct']:+.1f}%"
        wins = "" if r["wins"] is None else f", wins {r['wins']}/{result['fold_count']}"
        interval = "" if r["interval_pct"] is None else f", interval [{r['interval_pct'][0]:+.1f}%, {r['interval_pct'][1]:+.1f}%]"
        lines.append(f"{r['name']} ({r['kind']}): MASE gain {gain}, pinball {pin}{wins}{interval}, {r['decision']}")
    return {
        "horizon": result["horizon"],
        "period": result["period"],
        "folds": result["fold_count"],
        "calls_spent": result["calls_spent"],
        "recommended": result["recommended"],
        "baseline_beats_default": result.get("baseline_beats_default"),
        "table": lines,
        "skipped": [f"{s['candidate']}: {s['reason']}" for s in result["skipped"][:5]],
        "failed": [f"{f['candidate']}: {f['error']}" for f in result["failed"][:5]],
        "caveats": result["caveats"][:MAX_OBSERVED_NOTES],
    }


def lab_tool(engine_factory: Callable[[], Any] | None = None) -> Tool:
    """The control_lab tool. ``engine_factory`` returns the engine; it defaults to the shared TimesFM 3 instance."""
    return Tool("control_lab", lambda args, key: _run(args, key, engine_factory), idempotent=True,
                doc="control_lab: measure which input treatments (history window, ignored outliers, znorm, symmetric "
                    "averaging, covariates) help TimesFM on this series, against the default and reference baselines.",
                title="Control lab", activity="Testing which input treatments help the forecast", observe=observe)


LAB_TOOL_SPEC = {
    "name": "control_lab",
    "title": "Control lab",
    "activity": "Testing which input treatments help the forecast",
    "description": "Run a bounded, leak-free walk-forward experiment on ONE series: each candidate input treatment for "
                   "TimesFM 3 (history windows, ignored or replaced outliers, znorm, symmetric averaging, covariates) is "
                   "forecast at the same past origins as the default, and compared with the default and with reference "
                   "baselines (naive, mean of the last 8, seasonal naive, zero and, when intermittent, TSB). Each row gives "
                   "MASE gain, RMSSE, pinball loss, the bootstrap interval, the win count and a Wilcoxon p-value, and a "
                   "decision: adopt, harmful or no evidence. A treatment is adopted only if MASE improves and pinball loss "
                   "does not get worse. Use it before presenting a forecast with a treatment; if nothing is adopted, keep "
                   "the default. Results describe this series in these windows only.",
    "args": {"values": VALUES_HELP, "source": SOURCE_HELP,
             "horizon": f"integer, 1 to {MAX_HORIZON}, in the series' own periods",
             "period": f"optional whole number of periods per cycle, 2 to {MAX_PERIOD} (leave out when there is no cycle)",
             "candidates": f"optional list of names from {list(CANDIDATE_NAMES)}; default is the full grid for this call "
                           "(the default is always run as the reference)",
             "past_covariates": "optional, drivers known only up to now: {name: list as long as the history} or a list "
                                "of names stored under 'covariates' in the source file",
             "future_covariates": "optional, drivers known ahead: {name: list as long as the history PLUS the horizon} or "
                                  "a list of names stored under 'covariates' in the source file"},
}
