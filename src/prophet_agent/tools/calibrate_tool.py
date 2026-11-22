"""calibrate_intervals: measure whether TimesFM's 10 to 90 percent band covers 80 percent, and widen it if it does not.

Motivation vs Logic
Motivation: the person reads the range, not the centre line. A range that is too narrow is the costliest silent error in a
forecast. Live coverage was 72 and 82 percent against a stated 80.
Logic: ``forecast.calibrate`` fits horizon-third scales on early past windows, tests them on later ones, and returns the raw
and calibrated band with the held-out coverage of both and a recommendation. The tool never changes the point forecast.
"""
from __future__ import annotations

from typing import Any

from prophet_agent.forecast.calibrate import calibrate
from prophet_agent.forecast.control import CONTROL_ARGS, CONTROL_HELP, build_control
from prophet_agent.forecast.timesfm import TimesFMEngine
from prophet_agent.runtime.runner import Tool
from prophet_agent.tools.series_io import SOURCE_HELP, VALUES_HELP, resolve_series, source_covariates

# season and centre are accepted and ignored: the planner passes on the arguments of backtest_series and forecast_report.
# The band is calibrated around TimesFM's centre; forecast_report recentres it on a rule (live WTI: three refused calls
# with centre, then a stall).
ALLOWED = {"values", "source", "horizon", "n_origins", "season", "centre"} | CONTROL_ARGS


def _run(args: dict, _key: str, engine_factory=None) -> dict[str, Any]:
    extra = sorted(set(args) - ALLOWED)
    if extra:
        raise ValueError(f"unknown arguments {extra}; allowed: values or source, horizon, n_origins, and the control options")
    horizon = args.get("horizon")
    if not isinstance(horizon, int) or isinstance(horizon, bool) or not 1 <= horizon <= 512:
        raise ValueError("horizon must be an integer from 1 to 512")
    n_origins = args.get("n_origins", 12)
    if not isinstance(n_origins, int) or isinstance(n_origins, bool) or not 6 <= n_origins <= 24:
        raise ValueError("n_origins must be an integer from 6 to 24")
    named = resolve_series(args)
    if len(named) != 1:
        raise ValueError("calibrate one series at a time (source.name)")
    series = named[0][1]
    engine = engine_factory() if engine_factory else TimesFMEngine.get()
    control = build_control(args, len(series), horizon, source_covariates(args))
    if control.future or control.past:
        raise ValueError("calibration with covariates is not supported: past windows would need the covariates at each origin")
    out = calibrate(series, horizon, engine=engine, control=control, n_origins=n_origins)
    if args.get("centre") not in (None, "timesfm"):
        out["centre_note"] = ("centre has no effect here: the band's width is calibrated around TimesFM's centre line, and "
                              "forecast_report with the same centre and calibrate true recentres the calibrated band on the rule")
    ignored = {k: v for k, v in CALIBRATE_IGNORED.items() if k in args}       # A6-11: say what was accepted and not applied
    if ignored:
        out["ignored_arguments"] = ignored
    return out


CALIBRATE_IGNORED = {
    "season": "ignored by calibrate_intervals: the calibration fits horizon-third scales and takes no season; "
              "backtest_series takes one",
}


def observe(result: dict) -> dict:
    keep = ("horizon", "target_coverage", "origins", "tested_on", "scales_by_horizon_third", "heldout_coverage_raw",
            "heldout_coverage_calibrated", "raw_gap", "calibrated_gap", "scale_level", "second_pass", "width_ratio",
            "recommend", "reason", "caveats", "centre_note", "method", "heldout_coverage_adaptive",
            "heldout_raw_interval90", "heldout_calibrated_interval90",
            "interval_method", "heldout_windows", "selection", "ignored_arguments")
    out = {k: result[k] for k in keep if k in result}
    out["calibrated"] = {k: v[:30] for k, v in result["calibrated"].items() if k in ("q10", "q90")}
    out["raw"] = {k: v[:30] for k, v in result["raw"].items()}
    out["point"] = result["point"][:30]
    return out


def calibrate_tool(engine_factory=None) -> Tool:
    return Tool("calibrate_intervals", lambda a, k: _run(a, k, engine_factory), idempotent=True, observe=observe)


CALIBRATE_SPEC = {
    "name": "calibrate_intervals",
    "title": "Interval calibration",
    "activity": "Checking and calibrating the forecast range",
    "description": "Check whether TimesFM's 10 to 90 percent range really covers 80 percent of what happens, on past windows it "
                   "did not see, and widen it only if that is measured to help. Returns raw and calibrated ranges, the "
                   "held-out coverage of each, and a recommendation (use_calibrated or keep_raw). It never moves the point "
                   "forecast. Use it before you tell the person how far to trust the range, whenever backtest coverage is "
                   "off the 80 percent target by more than five points.",
    "args": {"values": VALUES_HELP, "source": SOURCE_HELP, "horizon": "integer, 1 to 512",
             "n_origins": "optional integer, 6 to 24, past windows to use (default 12)", **CONTROL_HELP},
}
