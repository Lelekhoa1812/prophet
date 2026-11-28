"""One tool that forecasts a series with TimesFM 3 and builds the chart, spreadsheet, document and PDF pack.

Motivation vs Logic
Motivation: the user should get the forecast as files they can open, not only as numbers in the chat.
Logic: the forecast comes from the same TimesFM call that the timesfm_forecast tool uses, so the files and the numbers
cannot drift apart. The pack is built inside the sandbox, and the result carries the manifest path so the files can be
verified later. The tool is idempotent: it writes only to a fresh run folder with no external effect.
"""
from __future__ import annotations

from typing import Any

from prophet_agent.artifacts import MAX_PANEL, build_forecast_pack, build_panel_pack
from prophet_agent.forecast.combine import equal_blend
from prophet_agent.forecast.control import CONTROL_ARGS, CONTROL_HELP
from prophet_agent.runtime.contracts import is_level_name
from prophet_agent.runtime.runner import Tool
from prophet_agent.tools.series_io import SOURCE_HELP, VALUES_HELP, resolve_series, run_dates, source_covariates
from prophet_agent.tools.timesfm_tool import _run as timesfm_run

MAX_TITLE = 120


def _future_dates(dates: list[str], n: int) -> list[str] | None:
    """The next ``n`` dates, continuing the history's own rhythm (daily, weekly, month-end ...). None when the dates cannot
    be read or the history has no regular step, so a chart never invents a calendar."""
    import pandas as pd
    try:
        idx = pd.DatetimeIndex(pd.to_datetime(dates))
    except (ValueError, TypeError):
        return None
    if len(idx) < 3 or not idx.is_monotonic_increasing:
        return None
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")            # pandas is changing the return type of infer_freq; the string is what is used here
        freq = pd.infer_freq(idx[-min(len(idx), 90):])
    closed = _closed_weekdays(idx)
    if freq in ("B", "C") or (freq is None and closed is not None):
        # A trading-day series: skip the days it is shut (weekends, or one shut day such as a Saturday) and the holidays of
        # the exchange calendar its history follows, when the history shows any.
        from prophet_agent.forecast.calendar import infer_calendar, next_trading_days
        cal = infer_calendar(t.date() for t in idx)
        shut = closed if closed is not None else frozenset({5, 6})
        nxt = pd.DatetimeIndex([pd.Timestamp(d) for d in next_trading_days(idx[-1].date(), n, cal["calendar"], shut)])
    elif freq:
        nxt = pd.date_range(idx[-1], periods=n + 1, freq=freq)[1:]
    else:
        gaps = idx.to_series().diff().dropna()
        step = gaps.median()
        if step <= pd.Timedelta(0) or (gaps != step).mean() > 0.1:
            return None
        nxt = pd.DatetimeIndex([idx[-1] + step * (k + 1) for k in range(n)])
    return [t.strftime("%Y-%m-%d %H:%M") if (t.hour or t.minute) else t.strftime("%Y-%m-%d") for t in nxt]


def _closed_weekdays(idx) -> frozenset[int] | None:
    """The weekdays a daily history almost never trades on: the weekend, or one shut day such as a Saturday. None when the
    history is not a daily calendar of whole days, or no weekday is shut (then the median step decides).

    Live retail e6caf00e (2026-11-23): 633 days of a shop open Monday to Sunday and shut on Saturdays. The old rule wanted
    Monday to Friday only, so the next dates were None and the answer stated no forecast window. The median-step fallback
    also refused the series: its 105 Saturday gaps were 16.6% of the steps, over the 10% cap. A later run of the same shop
    had 603 days with a London holiday calendar and two shutdowns of 5 and 12 days; the first version still refused it.
    """
    import pandas as pd
    if len(idx) < 20 or (idx.hour != 0).any():
        return None
    # A weekday is shut when it is almost never traded: one stray Saturday in a Saturday-shut year does not reopen it.
    expected = len(idx) / 7
    closed = frozenset(d for d in range(7) if (idx.day_of_week == d).sum() < 0.05 * expected)
    if not closed or 7 - len(closed) < 5:
        return None
    # Every gap may skip shut weekdays and holidays or shutdowns (which skip open days too). Gaps that skip an open weekday
    # are the unexplained ones; too many of them means random missing days, and no calendar is invented from those.
    days = idx.normalize()
    unexplained = sum(
        1 for a, b in zip(days[:-1], days[1:])
        if any(int(x.day_of_week) not in closed for x in pd.date_range(a + pd.Timedelta(days=1), b - pd.Timedelta(days=1)))
    )
    return closed if unexplained <= 0.1 * (len(idx) - 1) else None


def _event_spans(args: dict, n: int, horizon: int) -> list[dict]:
    """Known-ahead drivers that are on/off flags, as runs of positions over history plus horizon, to shade on the chart."""
    pool = source_covariates(args)
    out = []
    for name in args.get("future_covariates") or []:
        v = pool.get(name)
        if not isinstance(v, list) or len(v) != n + horizon or not set(v) <= {0, 0.0, 1, 1.0}:
            continue
        spans, start = [], None
        for i, f in enumerate(v + [0]):
            if f and start is None:
                start = i
            elif not f and start is not None:
                spans.append([start, i])
                start = None
        if spans:
            out.append({"name": str(name), "spans": spans})
    return out
ALLOWED_ARGS = {"values", "source", "horizon", "title", "calibrate", "centre", "season", "tilt", "goal"} | CONTROL_ARGS
# Every simple rule backtest_series can name as best, so the pack can deliver the one that won (live retail: the backtest's
# best rule was mean_last_season, the pack refused it, and the planner fell back to naive). "blend" is the equal-weight
# blend of TimesFM and one of those rules, delivered only when this tool's backtest says it wins (see _blend_centre).
CENTRES = ("timesfm", "naive", "drift", "seasonal_naive", "mean_last_season", "blend", "combination")
# Panel-only centres: one model fitted on every series together, set for the whole panel (never inside a per-series dict).
GLOBAL_CENTRES = ("global_gbm", "global_blend")
PANEL_CENTRES = CENTRES + GLOBAL_CENTRES
GLOBAL_NOTE = {"global_gbm": "Centre line from one gradient-boosted model trained on all series together, set by the caller because "
                             "panel_backtest ranked it best (this tool cannot see that backtest). The range is that model's calibrated band.",
               "global_blend": "Centre line: an equal-weight blend of TimesFM 3 and one gradient-boosted model trained on all series "
                               "together, set by the caller because panel_backtest ranked the blend best (this tool cannot see that "
                               "backtest). The range is the average of the two bands."}
GLOBAL_ITEM_NOTE = {"global_gbm": "centre: one model trained on all series together",
                    "global_blend": "centre: equal blend of TimesFM 3 and one model trained on all series together"}
RULE_TEXT = {"momentum": "the recent change, over the last few horizons, extended", "damped_trend": "a damped trend fitted to the history", "recent_drift": "the recent average change extended",
             "mean_reversion": "a pull toward the level of the last five cycles, with the seasonal pattern added back",
             "naive": "the last value repeated", "drift": "the average historical change extended",
             "seasonal_naive": "the value one season back repeated", "mean_last_season": "the average of the last season"}


def _blend_note(centre: dict) -> str:
    """The provenance line for an adopted blend. It says what this tool checked and what that check cannot show.

    Motivation vs Logic
    Motivation: a blend that wins on the windows that chose it is the most flattering claim the pack could make, so the note
    must carry the selection caveat in the same sentence as the win (audit item 21: say only what the tool checked).
    Logic: the note reads the evidence the tool recorded (window count, horizon, MASE of each line) and names the selection.
    It never says "more accurate" without that caveat.
    """
    ev = centre.get("evidence") or {}
    rule = RULE_TEXT[centre["with"]] + (f" (season {centre['season']})" if centre.get("season") else "")
    mase = ev.get("mase") or {}
    figures = (f" (MASE {mase.get('blend')} against {mase.get('timesfm')} for TimesFM 3 and {mase.get(centre['with'])} for the rule)"
               if mase else "")
    return (f"Centre line: an equal-weight blend of TimesFM 3 and {rule}, adopted because this tool's backtest over "
            f"{ev.get('windows')} windows of {ev.get('horizon')} periods, with the pack's settings, found the blend beat both{figures}. "
            "Selected on these windows: the rule and the adoption were both chosen on them, so these figures are optimistic and not "
            "an unbiased test of accuracy. The range is TimesFM 3's band, recentred on the blend.")


def _tilt_rules() -> tuple[str, ...]:
    from prophet_agent.forecast.direction import TILT_RULES
    return TILT_RULES


def _combination_note(centre: dict) -> str:
    """The provenance line for a combined centre: the candidates, their weights, and what the backtest measured."""
    ws = ", ".join(f"{w['name']} {w['weight']:.2f}" for w in sorted(centre.get("weights", []), key=lambda w: -w["weight"]))
    cb = centre.get("combination") or {}
    band = centre.get("band") or {}
    gain = lambda v: "n/a" if v is None else f"{v:+.1f}%"
    widened = (" The band was widened, tail by tail, because its widths would have held "
               f"{band.get('final_band_held_past_errors')} of the past errors." if band.get("widened") else "")
    return (f"Centre line and range: a weighted average of the quantiles of {len(centre.get('weights', []))} candidates "
            f"({ws}), weighted on {centre.get('origins')} backtest origins from {centre.get('first_origin')} and shrunk towards "
            f"equal weights. On those origins the combination's interval score was {gain(cb.get('vs_equal_wis_pct'))} against "
            f"equal weights and {gain(cb.get('vs_best_so_far_wis_pct'))} against picking the best candidate so far; its 10-90 "
            f"band held {cb.get('coverage', float('nan')):.0%} of outcomes.{widened}")


def _method_note(config: dict, centre: dict | None = None) -> str:
    """One line in the document that says what produced the numbers, so the file is honest on its own."""
    if centre and centre.get("rule") == "combination":
        return _combination_note(centre)
    if centre and centre.get("rule") == "blend":
        return _blend_note(centre)
    if centre and centre.get("rule") in _tilt_rules():
        control = _method_note(config, None)
        return (f"Centre line: {RULE_TEXT[centre['rule']]}. The line from the model was flat, so a trend rule was tried and "
                f"delivered: {centre['reason']}. The range is the model's band, recentred on it. {control}")
    if centre:
        rule = RULE_TEXT[centre["rule"]] + (f" (season {centre['season']})" if centre.get("season") else "")
        # The pack cannot see the backtest, so it says only what it was asked to do (audit item 21).
        return (f"Centre line: {rule}, set for this pack by the caller (the pack does not re-check the backtest that chose it); "
                "the range is TimesFM 3's band, recentred on it.")
    c = config.get("control")
    if not c:
        return "TimesFM 3, default settings, no external drivers."
    parts = []
    if c.get("ignored_points"):
        parts.append(f"{c['ignored_points']} history points ignored")
    if c.get("future_covariates"):
        parts.append("known-ahead drivers: " + ", ".join(c["future_covariates"]))
    if c.get("past_covariates"):
        parts.append("past drivers: " + ", ".join(c["past_covariates"]))
    if c.get("context_points"):
        parts.append(f"last {c['context_points']} points only")
    if c.get("znorm"):
        parts.append("standardised")
    if c.get("symmetric_averaging"):
        parts.append("symmetric averaging")
    return "TimesFM 3: " + "; ".join(parts) + "."


def _health_note(health: dict | None) -> str:
    """A sentence for the document when the forecast is flat or the check found a cause, so the file explains itself."""
    if not health:
        return ""
    v = health.get("verdict")
    if v == "flat_expected":
        return (" The median is close to flat because the history shows no repeating pattern to carry forward; read the shaded "
                "interval, not the line, for how far values can move.")
    if v == "flat_unexpected":
        return " Warning: the history repeats on a cycle that this forecast does not reproduce: " + "; ".join(health.get("causes", [])[:2]) + "."
    if v == "check" and health.get("causes"):
        return " Check: " + "; ".join(health["causes"][:2]) + "."
    if v == "ok" and health.get("median_path_note"):          # a trend with no cycle (panel round 11, CPI)
        return " The median follows the history's trend in a near-straight line; read the shaded interval for how far values can move."
    return ""


def _calibrated(fc: dict, values: list[float], args: dict) -> dict:
    """The forecast with its band replaced by the calibrated one when the held-out test says that is closer to 80 percent.
    The point forecast is never changed. When calibration cannot run (too short a series) the raw band stays and says so."""
    from prophet_agent.forecast.calibrate import calibrate
    from prophet_agent.forecast.control import build_control
    from prophet_agent.forecast.timesfm import TimesFMEngine
    from prophet_agent.tools.series_io import source_covariates
    horizon = len(fc["point"])
    try:
        control = build_control(args, len(values), horizon, source_covariates(args))
        if control.future or control.past:
            raise ValueError("calibration with covariates is not supported")
        res = calibrate(values, horizon, engine=TimesFMEngine.get(), control=control)
    except ValueError as exc:
        return {**fc, "interval_calibration": {"applied": False, "reason": str(exc)[:200]}}
    info = {"applied": res["recommend"] == "use_calibrated", "raw_heldout_coverage": res["heldout_coverage_raw"],
            "calibrated_heldout_coverage": res["heldout_coverage_calibrated"], "scales": res["scales_by_horizon_third"],
            "reason": res["reason"]}
    if not info["applied"]:
        return {**fc, "interval_calibration": info}
    # The chart caption says the band is calibrated only when the config says so (A7-13, web Results.tsx bandSource).
    config = {**(fc.get("config") or {}), "band_source": "calibrated",
              "band_coverage": info["calibrated_heldout_coverage"]}
    return {**fc, "quantiles": res["calibrated"]["quantiles"], "interval_calibration": info, "config": config}


def _rule_line(values: list[float], h: int, which: str, season: int | None) -> tuple[Any, int | None]:
    """A simple rule's centre line over the horizon, and the season it used (None for the rules without one).

    Motivation vs Logic
    Motivation: the delivered line must be the line the backtest scored, so each rule is computed in exactly one place.
    Logic: naive repeats the last value; the seasonal rules use the same season choice as backtest_series; drift extends the
    average change over the history. The single centre and the blend both read their rule from here.
    """
    import numpy as np
    last = float(values[-1])
    if which == "naive":
        return np.full(h, last), None
    if which in ("seasonal_naive", "mean_last_season"):
        # The same rules and the same season choice as the backtest, so the delivered line is the one that was scored. The
        # backtest measures an unset season on the history before its first test window (backtest_season); measuring it on the
        # whole history here gave season 7 where the backtest scored season 1 (audit: report_tool.py:225-226, 40 of 40 probes).
        from prophet_agent.forecast.rolling import DEFAULT_ORIGINS, _mean_last, _snaive, backtest_season
        x = np.asarray(values, dtype=float)
        m, _ = backtest_season(x, h, DEFAULT_ORIGINS, season)
        return (_snaive(x, h, m) if which == "seasonal_naive" else _mean_last(x, h, m)), m
    slope = (float(values[-1]) - float(values[0])) / max(len(values) - 1, 1)   # drift: the average change, extended
    return last + slope * np.arange(1, h + 1), None


def _centre(fc: dict, values: list[float], which: str, season: int | None = None, *, rule: str | None = None,
            evidence: dict | None = None) -> dict:
    """The forecast with its centre line replaced by a simple rule (the band stays TimesFM's, shifted to the new centre so the
    band keeps its width around it). ``which`` = "blend" puts the equal-weight blend of TimesFM's centre and ``rule`` as the
    centre; its evidence is the backtest that _blend_centre ran. Records what was done.

    Motivation vs Logic
    Motivation: the caller can ask for a simple rule; the pack must draw it and say exactly what it is.
    Logic: the rule's line comes from _rule_line. For a single rule the centre is that line. For the blend the centre is the
    plain mean of TimesFM's centre and that line (equal_blend, nothing fitted). Either way the band is shifted by the same
    amount at each step, so its width is kept.
    """
    import numpy as np
    h = len(fc["point"])
    old = np.asarray(fc["point"], dtype=float)
    if which == "blend":
        from prophet_agent.forecast.rolling import EQUAL_BLEND_RULES
        if rule not in EQUAL_BLEND_RULES:
            raise ValueError(f"the blend's rule must be one of {list(EQUAL_BLEND_RULES)}")
        line, m = _rule_line(values, h, rule, season)
        new = equal_blend(old, line)
        centre = {"rule": "blend", "with": rule, **({"season": int(m)} if m else {}),
                  "reason": "the backtest on these windows found the blend beat TimesFM's centre and the rule; the band is "
                            "TimesFM's, recentred on the blend", "evidence": evidence}
    else:
        new, m = _rule_line(values, h, which, season)
        centre = {"rule": which, **({"season": int(m)} if m else {}),
                  "reason": "the caller set this simple rule as the centre line; the band is TimesFM's, recentred on the rule"}
    shift = new - old
    q = {k: [round(float(v) + float(d), 4) for v, d in zip(vals, shift)] for k, vals in fc["quantiles"].items()}
    return {**fc, "point": [round(float(v), 4) for v in new], "quantiles": q, "timesfm_point": fc["point"],
            "centre": centre}


def _blend_centre(fc: dict, values: list[float], args: dict) -> dict:
    """The equal-weight blend of TimesFM's centre and the best simple rule, delivered only when this tool's own backtest over
    the same windows says it beats both. Otherwise the centre stays TimesFM's, and blend_check says why.

    Motivation vs Logic
    Motivation: a blend that is never checked is an opinion, and one checked on the windows it then claims to win is a
    selection. The report runs the check itself, so the line it draws is the line it scored, and it says what was selected.
    Logic: intermittent demand is not blended, because its centre is expected demand and the backtest scores the median.
    Otherwise run rolling_backtest with the pack's season and control and the blend switched on, and deliver the blend only
    when its centre_choice says it wins. A series too short to backtest keeps TimesFM's centre and says why.
    """
    if "median_point" in fc:
        return {**fc, "blend_check": {"adopt": False, "reason": "intermittent demand: the centre is expected demand, which the "
                                                                 "blend was not scored on"}}
    from prophet_agent.forecast.control import build_control
    from prophet_agent.forecast.rolling import rolling_backtest
    from prophet_agent.tools.series_io import source_covariates
    season = _season_arg(args)
    h = len(fc["point"])
    try:
        control = build_control(args, len(values), h, source_covariates(args))
        bt = rolling_backtest(values, h, season=season, control=control, blend_centre=True)
    except ValueError as exc:
        return {**fc, "blend_check": {"adopt": False, "reason": str(exc)[:200]}}
    ce = bt["centre_choice"]
    if not ce["adopt"]:
        return {**fc, "blend_check": ce}
    return {**_centre(fc, values, "blend", season, rule=ce["rule"], evidence=ce), "blend_check": ce}


MAX_COMBINATION_ORIGINS = 120


def _engine() -> Any:
    from prophet_agent.forecast.timesfm import TimesFMEngine
    return TimesFMEngine.get()


def _fetch(url: str) -> bytes:
    from prophet_agent.opendata import market
    return market._get(url, timeout_s=45.0, resolver=None)[0]


def _combination_centre(fc: dict, values: list[float], args: dict, name: str) -> dict:
    """The centre and band from the backtest-weighted combination of every candidate for this target (Mission 10).

    Motivation vs Logic
    Motivation (owner decision 2026-11-28, "earn it on backtest"): TimesFM, the simple rules and the domain's market and
    structural models compete on the same rolling origins, and the combination weighted by that backtest is what is drawn.
    Logic: forecast.candidates builds the candidates from the goal's domain and place; forecast.ensemble scores them on up to
    MAX_COMBINATION_ORIGINS origins one period apart and returns the combined quantiles from the end of the series. A series
    too short for the backtest keeps TimesFM's centre, and combination_check says why.
    """
    import re as _re

    import numpy as np

    from prophet_agent.forecast import candidates as cd
    from prophet_agent.forecast import ensemble as en
    from prophet_agent.research import driver_families as dfam
    h = len(fc["point"])
    goal = str(args.get("goal") or args.get("title") or "")
    domain, place = dfam.detect(goal), dfam.place_of(goal)
    target = ("gold" if _re.search(r"\bgold\b", goal, _re.I) else
              "spx" if _re.search(r"s&p|\bspx\b|sp ?500", goal, _re.I) else "")
    dates = run_dates(args, name)
    if dates is not None and len(dates) != len(values):
        dates = None
    cands, notes, rel = cd.build(values, dates or [], domain=domain, place=place, target=target, engine=_engine(),
                                 fetch=_fetch, season=_season_arg(args), drivers=source_covariates(args))
    n = len(values)
    min_train = max(24, min(60, n // 3))
    try:
        res = en.run(values, h, cands, n_origins=min(MAX_COMBINATION_ORIGINS, max(1, n - h - min_train)), step=1,
                     min_train=min_train, relative=rel)
    except ValueError as exc:
        return {**fc, "combination_check": {"adopt": False, "reason": str(exc)[:300], "notes": notes}}
    q = np.asarray(res["forecast"]["quantiles"], dtype=float)
    keys = ["0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9"]
    quant = {k: [round(float(v), 4) for v in q[:, j]] for j, k in enumerate(keys)}
    first = dates[res["origins"][0]][:10] if dates else f"value {res['origins'][0]}"
    centre = {"rule": "combination", "weights": res["weights"], "candidates": res["candidates"],
              "combination": res["combination"], "equal_weights": res["equal_weights"], "best_so_far": res["best_so_far"],
              "band": res["band"], "shrinkage": res["shrinkage"], "origins": len(res["origins"]), "first_origin": first,
              "notes": notes, "method": res["method"],
              "members_final": {k: [round(float(v), 4) for v in np.asarray(m)[:, 4]] for k, m in res["members"].items()},
              "reason": "every candidate scored on the same rolling origins; the drawn line and band are their weighted "
                        "combination", "figure_label": "measured backtest; the forecast is a model output"}
    return {**fc, "point": quant["0.5"], "quantiles": quant, "timesfm_point": fc["point"], "centre": centre,
            "combination_check": {"adopt": True}}


def _season_arg(args: dict) -> int | None:
    """season is used only by the seasonal centre rules; 1 or absent means "chosen from the data, as the backtest does"."""
    v = args.get("season")
    if v is None:
        return None
    if not isinstance(v, int) or isinstance(v, bool) or not 1 <= v <= 1000:
        raise ValueError("season must be an integer from 1 to 1000 (the cycle the backtest used)")
    return v


def _run(args: dict, key: str) -> dict[str, Any]:
    extra = sorted(set(args) - ALLOWED_ARGS)
    if extra:
        raise ValueError(f"unknown arguments {extra}; allowed: values or source, horizon, title, "
                         f"and the control options {sorted(CONTROL_ARGS)}")
    named = resolve_series(args)
    if len(named) > 1:
        return _panel(args, named, key)
    title = str(args.get("title", "Forecast"))[:MAX_TITLE] or "Forecast"
    centre = args.get("centre", "timesfm")
    if centre not in CENTRES:
        raise ValueError(f"centre must be one of {list(CENTRES)}")
    fc = timesfm_run({k: v for k, v in args.items() if k not in ("title", "calibrate", "centre", "season", "tilt", "goal")}, key)
    if "series" in fc:
        fc = next(iter(fc["series"].values()))
    values = [float(v) for v in named[0][1]]
    if args.get("calibrate") is True:
        fc = _calibrated(fc, values, args)
    if centre == "combination":
        fc = _combination_centre(fc, values, args, named[0][0])
    elif centre == "blend":
        fc = _blend_centre(fc, values, args)
    elif centre != "timesfm":
        fc = _centre(fc, values, centre, _season_arg(args))
    combined = (fc.get("centre") or {}).get("rule") == "combination"
    if args.get("tilt", True) is not False and not fc.get("median_point") and not combined:
        # A flat line over a repeating history used to skip this step and was drawn flat with a warning (owner, VN quarterly
        # unemployment, 2026-11-27: "look too flat"). The seasonal period now goes to the rules that carry the cycle.
        from prophet_agent.forecast.direction import tilt
        health = fc.get("health") or {}
        period = _season_arg(args)
        if not period or period < 2:
            period = health.get("period") if isinstance(health.get("period"), int) else None
        fc = tilt(fc, values, period)          # a flat centre gets the best-supported direction, or says why it has none
        if health.get("verdict") == "flat_unexpected" and (fc.get("tilt_check") or {}).get("adopt"):
            rule = fc["tilt_check"]["rule"]
            fc = {**fc, "health": {**fc["health"], "verdict": "check", "causes": [
                f"the model's own median missed the {period}-step cycle; the line drawn is {rule.replace('_', ' ')}, which "
                "carries the season and was chosen on backtest windows, not a validated gain"] + list(health.get("causes", []))[1:]}}
    fc = _positive_band(fc, values)
    fc = _floor_at_zero(fc, values)
    hist_dates = run_dates(args, named[0][0])
    if hist_dates is not None and len(hist_dates) != len(values):
        hist_dates = None
    ahead = _future_dates(hist_dates, len(fc["point"])) if hist_dates else None
    if ahead:
        fc = {**fc, "future_dates": ahead}     # the table's date column, read by observe from the same forecast
    q = fc["quantiles"]
    pack = build_forecast_pack(history=values, point=fc["point"], q10=q["0.1"], q90=q["0.9"], title=title,
                               note=_method_note(fc["config"], fc.get("centre")) + _health_note(fc.get("health")) + _floor_note(fc),
                               dates=hist_dates if ahead else None, future_dates=ahead,
                               events=_event_spans(args, len(values), len(fc["point"])),
                               ignored=args.get("ignore_ranges"), q30=q.get("0.3"), q70=q.get("0.7"))
    if not pack.ok:
        raise RuntimeError(f"artefact build failed: {pack.stderr[-500:]}")
    members = (fc.get("centre") or {}).get("members_final") if isinstance(fc.get("centre"), dict) else None
    if members:
        # The combination's candidates, read by scenario_forecast (path_of) to follow one candidate's path.
        import json as _json
        from pathlib import Path as _Path
        target = _Path(pack.workdir) / "data" / "members.json"
        if target.parent.is_dir():
            target.write_text(_json.dumps(members), encoding="utf-8")
    return {
        "forecast": {**fc, "level": is_level_name(title)},
        "run_id": pack.run_id,
        "manifest": f"{pack.workdir}/manifest.json",
        "files": [{"path": o.path, "sha256": o.sha256, "bytes": o.bytes} for o in pack.outputs],
    }


_RULE_TEXT = RULE_TEXT   # every rule a panel centre may name (live electricity rep8: 'seasonal_naive' raised KeyError)
BY_DAY_MAX = 31      # a horizon up to this many days is printed day by day in the panel view
PANEL_SHOWN = 30     # series with a line of their own in the observation; the rest are counted, and in the files


def _ladder(point: Any, q10: Any, q90: Any) -> dict[str, list[float]]:
    """Nine quantile levels (0.1 .. 0.9, the keys TimesFM's forecast uses) from a band with its point as the middle one.
    The levels between the given ones are linear, so the lines are the band's own."""
    import numpy as np
    levels = [round(0.1 * k, 1) for k in range(1, 10)]
    grid = np.stack([np.interp(levels, [0.1, 0.5, 0.9], [float(a), float(b), float(c)])
                     for a, b, c in zip(q10, point, q90)], axis=1)          # (levels, steps)
    return {f"{lvl:.1f}": [round(float(v), 6) for v in grid[i]] for i, lvl in enumerate(levels)}


def _global_centre(tfm: dict, pooled: dict, mode: str) -> dict:
    """One series of a panel with its centre from the pooled model (``global_gbm``), or the equal-weight blend of that model
    and TimesFM (``global_blend``): the point is the mean of the two, and every quantile level is the mean of the two bands.
    Nothing is fitted here; the pooled forecast comes from global_forecast over the whole panel."""
    import numpy as np
    gq = _ladder(pooled["point"], pooled["q10"], pooled["q90"])
    if mode == "global_gbm":
        return {**tfm, "point": [round(float(v), 6) for v in pooled["point"]], "quantiles": gq, "centre": {"rule": mode}}
    point = 0.5 * (np.asarray(tfm["point"], dtype=float) + np.asarray(pooled["point"], dtype=float))
    q = {k: [round(0.5 * (float(a) + float(b)), 6) for a, b in zip(tfm["quantiles"][k], gq[k])] for k in gq}
    return {**tfm, "point": [round(float(v), 6) for v in point], "quantiles": q, "timesfm_point": tfm["point"],
            "centre": {"rule": mode}}


def _panel_total(fcs: dict[str, dict], horizon: int, withheld: list[str]) -> dict | None:
    """The panel's total over the horizon: the sum of each series' centre-line total, with two bounds for its 80 percent range.

    Motivation vs Logic
    Motivation: the goal asked for one total across the panel, and the pack gave only each series' line, so the answer added
    the numbers by hand.
    Logic: each series' bounds come from total_range, not re-derived. The series' independent half-widths add in quadrature
    (they are taken to miss independently); the together bound is the sum of the series' together bounds. A series whose
    headline total was withheld is still summed, and ``withheld_for`` names it so the reader can see what the sum includes.
    """
    ranges = [total_range(fc) for fc in fcs.values()]
    if not ranges or any(r is None for r in ranges):
        return None
    centre = sum(r["centre"] for r in ranges)
    half = sum(((r["if_days_miss_independently"]["q90"] - r["if_days_miss_independently"]["q10"]) / 2) ** 2 for r in ranges) ** 0.5
    total = {"centre": round(centre, 2), "steps": horizon,
             "if_entities_miss_independently": {"q10": round(centre - half, 2), "q90": round(centre + half, 2)},
             "if_entities_miss_together": {"q10": round(sum(r["if_days_miss_together"]["q10"] for r in ranges), 2),
                                           "q90": round(sum(r["if_days_miss_together"]["q90"] for r in ranges), 2)},
             "note": "the 80 percent range of the panel total lies between these two bounds, under two assumptions (series move "
                     "together; series move independently)"}
    if withheld:
        total["withheld_for"] = withheld
    return total


def _panel_centres(centre: Any, names: list[str]) -> dict[str, str]:
    """The centre rule per series: one rule for every series, or a dict {series name: rule}. Series not named keep TimesFM;
    the result holds only the series whose centre is not TimesFM's."""
    blend_refused = ("on a panel the blend is the pooled model's blend, set for every series at once: pass centre 'blend' or "
                     "'global_blend' as one string, not per series. A per-series blend needs that series on its own with "
                     "forecast_report and centre 'blend'")
    if isinstance(centre, str):
        if centre == "blend":
            raise ValueError(blend_refused)
        if centre in GLOBAL_CENTRES:
            return {}                               # the pooled model is set for the whole panel by _panel
        if centre not in CENTRES:
            raise ValueError(f"centre must be one of {list(PANEL_CENTRES)}, or a dict of series name to rule")
        return {} if centre == "timesfm" else {n: centre for n in names}
    if isinstance(centre, dict):
        unknown = sorted(set(centre) - set(names))
        if unknown:
            raise ValueError(f"centre names series not in the panel: {unknown[:10]}; the panel holds {names[:20]}")
        if "blend" in centre.values():
            raise ValueError(blend_refused)
        bad = {k: v for k, v in centre.items() if v not in CENTRES}
        if bad:
            raise ValueError(f"each centre rule must be one of {list(CENTRES)}; got {bad}")
        return {k: v for k, v in centre.items() if v != "timesfm"}
    raise ValueError(f"centre must be one of {list(PANEL_CENTRES)}, or a dict of series name to rule")


def _panel(args: dict, named: list[tuple[str, Any]], key: str) -> dict[str, Any]:
    """Forecast every series of a panel in one TimesFM call and build one pack for all of them.

    Motivation vs Logic
    Motivation: panel_backtest can rank the pooled gradient-boosted model or its blend first, and the pack must be able to draw
    the line that was ranked.
    Logic: a global centre is one model fitted on every series together (global_forecast), set for the whole panel. Each
    series keeps its own TimesFM health check. The panel total is built from the centre lines and their bands (_panel_total).
    """
    import numpy as np
    names = [n for n, _ in named]
    if len(named) > MAX_PANEL:
        raise ValueError(f"a panel holds up to {MAX_PANEL} series; this file has {len(named)}. Pick up to {MAX_PANEL} with source.name")
    if args.get("calibrate") is True:
        raise ValueError("calibrate applies to one series at a time; build the panel pack without it and calibrate the series that matters")
    used = sorted(k for k in CONTROL_ARGS if args.get(k) not in (None, False, [], {}, ""))
    if used:
        raise ValueError(f"control options {used} apply to one series; build the panel with the defaults, then forecast the series "
                         "that needs them on its own with forecast_report")
    centre_arg = args.get("centre", "timesfm")
    if centre_arg == "blend":
        # panel_backtest's "blend" is the equal-weight mean of the pooled model and TimesFM, which is global_blend here. Live
        # electricity rep10 (2026-11-23): "blend" was refused twice, the pack fell back to TimesFM, and the answer still said
        # the blend was the centre line.
        centre_arg = "global_blend"
    global_mode = centre_arg if isinstance(centre_arg, str) and centre_arg in GLOBAL_CENTRES else None
    rules = {} if global_mode else _panel_centres(centre_arg, names)
    title = str(args.get("title", "Forecast"))[:MAX_TITLE] or "Forecast"
    fcs = timesfm_run({k: v for k, v in args.items() if k not in ("title", "calibrate", "centre", "season", "tilt")}, key)["series"]
    horizon = len(next(iter(fcs.values()))["point"])
    pooled, fit_info = {}, {}
    if global_mode:
        from prophet_agent.forecast.global_gbm import global_forecast
        season = _season_arg(args)
        # One model fitted on every series together. A season of 1 means "chosen from the data" to the rules, which the pooled
        # model does not take, so it is passed as no season.
        pooled = global_forecast({n: np.asarray(v, dtype=float) for n, v in named}, horizon,
                                 season=season if season and season >= 2 else None, report_into=fit_info)
    items, out_fc = [], {}
    for name, arr in named:
        values = [float(v) for v in arr]
        fc = fcs[name]
        rule = rules.get(name, "timesfm")
        if global_mode:
            fc = _global_centre(fc, pooled[name], global_mode)
        elif rule != "timesfm":
            fc = _centre(fc, values, rule, _season_arg(args))
        fc = _floor_at_zero(fc, values)
        out_fc[name] = {**fc, "level": is_level_name(title)}
        dates = run_dates(args, name)
        if dates is not None and len(dates) != len(values):
            dates = None
        ahead = _future_dates(dates, len(fc["point"])) if dates else None
        item: dict[str, Any] = {"name": name, "history": values, "point": fc["point"],
                                "q10": fc["quantiles"]["0.1"], "q90": fc["quantiles"]["0.9"]}
        if ahead:
            item.update(dates=dates, future_dates=ahead)
        if global_mode:
            item["centre_note"] = GLOBAL_ITEM_NOTE[global_mode]
        elif rule != "timesfm":
            item["centre_note"] = f"centre: {_RULE_TEXT[rule]}"
        items.append(item)
    ruled = [n for n in names if n in rules]
    flagged = [n for n in names if (out_fc[n].get("health") or {}).get("verdict") in ("check", "flat_unexpected")]
    note = f"{len(named)} series forecast together by TimesFM 3, default settings, no external drivers."
    if global_mode:
        note += " " + GLOBAL_NOTE[global_mode]
        fallback = (fit_info.get("calibration") or {}).get("fallback")
        if fallback:
            note += f" Band calibration: {fallback}."
    if ruled:
        note += " Centre line from a simple rule, where the backtest found it more accurate: " + ", ".join(ruled[:20]) + "."
    if flagged:
        note += " Read the health check for: " + ", ".join(flagged[:20]) + "."
    floored = [n for n in names if out_fc[n].get("floor")]
    if floored:
        note += " Days below zero were floored at zero (history never negative) in: " + ", ".join(floored[:20]) + "."
    withheld = [n for n in names if _headline(out_fc[n])["total"] is None]
    pack = build_panel_pack(series=items, title=title, note=note)
    if not pack.ok:
        raise RuntimeError(f"artefact build failed: {pack.stderr[-500:]}")
    return {
        "forecast": {"series": out_fc},
        "n_series": len(named),
        "run_id": pack.run_id,
        "manifest": f"{pack.workdir}/manifest.json",
        "files": [{"path": o.path, "sha256": o.sha256, "bytes": o.bytes} for o in pack.outputs],
        # A level (PM2.5, a price) has no total over series or days: the pack withholds it at source, as the contracts do for
        # the answer. Live Beijing d861b74e (2026-11-23): the planner took this total, 12,222, and wrote 873 as a daily average.
        "panel_total": None if is_level_name(title) else _panel_total(out_fc, horizon, withheld),
    }


def _panel_observe(result: dict) -> dict:
    items = list(result["forecast"]["series"].items())
    lines = {}
    for name, fc in items[:PANEL_SHOWN]:
        p, q, health = fc["point"], fc["quantiles"], fc.get("health") or {}
        head = _headline(fc)
        rule = (fc.get("centre") or {}).get("rule", "timesfm")
        # The first step's band is in the view too. Live Beijing d1a1d077 (2026-11-23): the view gave only the last step's band,
        # and the answer wrote day-1 bands that were another station's (Aotizhongxin 8.8-131.2; the stored band is 4.5-114.2).
        line = {"point_step_1": p[0], "q10_step_1": q["0.1"][0], "q90_step_1": q["0.9"][0],
                "point_last_step": p[-1], "q10_last_step": q["0.1"][-1], "q90_last_step": q["0.9"][-1],
                "centre": rule, "health": health.get("verdict"),
                "range_vs_centre_pct": range_vs_centre(fc), "centre_line_total": head["total"]}
        if len(p) <= BY_DAY_MAX:
            # Every day's figures, so a table copies each station's own day (live Beijing 0137871e, 2026-11-23: the view gave
            # only the first and last steps, and the table's other days were not in the evidence). Printed to one decimal.
            line["by_day"] = [f"{a:.1f} ({b:.1f} to {c:.1f})" for a, b, c in zip(p, q["0.1"], q["0.9"])]
        if rule != "timesfm":
            # The health check reads TimesFM's own line; with another centre it is not a verdict on the line drawn
            # (live electricity rep14: flat_expected for all 60 clients under the pooled blend).
            line["health_of"] = "TimesFM's line, not the drawn centre"
        if _window(fc):
            line["window"] = _window(fc)
        if head["withheld"]:
            line["total_withheld"] = head["withheld"]
        if health.get("verdict") in ("check", "flat_unexpected") and health.get("causes"):
            line["causes"] = health["causes"][:2]
        lines[name] = line
    cfgs = [fc.get("config") or {} for _, fc in items]
    ctx_used = [c["context_used"] for c in cfgs if isinstance(c.get("context_used"), int)]
    config = {"horizon": len(items[0][1]["point"]) if items else None,
              "context_points": ([min(ctx_used), max(ctx_used)] if ctx_used and min(ctx_used) != max(ctx_used)
                                 else (ctx_used[0] if ctx_used else None)),
              "model": next((c.get("checkpoint") for c in cfgs if c.get("checkpoint")), None)}
    return {"n_series": result["n_series"], "series": lines, "config": config,
            **({"not_shown": len(items) - PANEL_SHOWN} if len(items) > PANEL_SHOWN else {}),
            **({"panel_total": result["panel_total"]} if result.get("panel_total") else {}),
            "run_id": result["run_id"], "files": [f["path"] for f in result["files"]], "manifest": result["manifest"],
            "deliverables": "one chart grid per 12 series (forecast_grid.png and .svg, then forecast_grid_2 and so on), a workbook "
                            "with Summary and Forecast sheets, a report and a PDF, all listed in the manifest; no further build is needed"}


TABLE_FULL = 120     # a horizon up to this many periods is tabulated in full (timesfm_tool._compact thins beyond it)
TABLE_EDGE = 5       # beyond TABLE_FULL, this many rows from each end are shown and the rest are named


def _table(fc: dict) -> dict:
    """The forecast's rows, read from the same arrays the pack writes to out/forecast.json (audit A7-02).

    Motivation vs Logic
    Motivation: the answer printed a table the planner wrote from memory: day 1 showed day 14's band and the totals did not
    match the file. The planner needs each step's own numbers, exactly as they are in the file, so it can quote them.
    Logic: one row per step, with the floats the pack was given (no rounding), so a quoted value equals the file's value.
    A horizon above TABLE_FULL keeps its first and last TABLE_EDGE rows and names the file for the rest.
    """
    point, q = fc["point"], fc["quantiles"]
    dates = fc.get("future_dates")
    rows = []
    for i, p in enumerate(point):
        row = {"step": i + 1, "point": float(p), "q10": float(q["0.1"][i]), "q90": float(q["0.9"][i])}
        if dates:
            row["date"] = dates[i]
        rows.append(row)
    out = {"file": "out/forecast.json", "columns": ["step", "date", "point", "q10", "q90"] if dates else ["step", "point", "q10", "q90"]}
    if len(rows) > TABLE_FULL:
        out.update(rows=rows[:TABLE_EDGE] + rows[-TABLE_EDGE:], rows_omitted=len(rows) - 2 * TABLE_EDGE)
    else:
        out.update(rows=rows, rows_omitted=0)
    return out


def _window(fc: dict) -> dict | None:
    """The first and last forecast dates and the step count, so the answer can state the period it covers.

    Motivation vs Logic
    Motivation (live retail e6caf00e, rep9, 2026-11-23): the forecast had its dates, and the table carried them, but no line
    named the window, so the answer stated the total without its period. The window is now a field of its own.
    Logic: the first and last of ``future_dates``; None when the forecast has no dates.
    """
    ahead = fc.get("future_dates")
    if not ahead:
        return None
    return {"first": ahead[0], "last": ahead[-1], "steps": len(ahead)}


def observe(result: dict) -> dict:
    from prophet_agent.tools.timesfm_tool import _compact
    if "n_series" in result:
        return _panel_observe(result)
    # The run id is what proves where the files are: without it the planner cannot cite or re-open the workbook and asks
    # where it went (seen live, as "the forecast workbook could not be found").
    out = {"forecast": _compact(result["forecast"]), "run_id": result["run_id"],
           "files": [f["path"] for f in result["files"]], "manifest": result["manifest"],
           "range_vs_centre_pct": range_vs_centre(result["forecast"]),
           "headline": _headline(result["forecast"]),
           "centre_line_total": _headline(result["forecast"])["total"],
           "total_range_80": total_range(result["forecast"]),
           "table": _table(result["forecast"]),
           "deliverables": "built and listed in the manifest; no further build is needed"}
    if _window(result["forecast"]):
        out["window"] = _window(result["forecast"])
    if "combination_check" in result["forecast"]:
        out["combination_check"] = result["forecast"]["combination_check"]
    centre = result["forecast"].get("centre")
    if isinstance(centre, dict) and centre.get("rule") == "combination":
        cb = centre.get("combination") or {}
        out["combination"] = {
            "weights": {w["name"]: w["weight"] for w in centre.get("weights", [])},
            "scores": [f"{c['name']} ({c['kind']}): WIS {c['wis']:.4g}, MASE {c['mase']:.3g}, 10-90 coverage "
                       f"{c['coverage']:.0%}, {c['origins']} origins" for c in centre.get("candidates", [])],
            "combination": f"WIS {cb.get('wis', float('nan')):.4g}, coverage {cb.get('coverage', float('nan')):.0%}; against "
                           f"equal weights {cb.get('vs_equal_wis_pct')}%, against the best candidate so far "
                           f"{cb.get('vs_best_so_far_wis_pct')}%, against the best single in hindsight "
                           f"{cb.get('vs_best_single_wis_pct')}% ({cb.get('best_single')})",
            "origins": f"{centre.get('origins')} from {centre.get('first_origin')}", "band": centre.get("band"),
            "members_final_median": centre.get("members_final"), "notes": centre.get("notes"),
            "state": "state the weights and the scores beside the forecast; a candidate with little weight still bounds the "
                     "scenarios"}
    if "blend_check" in result["forecast"]:       # the blend was asked for: say whether its own backtest adopted it, and why
        out["blend_check"] = result["forecast"]["blend_check"]
    if result["forecast"].get("tilt_check"):      # the centre was flat: what trend rules were tried, and whether one was delivered
        tc = result["forecast"]["tilt_check"]
        out["tilt_check"] = {"adopt": tc.get("adopt"), "rule": tc.get("rule"), "reason": tc.get("reason"),
                             "tried": {k: v.get("result") for k, v in (tc.get("tried") or {}).items()}}
    if isinstance(result["forecast"].get("centre"), dict):
        # The centre the pack draws. The gate reads it to judge baseline_compared; _compact drops it. The blend's evidence is
        # the backtest behind the choice and is left out of the observation to keep it short.
        out["forecast"]["centre"] = {k: v for k, v in result["forecast"]["centre"].items()
                                     if k not in ("evidence", "candidates", "members_final", "weights", "method")}
    return out


def _total(fc: dict) -> float | None:
    """The sum of the centre line over the horizon (live retail: an answer quoted the 14-day total, which no tool printed).
    It is the total of the line, not a quantile of the total: the band's edges do not add up across days."""
    try:
        return round(float(sum(float(v) for v in fc["point"])), 2)
    except (KeyError, TypeError, ValueError):
        return None


BAND_FLOOR_SHARE = 0.25     # a lower quantile under this share of the centre means the additive band is too wide for a positive series


def _positive_band(fc: dict, values: list[float]) -> dict:
    """For a strictly positive history (a price, a rate), a lower quantile that falls to a small share of the centre line is
    rebuilt in logarithms: each lower quantile is the centre squared over the matching upper quantile, so the band is symmetric
    in ratio and stays above zero. The upper quantiles are not changed.

    Motivation vs Logic
    Motivation (owner review of round 14 charts, 2026-11-27): the NVDA band reached zero from the 29th day, a flat floor under
    a price that has not been near it. An additive width on a price is not a range anyone can use.
    Logic: only when min(history) > 0 and some step's 10 percent quantile is under BAND_FLOOR_SHARE of its centre. Recorded
    in ``band_shape`` so the observation and the pack's note say what was changed.
    """
    import numpy as np
    if not values or min(values) <= 0:
        return fc
    point = np.asarray(fc["point"], dtype=float)
    q = fc["quantiles"]
    if not (point > 0).all() or not (np.asarray(q["0.1"], dtype=float) < BAND_FLOOR_SHARE * point).any():
        return fc
    out = dict(q)
    for lvl in ("0.1", "0.2", "0.3", "0.4"):
        mirror = f"{round(1 - float(lvl), 1):.1f}"
        if mirror in q:
            upper = np.maximum(np.asarray(q[mirror], dtype=float), point)
            out[lvl] = [round(float(v), 4) for v in point * point / upper]
    return {**fc, "quantiles": out, "band_shape": {"log_symmetric": True,
            "reason": "the additive lower band fell to under a quarter of the centre line on a series that stays positive"}}


def _floor_at_zero(fc: dict, values: list[float]) -> dict:
    """For a series whose history is never negative (sales, counts, prices), a forecast day below zero is not a forecast:
    the point and the band are floored at zero, and the forecast records what was floored. Otherwise the forecast is unchanged.

    Motivation vs Logic
    Motivation (audit A7-03): a 14-day total with a day at minus 3,606 was the headline of a series that cannot go below zero.
    Logic: only when the history has no negative value and the forecast does. The floor is recorded (``floor``) so the
    observation and the pack's note say what was changed, and the headline total is withheld (see _headline).
    """
    if not values or min(values) < 0:
        return fc                       # a series that goes negative in its history (net sales, returns) keeps its forecast
    raw = [float(v) for v in fc["point"]]
    if min([*raw, *(float(v) for vs in fc["quantiles"].values() for v in vs)]) >= 0:
        return fc
    days = sum(1 for v in raw if v < 0)
    # "applied" means a centre day was below zero, which is what withholds the total. A band that dips below zero is floored
    # too (it cannot be below zero for this series), but on its own it does not change the total.
    floor = {"applied": days > 0, "days_below_zero": days, "raw_min_point": round(min(raw), 3)}
    return {**fc, "point": [max(0.0, v) for v in raw],
            "quantiles": {k: [max(0.0, float(v)) for v in vs] for k, vs in fc["quantiles"].items()}, "floor": floor}


def _floor_note(fc: dict) -> str:
    """The pack's note for a floored forecast, in words a reader can check against the file."""
    shape = (" The lower range is symmetric in ratio with the upper (the additive range would have reached zero)."
             if (fc.get("band_shape") or {}).get("log_symmetric") else "")
    f = fc.get("floor") or {}
    if shape and not f.get("applied"):
        return shape
    if not f.get("applied"):
        return ""
    return (f" Forecast days below zero: {f['days_below_zero']} (lowest raw day {f['raw_min_point']:,}). The history never goes "
            "below zero, so these days are floored at zero in the chart, the workbook and the table.")


# flat_expected is a valid verdict: the history has no repeating pattern, so the level line and its range are the forecast.
HEADLINE_VERDICTS = ("ok", "flat_expected")


def _headline(fc: dict) -> dict:
    """The one total an answer may headline, with the health verdict beside it (audit A7-03).

    Motivation vs Logic
    Motivation: the headline total was shown next to a health verdict of flat_unexpected and a day below zero, and the answer
    text called the verdict ok. The total must not stand alone when the check that judges the forecast did not pass.
    Logic: the total is shown only when the health verdict is ok or flat_expected and no day had to be floored. Otherwise it is withheld and
    ``withheld`` says why, in words, so the answer can show the reason first. The verdict is always reported.
    """
    verdict = (fc.get("health") or {}).get("verdict")
    reasons = []
    if fc.get("level"):
        # A level (PM2.5, a price, a rate) has no total over days: the sum of its daily values is not a figure anyone reports.
        # Live Beijing 3581d07d and d861b74e (2026-11-23): the per-station "14-day total" was still emitted after the panel total
        # was withheld. The average over the horizon and each day's own values are what the pack gives.
        reasons.append("the measure is a level (a price, rate or concentration), so its total over the horizon means nothing; "
                       "give the average over the horizon and each day's own values instead")
    if (fc.get("floor") or {}).get("applied"):
        reasons.append(f"{fc['floor']['days_below_zero']} forecast day(s) were below zero in a series whose history never is, "
                       "so they are floored at zero and the total is not a headline")
    if verdict not in HEADLINE_VERDICTS:
        reasons.append(f"the health check verdict is {verdict or 'missing'}, not ok")
    return {"verdict": verdict, "total": None if reasons else _total(fc), "withheld": "; ".join(reasons) or None}


Z90 = 1.2815515655446004          # the 90th percentile of a standard normal: the 10-90 band is +-Z90 standard deviations


def total_range(fc: dict) -> dict | None:
    """An 80 percent range for the TOTAL over the horizon, which is not the sum of the days' band edges.

    Live retail: the planner added the per-day q10 and q90 into "329,049 to 851,044" and called it the range of the 14-day total.
    That is only right if every day misses in the same direction by the same amount, so it overstates. Two bounds are given:
    ``if_days_miss_independently`` (the days' standard deviations, read from each band, add in quadrature) and
    ``if_days_miss_together`` (the edges add: the widest the total could be). The range under two assumptions lies between them; the answer states both
    and never one of them as exact. Normal approximation, centred on the centre line's total.
    """
    if fc.get("level"):
        return None                       # a level has no range for a total (see _headline)
    try:
        point, lo, hi = fc["point"], fc["quantiles"]["0.1"], fc["quantiles"]["0.9"]
        centre = sum(float(v) for v in point)
        sd = [(float(h) - float(l)) / (2 * Z90) for l, h in zip(lo, hi)]
        spread = Z90 * sum(v * v for v in sd) ** 0.5
        n = len(point)
        # The average over the horizon is the total over the step count, and so is its range: an average's range is not the
        # mean of the daily ranges (live wti_budget averaged the daily bounds by hand, and the figure could not be grounded).
        return {"centre": round(centre, 2), "steps": n, "mean_per_step": round(centre / n, 2),
                "if_days_miss_independently": {"q10": round(centre - spread, 2), "q90": round(centre + spread, 2),
                                               "mean_q10": round((centre - spread) / n, 2), "mean_q90": round((centre + spread) / n, 2)},
                "if_days_miss_together": {"q10": round(sum(float(v) for v in lo), 2), "q90": round(sum(float(v) for v in hi), 2),
                                          "mean_q10": round(sum(float(v) for v in lo) / n, 2),
                                          "mean_q90": round(sum(float(v) for v in hi) / n, 2)},
                "note": "the 80 percent range of the total, under two assumptions (days move together; days move independently), lies between these two bounds; the edges of different days do not simply add"}
    except (KeyError, TypeError, ValueError):
        return None


def range_vs_centre(fc: dict) -> dict | None:
    """How far the 10-90 percent band reaches below and above the centre line on the first and last step, in percent of the
    centre. Answers quote these ("day 30: +3.2% / -3.5%"); live, the grounding check flagged them because no tool had
    printed them."""
    try:
        point, lo, hi = fc["point"], fc["quantiles"]["0.1"], fc["quantiles"]["0.9"]
        out = {}
        for name, i in (("first_step", 0), ("last_step", len(point) - 1)):
            c = float(point[i])
            if c == 0:
                return None
            out[name] = {"below_pct": round((float(lo[i]) - c) / abs(c) * 100, 1),
                         "above_pct": round((float(hi[i]) - c) / abs(c) * 100, 1)}
        return out
    except (KeyError, IndexError, TypeError, ValueError):
        return None


def forecast_report_tool() -> Tool:
    return Tool("forecast_report", _run, idempotent=True, observe=observe)


FORECAST_REPORT_SPEC = {
    "name": "forecast_report",
    "title": "Forecast pack",
    "activity": "Building the chart, workbook and report",
    "description": "Forecast with TimesFM 3 and produce the deliverables a person opens: a chart (PNG, SVG), a spreadsheet "
                   "(XLSX), a report (DOCX) and a PDF, listed in a sha256 manifest. It takes ONE series, or a panel of up to 50 "
                   "named series (source pointing at a multi-series file, such as one saved with pl.save_panel) in one call: the "
                   "panel gets one chart grid (12 series per figure), a workbook with a summary row per series, a report and a PDF. "
                   "Use it for the final forecast the person asked for, after any checking is done. It takes the same control "
                   "arguments as timesfm_forecast for one series; a panel uses the defaults. Pass exactly the configuration that won "
                   "your backtest, or the pack will show the default forecast. n_origins belongs to backtest_series and is refused "
                   "here; season is used only by the seasonal centre rules and the blend's rule.",
    "args": {"values": VALUES_HELP, "source": SOURCE_HELP, "horizon": "integer, 1 to 512", "title": "short title for the chart and report",
             "goal": "optional, the forecast asked for in words; with centre 'combination' it names the domain and place, "
                     "which choose the market and structural candidates",
             "centre": "optional 'combination' (every candidate - TimesFM, the simple rules and the domain's market and structural models - scored on the same rolling origins, and the line and band drawn from their backtest-weighted combination; pass goal), 'timesfm' (default) or the simple rule backtest_series named as best_baseline ('naive', 'drift', 'seasonal_naive', 'mean_last_season'): when the backtest says the rule was more accurate, chart it as the centre line with TimesFM's band around it. 'blend' is an equal-weight average of TimesFM's centre and the best of those rules; this tool backtests it on the same windows and draws it only when it beats both, otherwise the centre stays TimesFM's and blend_check says why (one series only, not a panel). For a panel, one rule for every series, or a dict {series name: rule} to set it per series (others keep timesfm); 'global_gbm' or 'global_blend' (panel only, the whole panel) sets the pooled gradient-boosted model, or its blend with TimesFM, as the centre ('blend' on a panel means global_blend)",
             "season": "optional integer, only for the seasonal centre rules: the season backtest_series used (default: chosen from the data by the same rule as the backtest)",
             "calibrate": "optional true, one series only (refused for a panel): replace the 10-90 percent band by the calibrated one when a held-out test on past windows shows it covers closer to 80 percent (the point forecast never changes)",
             **CONTROL_HELP},
}
