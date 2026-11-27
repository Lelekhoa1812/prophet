"""driver_map and scenario_forecast: the fundamentals layer a forecast is held to before it is delivered.

Motivation vs Logic
Motivation (owner review, 2026-11-27): "the predictions on most missions haven't accounted for other fundamental factors,
such as political and geopolitical, economic and other factors that could drive the final decision". The planner searched
for the series and, at most, one covariate; nothing made it map the drivers, measure them, compare with official and market
forecasts, or carry an event risk into the delivered range.
Logic:
- ``driver_map`` action ``plan`` returns the driver families for the target's domain (research.driver_families): what each
  family is, the data that measures it and search queries, the benchmarks to compare with, and the minimum coverage.
  Action ``record`` takes the findings (family, signal, kind researched / tested / benchmark, direction, effect, evidence),
  the benchmarks (source, value, evidence) and the scenarios (name, probability, impact), and writes driver_map.csv, .json,
  .png and .svg. Evidence is a URL, a dataset or file name, or a run id that a call in this mission used or returned; the
  finish check (runtime.prediction_quality.check_fundamentals) traces each one to that call and drops what it cannot trace.
- ``scenario_forecast`` takes the run id of a forecast_report pack and up to six scenarios (probability, shift in units or
  per cent, start, ramp, end), mixes them into the pack's distribution (forecast.scenarios) and writes the scenario chart,
  table and JSON. The mixed quantiles are the delivered range when this is the newest prediction.
"""
from __future__ import annotations

import json
import re
from typing import Any

from prophet_agent.research import driver_families as dfam
from prophet_agent.runtime.runner import Tool
from prophet_agent.sandbox.runner import SANDBOX_ROOT

KINDS = ("researched", "tested", "benchmark")
DIRECTIONS = ("up", "down", "mixed", "none")
MAX_FINDINGS = 60
RUN_ID = re.compile(r"^[0-9a-f]{32}$")


def _text(v: Any, n: int) -> str:
    return re.sub(r"\s+", " ", str(v if v is not None else "")).strip()[:n]


def _findings(raw: Any, domain: str) -> list[dict]:
    if raw is None:
        return []
    if not isinstance(raw, list) or len(raw) > MAX_FINDINGS:
        raise ValueError(f"findings must be a list of at most {MAX_FINDINGS} objects")
    known = set(dfam.family_keys(domain))
    out = []
    for i, f in enumerate(raw, start=1):
        if not isinstance(f, dict):
            raise ValueError(f"finding {i} must be an object {{family, signal, kind, direction, effect, evidence}}")
        fam = _text(f.get("family"), 40).lower().replace(" ", "_")
        if fam not in known and not fam.startswith("other:"):
            raise ValueError(f"finding {i}: family {fam!r} is not one of {sorted(known)}; name another as 'other:<name>'")
        kind = _text(f.get("kind"), 12).lower()
        if kind not in KINDS:
            raise ValueError(f"finding {i}: kind must be one of {list(KINDS)}: researched (read in a source), tested "
                             "(measured against the target: a covariate backtest, a lead-lag or regression test), or "
                             "benchmark (an official or market forecast of the target)")
        direction = _text(f.get("direction"), 8).lower() or "none"
        if direction not in DIRECTIONS:
            raise ValueError(f"finding {i}: direction must be one of {list(DIRECTIONS)} (its push on the target)")
        evidence = _text(f.get("evidence"), 300)
        if len(evidence) < 6:
            raise ValueError(f"finding {i}: evidence must name where it came from: the URL, dataset or file name, or the "
                             "run id of the call that found or tested it")
        signal = _text(f.get("signal"), 120)
        if not signal:
            raise ValueError(f"finding {i}: signal must say what was found (for example 'trimmed mean CPI 3.1% y/y, Q3')")
        out.append({"family": fam, "signal": signal, "kind": kind, "direction": direction,
                    "effect": _text(f.get("effect"), 300), "evidence": evidence})
    return out


def _benchmarks(raw: Any) -> list[dict]:
    if raw is None:
        return []
    if not isinstance(raw, list) or len(raw) > 12:
        raise ValueError("benchmarks must be a list of at most 12 objects {source, value, what, evidence}")
    out = []
    for i, b in enumerate(raw, start=1):
        if not isinstance(b, dict):
            raise ValueError(f"benchmark {i} must be an object")
        value = b.get("value")
        if isinstance(value, bool) or not isinstance(value, (int, float, str)) or _text(value, 60) == "":
            raise ValueError(f"benchmark {i}: value must be the forecast it states (a number, or a short range '3.85-4.10')")
        evidence = _text(b.get("evidence"), 300)
        if len(evidence) < 6:
            raise ValueError(f"benchmark {i}: evidence must be the URL or file it was read from")
        out.append({"source": _text(b.get("source"), 80) or "unnamed", "value": _text(value, 60),
                    "what": _text(b.get("what"), 120), "evidence": evidence})
    return out


def _probability_source(s: dict, label: str) -> str:
    """'market' or 'judgement' (the default). Owner rule (2026-11-27): a scenario's probability comes from market pricing
    where a market prices the event, and from the agent's researched judgement only where none does; a market probability
    needs the URL it was read from."""
    src = str(s.get("probability_source") or "judgement").strip().lower()
    if src not in ("market", "judgement"):
        raise ValueError(f"{label}: probability_source must be 'market' or 'judgement'")
    if src == "market" and len(str(s.get("evidence") or "").strip()) < 6:
        raise ValueError(f"{label}: a market probability needs the URL of the pricing it was read from as evidence")
    return src


def _scenario_notes(raw: Any) -> list[dict]:
    if raw is None:
        return []
    if not isinstance(raw, list) or len(raw) > 6:
        raise ValueError("scenarios must be a list of at most 6 objects {name, probability, impact}")
    out = []
    for i, s in enumerate(raw, start=1):
        if not isinstance(s, dict):
            raise ValueError(f"scenario {i} must be an object")
        p = s.get("probability")
        if not isinstance(p, (int, float)) or isinstance(p, bool) or not 0 < p < 1:
            raise ValueError(f"scenario {i}: probability must be between 0 and 1")
        name, impact = _text(s.get("name"), 60), _text(s.get("impact"), 200)
        if not name or not impact:
            raise ValueError(f"scenario {i}: give a name and the impact on the target (for example '+0.25 pp from May')")
        src = _probability_source(s, f"scenario {i}")
        out.append({"name": name, "probability": float(p), "impact": impact, "evidence": _text(s.get("evidence"), 300),
                    "probability_source": src})
    if sum(s["probability"] for s in out) > 1 + 1e-9:
        raise ValueError("scenario probabilities must add to at most 1")
    return out


def coverage(domain: str, findings: list[dict], benchmarks: list[dict], scenarios: list[dict]) -> dict:
    """Counts against the domain's minimum, before the evidence is traced."""
    minimum = dfam.DOMAINS.get(domain, dfam.DOMAINS["general"])["minimum"]
    researched = sorted({f["family"] for f in findings})
    tested = sorted({f["family"] for f in findings if f["kind"] == "tested"})
    bench = len(benchmarks) + sum(1 for f in findings if f["kind"] == "benchmark")
    missing = []
    if len(researched) < minimum["researched"]:
        missing.append(f"{minimum['researched'] - len(researched)} more driver families researched")
    if len(tested) < minimum["tested"]:
        missing.append(f"{minimum['tested'] - len(tested)} more families tested against the target")
    if bench < minimum["benchmarks"]:
        missing.append("an official or market benchmark forecast with its value")
    if len(scenarios) < minimum["scenarios"]:
        missing.append(f"{minimum['scenarios'] - len(scenarios)} more scenarios with probability and impact")
    untouched = [k for k in dfam.family_keys(domain) if k not in researched]
    return {"families_researched": researched, "families_tested": tested, "benchmarks": bench,
            "scenarios": len(scenarios), "minimum": minimum, "missing": missing, "families_not_covered": untouched}


def _driver_map(args: dict, _key: str) -> dict[str, Any]:
    extra = sorted(set(args) - {"action", "target", "domain", "findings", "benchmarks", "scenarios", "title"})
    if extra:
        raise ValueError(f"unknown arguments {extra}; allowed: action, target, domain, findings, benchmarks, scenarios, title")
    action = args.get("action", "plan")
    target = _text(args.get("target"), 300)
    domain = args.get("domain")
    if domain is not None and domain not in dfam.DOMAINS:
        raise ValueError(f"domain must be one of {sorted(dfam.DOMAINS)}, or left out to read it from the target")
    if action == "plan":
        if not target:
            raise ValueError("plan needs target: the forecast asked for, in words (for example 'RBA cash rate, next 5 years')")
        p = dfam.plan(target, domain)
        return {**p, "next": ("Research the families broadly: pass `lanes` from this result to the plan action lanes as they "
                              "are (one research lane per two families and one for the benchmarks and scenario odds; edit "
                              "them if the goal needs it), fetch the "
                              "data each one names, and test the measurable ones against the target (control_lab "
                              "with_covariates, backtest_series with covariates, or a lead-lag or regression cell). Read the "
                              "benchmarks and note their values. State two or more scenarios for the political, policy and "
                              "geopolitical risks, apply them with scenario_forecast, then record everything with "
                              "driver_map action record. Every finding names its evidence: the URL, file or run id.")}
    if action != "record":
        raise ValueError("action must be plan (families, data and queries for the target) or record (the findings)")
    dom = domain or dfam.detect(target)
    findings = _findings(args.get("findings"), dom)
    benchmarks = _benchmarks(args.get("benchmarks"))
    scenarios = _scenario_notes(args.get("scenarios"))
    if not findings:
        raise ValueError("record needs findings: at least one {family, signal, kind, direction, effect, evidence}")
    cov = coverage(dom, findings, benchmarks, scenarios)
    from prophet_agent.artifacts.fundamentals import build_driver_map
    title = _text(args.get("title"), 110) or f"What drives {target[:70] or 'the forecast'}"
    payload = {"title": title, "families": dfam.family_keys(dom) + sorted({f["family"] for f in findings
                                                                         if f["family"].startswith("other:")}),
               "findings": findings, "benchmarks": benchmarks, "scenarios": scenarios}
    pack = build_driver_map(payload)
    if not pack.ok:
        raise ValueError("the driver map could not be built: " + str(pack.stderr)[-200:])
    return {"domain": dom, "coverage": cov, "findings": findings, "benchmarks": benchmarks, "scenarios": scenarios,
            "run_id": pack.run_id, "manifest": f"{pack.workdir}/manifest.json",
            "files": [{"path": o.path, "sha256": o.sha256, "bytes": o.bytes} for o in pack.outputs],
            "note": "the finish check traces each evidence to a call in this mission and drops what it cannot trace"}


def _observe_map(result: dict) -> dict:
    if "coverage" not in result:
        return result
    return {k: v for k, v in result.items() if k not in ("findings",)} | {
        "findings": [f"{f['family']}: {f['signal']} ({f['kind']}, {f['direction']})" for f in result["findings"]],
        "files": [f["path"] for f in result.get("files", [])]}


def _pack_input(run_id: Any) -> dict:
    if not isinstance(run_id, str) or not RUN_ID.match(run_id):
        raise ValueError("pack must be the 32-character run_id of a forecast_report pack (one series)")
    path = SANDBOX_ROOT / run_id / "data" / "input.json"
    if not path.is_file():
        raise ValueError(f"run {run_id} has no forecast pack input; give the run_id forecast_report returned")
    d = json.loads(path.read_text(encoding="utf-8"))
    if not all(isinstance(d.get(k), list) and d.get(k) for k in ("point", "q10", "q90", "history")):
        raise ValueError(f"run {run_id} is not a single-series forecast pack (a panel pack is not supported); give the "
                         "run_id of a one-series forecast_report")
    return d


def _scenario_forecast(args: dict, _key: str) -> dict[str, Any]:
    from prophet_agent.artifacts.fundamentals import build_scenario_chart
    from prophet_agent.forecast import scenarios as sc
    extra = sorted(set(args) - {"pack", "scenarios", "floor", "title", "unit", "seed"})
    if extra:
        raise ValueError(f"unknown arguments {extra}; allowed: pack, scenarios, floor, title, unit, seed")
    d = _pack_input(args.get("pack"))
    h = len(d["point"])
    dates = d.get("future_dates") if isinstance(d.get("future_dates"), list) and len(d["future_dates"]) == h else None
    scen = sc.validate(args.get("scenarios"), h, dates)
    knots = {0.1: d["q10"], 0.5: d["point"], 0.9: d["q90"]}
    if isinstance(d.get("q30"), list) and len(d["q30"]) == h and isinstance(d.get("q70"), list) and len(d["q70"]) == h:
        knots.update({0.3: d["q30"], 0.7: d["q70"]})
    floor = args.get("floor")
    if floor is not None and (not isinstance(floor, (int, float)) or isinstance(floor, bool)):
        raise ValueError("floor must be a number (for example 0 for a rate or a price), or left out")
    seed = args.get("seed", 0)
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise ValueError("seed must be a whole number")
    mixed = sc.mix(knots, scen, seed=seed, floor=None if floor is None else float(floor))
    title = _text(args.get("title"), 110) or (_text(d.get("title"), 80) + ": with scenarios")
    note = ("Scenario-weighted forecast: the base pack's distribution with each scenario's shift applied from its start, "
            "weighted by its probability (base case " + f"{100 * mixed['base_case_probability']:.0f}%). "
            + "; ".join(f"{s['name']} {100 * s['probability']:.0f}% ({s['probability_source']}): {s['last_shift']:+.3g} by the end"
                        for s in mixed["scenarios"]))[:600]
    payload = {"history": d["history"], "dates": d.get("dates"), "future_dates": dates, "point": mixed["point"],
               "q10": mixed["quantiles"]["0.1"], "q90": mixed["quantiles"]["0.9"], "quantiles": mixed["quantiles"],
               "base_point": d["point"], "base_q10": d["q10"], "base_q90": d["q90"], "scenarios": mixed["scenarios"],
               "title": title, "note": note, "unit": _text(args.get("unit") or d.get("unit"), 40)}
    pack = build_scenario_chart(payload)
    if not pack.ok:
        raise ValueError("the scenario chart could not be built: " + str(pack.stderr)[-200:])
    fc = {"point": mixed["point"], "quantiles": mixed["quantiles"], **({"future_dates": dates} if dates else {})}
    rows = [{"step": i + 1, **({"date": dates[i]} if dates else {}), "median": mixed["point"][i],
             "q10": mixed["quantiles"]["0.1"][i], "q90": mixed["quantiles"]["0.9"][i]} for i in range(h)]
    return {"forecast": fc, "pack": args["pack"], "scenarios": [{k: s[k] for k in ("name", "probability", "probability_source",
                                                                                   "start", "last_shift", "why")}
                                                                for s in mixed["scenarios"]],
            "base_case_probability": mixed["base_case_probability"], "weighted_last_shift": mixed["weighted_last_shift"],
            "end": {"q10": rows[-1]["q10"], "median": rows[-1]["median"], "q90": rows[-1]["q90"],
                    **({"date": dates[-1]} if dates else {})},
            "base_end": {"q10": d["q10"][-1], "point": d["point"][-1], "q90": d["q90"][-1]},
            "table": rows if h <= 60 else rows[:5] + rows[-5:],
            "run_id": pack.run_id, "manifest": f"{pack.workdir}/manifest.json",
            "files": [{"path": o.path, "sha256": o.sha256, "bytes": o.bytes} for o in pack.outputs],
            "deliverables": "scenario_forecast.png and .svg (chart), .csv and .json (table); this range is now the "
                            "delivered one: quote end.q10 and end.q90 for the range at the end of the horizon"}


def _observe_scenario(result: dict) -> dict:
    if "forecast" not in result:
        return result
    out = {k: v for k, v in result.items() if k != "forecast"}
    out["files"] = [f["path"] for f in result.get("files", [])]
    return out


def driver_map_tool() -> Tool:
    return Tool("driver_map", _driver_map, idempotent=True, observe=_observe_map)


def scenario_forecast_tool() -> Tool:
    return Tool("scenario_forecast", _scenario_forecast, idempotent=True, observe=_observe_scenario)


DRIVER_MAP_SPEC = {
    "name": "driver_map",
    "title": "Driver map",
    "activity": "Mapping what drives the forecast",
    "description": "The fundamentals a forecast must account for. action plan, with target (the forecast in words): the "
                   "driver families for its domain (policy rate, labour market, precious metal, equity index, currency, "
                   "commodity, sports league, or general), each with what it is, the data that measures it and search "
                   "queries for the news and official commentary that move it (political, geopolitical, economic), plus "
                   "the benchmarks to compare with (official projections, market-implied paths, consensus) and the "
                   "minimum coverage the finish check holds the mission to. action record writes the driver map "
                   "(driver_map.csv, .json, .png, .svg): findings, benchmarks and scenarios, each with its evidence (the "
                   "URL, file or run id of the call that found or tested it). Record again as findings arrive; the "
                   "newest entry per family and signal counts.",
    "args": {
        "action": "plan (default) or record",
        "target": "the forecast asked for, in words (for example 'RBA cash rate, next 5 years')",
        "domain": "optional domain name, to override the one read from the target",
        "findings": "record: list of {family (a key from plan, or 'other:<name>'), signal (what was found, with its figure "
                    "and date), kind ('researched' read in a source, 'tested' measured against the target by a covariate "
                    "backtest, lead-lag or regression, 'benchmark'), direction ('up', 'down', 'mixed', 'none': its push "
                    "on the target), effect (size and timing in words or numbers), evidence (URL, file name or run id)}",
        "benchmarks": "record: list of {source (for example 'RBA SMP November 2026'), value (its forecast of the target), "
                      "what (measure and date), evidence (URL or file)}",
        "scenarios": "record: list of {name, probability (0 to 1), impact (on the target, in words and numbers), evidence, "
                     "probability_source ('market' when the probability is read from market pricing such as a rate tracker, futures or outright odds, with that URL as evidence; 'judgement' when no market prices it)}",
        "title": "optional chart title",
    },
}

SCENARIO_FORECAST_SPEC = {
    "name": "scenario_forecast",
    "title": "Scenario forecast",
    "activity": "Weighing the scenarios into the forecast",
    "description": "Apply political, policy, geopolitical or economic scenarios to a delivered forecast_report pack and "
                   "mix them by probability into one distribution. Each scenario shifts the base forecast from its start "
                   "(a step number or a date such as '2027-03'), reached over ramp steps, held or faded after end, with "
                   "its own uncertainty; the rest of the probability is the base case. Writes scenario_forecast.png and "
                   ".svg (history, base pack, each scenario's path, the weighted median and 10 to 90 range), .csv and "
                   ".json. The weighted range becomes the delivered range: quote its end.q10 and end.q90. Use it for "
                   "event risks the history cannot show (a signalled rate move, tariffs, an election, a war, a budget).",
    "args": {
        "pack": "run_id of the forecast_report pack to build on (one series)",
        "scenarios": "list of 1 to 6 {name, probability (0 to 1; together at most 1), shift (in the series' units, "
                     "for example 0.25) or shift_pct (per cent of the base median, for example -8), start (step or date, "
                     "default 1), ramp (steps to reach the full shift, default 1), end (optional step after which it "
                     "fades), shift_sd (uncertainty of the shift as a share of it, default 0.3), band_scale (0.5 to 3, "
                     "default 1), why (one line), evidence (URL or run id behind it), probability_source ('market' when the probability is read from market pricing such as a rate tracker, futures or outright odds, with that URL as evidence; 'judgement' when no market prices it)}",
        "floor": "optional lower bound for the values (0 for a rate or a price)",
        "title": "optional chart title",
        "unit": "optional unit label",
        "seed": "optional random seed (default 0)",
    },
}
