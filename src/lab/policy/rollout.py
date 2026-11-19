"""Rollout guardrails for a new tenant: readiness check, shadow period, bounded learning, drift monitor.

Motivation vs Logic
Motivation: an agent that works on one business can fail quietly on another. Before alerts reach a person the platform
must show, on that client's own history, what it would have caught and how noisy it would have been; for the first
weeks alerts are recorded but hidden; learning from approvals may adjust priorities only within visible limits; and a
monthly check must catch a model whose calibration has slipped.
Logic: tenant settings live in one JSON file per dataset (economics, go-live date, learning log). `readiness` replays
one agent over the tenant's recent history under the tenant's policy and scores it against observed stockouts and
write-offs. `phase` says setup / shadow / live. `learn` turns approvals and dismissals into a per-agent score weight,
clamped and logged. `drift_report` compares forecast ranges with what happened, per volume class.
"""
from __future__ import annotations

import json
from datetime import date, datetime, timezone

import numpy as np
import pandas as pd

from ..config import DATA
from ..engine import metrics
from ..engine.state import ForecastService, get_world, weekly_matrix
from . import alerts

SHADOW_DAYS = 14
MIN_HISTORY_WEEKS = 12            # below this no agent switches on
SEASONAL_WEEKS = 52               # below this no seasonal claims are made
MIN_PRECISION = 0.30              # below this an agent stays in shadow
MIN_DECISIONS_TO_LEARN = 20
TARGET_PRECISION = 0.50
MAX_STEP = 0.10                   # one learning step moves a weight by at most 10%


def _path(ds: str):
    return DATA / f"tenant_{ds}.json"


def load(ds: str) -> dict:
    p = _path(ds)
    cfg = json.loads(p.read_text()) if p.exists() else {}
    cfg.setdefault("economics", alerts.Economics().describe())
    cfg.setdefault("go_live", None)
    cfg.setdefault("shadow_days", SHADOW_DAYS)
    cfg.setdefault("learning_log", [])
    cfg.setdefault("retune_log", [])
    return cfg


def save(ds: str, cfg: dict) -> None:
    _path(ds).write_text(json.dumps(cfg, indent=1, default=str))


def economics_of(cfg: dict) -> alerts.Economics:
    e = cfg["economics"]
    return alerts.from_dict({k: e[k] for k in ("review_cost", "effectiveness", "budget_per_owner", "false_alarm_cost", "agent_weight") if k in e})


def phase(cfg: dict, as_of: date | str) -> dict:
    """demo (no go-live date), shadow (first days after go-live: alerts recorded, hidden) or live."""
    gl = cfg.get("go_live")
    if not gl:
        return {"phase": "demo", "label": "Demo: no go-live date set", "shadow_ends": None}
    d = pd.Timestamp(as_of)
    start = pd.Timestamp(gl)
    end = start + pd.Timedelta(days=int(cfg.get("shadow_days", SHADOW_DAYS)))
    if d < start:
        return {"phase": "setup", "label": "Set-up: not live yet", "shadow_ends": str(end.date())}
    if d < end:
        return {"phase": "shadow", "label": f"Shadow mode until {end.date()}: alerts are recorded and scored but not shown", "shadow_ends": str(end.date())}
    return {"phase": "live", "label": "Live", "shadow_ends": str(end.date())}


def learn(ds: str, rows: list[tuple], cfg: dict | None = None) -> list[dict]:
    """rows: (agent, decision, reason) from approvals and dismissals. Returns the weight changes made (also logged).

    Only 'wrong alert' and 'not useful' dismissals count against an agent; 'one-off' and 'known promo' are the
    operator's context, not a bad alert. Weights move at most MAX_STEP per call and stay inside the tenant's bounds.
    """
    cfg = cfg or load(ds)
    econ = economics_of(cfg)
    by: dict[str, list[int]] = {}
    for agent, decision, reason in rows:
        c = by.setdefault(agent, [0, 0])                     # [good, bad]
        if decision in ("approve", "adjust"):
            c[0] += 1
        elif decision == "dismiss" and reason in ("wrong_alert", "not_useful"):
            c[1] += 1
    changes = []
    for agent, (good, bad) in by.items():
        n = good + bad
        if n < MIN_DECISIONS_TO_LEARN:
            continue
        prec = good / n
        old = econ.agent_weight.get(agent, 1.0)
        step = float(np.clip(prec / TARGET_PRECISION, 1 - MAX_STEP, 1 + MAX_STEP))
        new = alerts.bounded_weight(econ, agent, old * step)
        if abs(new - old) > 1e-9:
            changes.append({"at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "agent": agent, "old": round(old, 3),
                            "new": round(new, 3), "basis": {"useful": good, "not_useful": bad, "precision": round(prec, 3), "target": TARGET_PRECISION}})
    if changes:
        cfg["economics"] = econ.describe()
        cfg["learning_log"] = (cfg["learning_log"] + changes)[-200:]
        save(ds, cfg)
    return changes


def readiness(ds: str, agent: str, weeks: int = 13, econ: alerts.Economics | None = None) -> dict:
    """Replay one agent over the tenant's recent history under its policy and say whether it is safe to switch on."""
    from ..sim.replay import replay, scored_window
    w = get_world(ds)
    history_weeks = int(len(w.days) // 7)
    econ = econ or economics_of(load(ds))
    lo, hi = scored_window(w)
    start = max(lo, hi - pd.Timedelta(days=7 * weeks))
    res = replay(ds, start=str(start.date()), end=str(hi.date()), agents=[agent], policy=econ.describe(), persist=False)
    a = (res.get("agents") or {}).get(agent) or {}
    pol = res.get("policy", {})
    reasons, notes = [], []
    verdict = "ready"
    if history_weeks < MIN_HISTORY_WEEKS:
        verdict = "not_ready"; reasons.append(f"Only {history_weeks} weeks of history; at least {MIN_HISTORY_WEEKS} are needed.")
    if history_weeks < SEASONAL_WEEKS:
        notes.append(f"{history_weeks} weeks is under a year: no seasonal claims (Christmas, Easter) are made.")
    if a.get("proposals", 0) == 0:
        notes.append("No alerts in the window, so precision and recall could not be measured.")
        if verdict == "ready":
            verdict = "shadow_first"
    else:
        if (a.get("precision") or 0) < MIN_PRECISION:
            verdict = "shadow_first" if verdict == "ready" else verdict
            reasons.append(f"Precision {100 * (a.get('precision') or 0):.0f}% is under {100 * MIN_PRECISION:.0f}%: keep in shadow until it improves.")
    return {"dataset": ds, "agent": agent, "history_weeks": history_weeks, "window": [str(start.date()), str(hi.date())],
            "verdict": verdict, "reasons": reasons, "notes": notes,
            "alerts_shown_per_day": pol.get("shown_per_day"), "digest_per_day": pol.get("digest_per_day"),
            "precision": a.get("precision"), "recall": a.get("recall"), "events": a.get("events"), "caught": a.get("caught"),
            "dollars_truth_caught": a.get("dollars_truth_caught"), "dollars_truth_total": a.get("dollars_truth_total"),
            "policy": econ.describe(),
            "caveat": "Scored against stockouts and write-offs recorded in the data. Flagged is not the same as prevented."}


def drift_report(ds: str, as_of: str | None = None, origins: int = 8, horizon: int = 4) -> dict:
    """Coverage, bias and pinball of the forecast range per volume class over the latest completed origins."""
    w = get_world(ds)
    fs = ForecastService(w)
    end = pd.Timestamp(as_of) if as_of else w.days[-1]
    mat, weeks = weekly_matrix(w, end)
    n = mat.shape[1]
    vol = mat[:, -52:].sum(axis=1)
    order = np.argsort(-vol)
    cls = np.array(["small"] * len(vol), dtype=object)
    active_rank = [i for i in order if vol[i] > 0]
    k = len(active_rank)
    for r, i in enumerate(active_rank):
        cls[i] = "big" if r < 0.2 * k else "mid" if r < 0.6 * k else "small"
    acc: dict[str, dict] = {c: {"a": [], "q": []} for c in ("big", "mid", "small")}
    used = []
    for o in range(n - horizon - origins + 1, n - horizon + 1):
        if o < 8:
            continue
        res = fs.weekly(weeks[o], horizon)
        if res["fc"] is None:
            continue
        used.append(str(weeks[o].date()))
        for kk, pi in enumerate(res["active"]):
            acc[cls[pi]]["a"].append(mat[pi, o:o + horizon]); acc[cls[pi]]["q"].append(res["fc"].q[kk])
    table = {}
    for c, d in acc.items():
        if not d["a"]:
            continue
        a = np.stack(d["a"]); q = np.stack(d["q"])
        cov = metrics.coverage(a, q); b = metrics.bias(a, q[..., 4]); pb = metrics.pinball(a, q)
        # Bugs vs Fixes. Bug: flagging a median that sits below total units called small sellers drifted, but lumpy
        # demand always has mean above median, so a unit-bias test fires on a healthy model. Fix: test calibration
        # (share of actuals at or below the median, and 10-90 coverage); keep unit bias as information only.
        below = float((a <= q[..., 4]).mean())
        flags = []
        if cov < 0.70 or cov > 0.90:
            flags.append(f"10-90 range covered {100 * cov:.0f}% of actuals (target 80%)")
        if below < 0.40 or below > 0.60:
            flags.append(f"{100 * below:.0f}% of actuals fell at or below the median (target 50%)")
        table[c] = {"series_origins": int(len(a)), "coverage_10_90": round(cov, 3), "share_below_median": round(below, 3),
                    "median_bias": round(b, 3), "pinball": round(pb, 3),
                    "status": "drift" if flags else "ok", "flags": flags}
    status = "drift" if any(v["status"] == "drift" for v in table.values()) else "ok"
    return {"dataset": ds, "origins": used, "horizon_weeks": horizon, "status": status, "by_volume_class": table,
            "action": "Re-tune and re-run the readiness check before trusting new alerts." if status == "drift" else "No action needed."}
