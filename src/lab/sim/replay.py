"""Time-travel replay: step the business day by day, run the agents, govern, store, score."""
from __future__ import annotations

import json
import time
import uuid
from datetime import datetime, timezone

import pandas as pd

from ..agents import REGISTRY
from ..agents.base import AgentContext
from ..engine.state import ForecastService, get_world
from ..policy import alerts as alert_policy
from ..policy.governor import Governor, Lesson
from ..store import open_lab
from .scorer import score


def scored_window(w) -> tuple[pd.Timestamp, pd.Timestamp]:
    n = len(w.days)
    return w.days[int(n * 0.55)], w.days[-1] - pd.Timedelta(days=14)


def replay(dataset: str, start: str | None = None, end: str | None = None, agents: list[str] | None = None,
           params: dict | None = None, policy: dict | None = None, learn: bool = False, persist: bool = True, progress=None, keep_rows: bool = False) -> dict:
    t0 = time.time()
    w = get_world(dataset)
    fs = ForecastService(w)
    lo, hi = scored_window(w)
    lo = pd.Timestamp(start) if start else lo
    hi = pd.Timestamp(end) if end else hi
    agents = agents or list(REGISTRY)
    params = params or {}
    gov = Governor()
    econ = alert_policy.from_dict(policy) if policy is not None else None
    if econ is not None:
        # Under a score-plus-budget policy the agents emit every candidate above the score floor and the policy decides
        # what is shown, so the old probability and share gates are opened up (customer and margin keep their own).
        params = {**params, "shortfall": {"risk_threshold": 0.0, "min_dollars": econ.floor, **params.get("shortfall", {})},
                  "freshness": {"min_dollars": econ.floor, "min_share_at_risk": 0.05, **params.get("freshness", {})}}
    digested: list = []
    days = list(pd.date_range(lo, hi, freq="D"))      # restaurants and caterers order on weekends too
    raised: list = []
    lessons_made = 0
    for k, d in enumerate(days):
        day_all: list = []
        for name in agents:
            ctx = AgentContext(w, d, fs, params.get(name, {}), gov.lessons)
            ps = REGISTRY[name].run(ctx)
            if econ is None:
                kept = gov.filter(ps, d.date())
            else:
                kept = gov.filter(ps, d.date(), commit=False)
                day_all.extend(kept)
                continue
            raised += kept
            if learn:                                  # simulated operator: dismiss false alarms with a reason
                from .scorer import score as _s     # noqa: F401  (kept local to avoid cycles)
        if econ is not None:
            shown, digest, _below = alert_policy.select(day_all, econ)
            gov.commit(shown, d.date())
            raised += shown
            digested += digest
        if progress and (k % 10 == 0 or k == len(days) - 1):
            progress(k + 1, len(days))
    rows = [dict(agent=p.agent, entity_type=p.entity_type, entity_id=p.entity_id, as_of=p.as_of, dollars_at_stake=p.dollars_at_stake,
                 severity=p.severity, dedupe_key=p.dedupe_key, action=p.action, proposal_id=p.proposal_id) for p in raised]
    result = score(dataset, rows, (lo, hi))
    if econ is not None:
        result["policy"] = {**econ.describe(), "digested": len(digested), "shown": len(raised),
                            "shown_per_day": round(len(raised) / max(len(days), 1), 2),
                            "digest_per_day": round(len(digested) / max(len(days), 1), 2)}
        result["digest_score"] = score(dataset, [dict(agent=p.agent, entity_type=p.entity_type, entity_id=p.entity_id, as_of=p.as_of,
                                                      dollars_at_stake=p.dollars_at_stake, severity=p.severity, dedupe_key=p.dedupe_key,
                                                      action=p.action, proposal_id=p.proposal_id) for p in raised + digested], (lo, hi))["agents"]
    result.update({"dataset": dataset, "days": len(days), "proposals_total": len(raised),
                   "held_back": len(gov.suppressed), "seconds": round(time.time() - t0, 1)})
    if persist:
        lab = open_lab()
        rid = f"RP-{datetime.now(timezone.utc):%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:4]}"
        lab.execute("insert into runs values (?,?,?,?,?,?,?,?,?,?,?,?)",
                    [rid, "replay", dataset, ",".join(agents), hi.date(), datetime.now(timezone.utc), datetime.now(timezone.utc),
                     "done", "dev", "timesfm-3.0", json.dumps({"start": str(lo.date()), "end": str(hi.date()), "params": params}),
                     json.dumps({k: v for k, v in result.items() if k != "events"})])
        lab.execute("delete from proposals where dataset=? and run_id like 'RP-%'", [dataset])
        for p in raised:
            # Bugs vs Fixes. Bug: a time-machine (TT-live) row with the same content-derived id made the insert fail
            # with a duplicate key, so the whole replay aborted. Fix: replace any row with the same id before inserting.
            lab.execute("delete from proposals where proposal_id=?", [p.proposal_id])
            lab.execute("""insert into proposals values (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                        [p.proposal_id, rid, dataset, p.agent, p.as_of, p.entity_type, p.entity_id, p.title, p.severity,
                         p.dollars_at_stake, p.confidence, p.autonomy_level, p.owner, json.dumps(p.evidence, default=str),
                         json.dumps(p.action, default=str), "open", p.dedupe_key, None, None, datetime.now(timezone.utc)])
        lab.close()
        result["run_id"] = rid
    result["suppressed_sample"] = gov.suppressed[:50]
    if persist:
        from ..config import DATA
        (DATA / f"scorecard_{dataset}.json").write_text(json.dumps({k: v for k, v in result.items() if k != "_rows"}, default=str))
    if keep_rows:
        result["_rows"] = rows
    return result
