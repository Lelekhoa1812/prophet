"""Re-run alert selection offline on recorded daily candidates, so policy settings can be compared in seconds.

Motivation vs Logic
Motivation: a full replay costs tens of minutes, so sweeping review cost, effectiveness, budget and ranking is not
practical if each point needs forecasts. Candidates (everything the agents would raise before cooldown and budget) are
recorded once. Logic: for each day, apply cooldown (only for alerts actually shown), then the score-and-budget policy,
then score what was shown against what really happened.
"""
from __future__ import annotations

import pickle
from datetime import date
from types import SimpleNamespace

import pandas as pd

from ..config import DATA
from ..policy import alerts
from ..policy.learning import SegmentLearner
from ..policy.governor import Governor
from .scorer import score


def load_candidates(name: str = "cand_twin.pkl") -> list[dict]:
    return pickle.loads((DATA / name).read_bytes())


def flag_candidates(candidates: list[dict], dataset: str, window: tuple[pd.Timestamp, pd.Timestamp]) -> dict[str, bool]:
    """Whether each candidate would have been useful (matches a recorded problem), independent of what gets shown."""
    res = score(dataset, [{k: c[k] for k in ("agent", "entity_type", "entity_id", "as_of", "dollars_at_stake", "severity", "dedupe_key", "action", "proposal_id")}
                          for c in candidates], window)
    out: dict[str, bool] = {}
    for d in (res.get("flags") or {}).values():
        out.update(d)
    return out


def simulate(candidates: list[dict], econ: alerts.Economics, dataset: str, window: tuple[pd.Timestamp, pd.Timestamp],
             flags: dict | None = None, learn_delay: int | None = None, segments: dict | None = None) -> dict:
    """learn_delay: days until an operator's verdict on a shown alert reaches the learner (None = no learning)."""
    learner = SegmentLearner(bounds=econ.weight_bounds) if learn_delay is not None else None
    pending: dict[str, list] = {}
    by_day: dict[str, list[dict]] = {}
    for c in candidates:
        by_day.setdefault(c["as_of"], []).append(c)
    gov = Governor()
    shown_rows: list[dict] = []
    digest_n = 0
    days = sorted(by_day)
    for d in days:
        if learner is not None:
            for item in pending.pop(d, []):
                learner.observe(item[0], item[1], flags[item[2]])
            econ.segment_weight = learner.weights()
        objs = [SimpleNamespace(**c) for c in by_day[d]]
        if segments:
            for o in objs:
                if getattr(o, "segment", None) is None:
                    o.segment = segments.get(o.entity_id)
        dt = date.fromisoformat(d)
        kept = gov.filter(objs, dt, commit=False)
        shown, digest, _ = alerts.select(kept, econ)
        gov.commit(shown, dt)
        if learner is not None:
            due = str((pd.Timestamp(d) + pd.Timedelta(days=learn_delay)).date())
            pending.setdefault(due, []).extend((o.agent, getattr(o, "segment", None), o.proposal_id) for o in shown)
        digest_n += len(digest)
        shown_rows += [{k: v for k, v in vars(o).items() if k in ("agent", "entity_type", "entity_id", "as_of", "dollars_at_stake", "severity", "dedupe_key", "action", "proposal_id")} for o in shown]
    n_days = max((window[1] - window[0]).days + 1, 1)
    res = score(dataset, shown_rows, window)
    out = {"policy": econ.describe(), "shown": len(shown_rows), "shown_per_day": round(len(shown_rows) / n_days, 2),
           "digest_per_day": round(digest_n / n_days, 2), "agents": {}}
    for k, a in (res.get("agents") or {}).items():
        out["agents"][k] = {"alerts": a["proposals"], "precision": a["precision"], "recall": a["recall"], "events": a["events"], "caught": a["caught"],
                            "dollar_recall": (a["dollars_truth_caught"] / a["dollars_truth_total"]) if a.get("dollars_truth_total") else None,
                            "stats": {k: a.get(k) for k in ("proposals", "useful", "dollars_truth_caught", "dollars_truth_total", "dollars_at_stake_total")}}
    return out
