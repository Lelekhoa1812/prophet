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
from ..policy.governor import Governor
from .scorer import score


def load_candidates(name: str = "cand_twin.pkl") -> list[dict]:
    return pickle.loads((DATA / name).read_bytes())


def simulate(candidates: list[dict], econ: alerts.Economics, dataset: str, window: tuple[pd.Timestamp, pd.Timestamp]) -> dict:
    by_day: dict[str, list[dict]] = {}
    for c in candidates:
        by_day.setdefault(c["as_of"], []).append(c)
    gov = Governor()
    shown_rows: list[dict] = []
    digest_n = 0
    days = sorted(by_day)
    for d in days:
        objs = [SimpleNamespace(**c) for c in by_day[d]]
        dt = date.fromisoformat(d)
        kept = gov.filter(objs, dt, commit=False)
        shown, digest, _ = alerts.select(kept, econ)
        gov.commit(shown, dt)
        digest_n += len(digest)
        shown_rows += [{k: v for k, v in vars(o).items() if k in ("agent", "entity_type", "entity_id", "as_of", "dollars_at_stake", "severity", "dedupe_key", "action", "proposal_id")} for o in shown]
    n_days = max((window[1] - window[0]).days + 1, 1)
    res = score(dataset, shown_rows, window)
    out = {"policy": econ.describe(), "shown": len(shown_rows), "shown_per_day": round(len(shown_rows) / n_days, 2),
           "digest_per_day": round(digest_n / n_days, 2), "agents": {}}
    for k, a in (res.get("agents") or {}).items():
        out["agents"][k] = {"alerts": a["proposals"], "precision": a["precision"], "recall": a["recall"], "events": a["events"], "caught": a["caught"],
                            "dollar_recall": (a["dollars_truth_caught"] / a["dollars_truth_total"]) if a.get("dollars_truth_total") else None}
    return out
