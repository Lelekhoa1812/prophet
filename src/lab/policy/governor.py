"""Governance between an agent and a person: dedupe, cooldown, lessons, autonomy ladder.

Rules (Liam, 19 Nov): dedupe and cooldown; learn from dismissals (visible and undoable); earned autonomy ladder.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta

LADDER = ["explains", "suggests", "prepares", "acts", "runs"]
LADDER_LABEL = {"explains": "Explains", "suggests": "Suggests", "prepares": "Prepares for your OK",
                "acts": "Acts within limits", "runs": "Runs the workflow"}

DEFAULT_COOLDOWN_DAYS = {
    "shortfall": 3, "freshness": 2, "customer-slowdown": 14, "customer-lapse": 14, "keying": 0,
    "margin-cost": 14, "price": 5,
}

# Promotion needs evidence; demotion follows a miss. Used by the console's autonomy view.
PROMOTION = {"suggests->prepares": {"min_proposals": 30, "min_precision": 0.70, "max_severe_false": 0.05},
             "prepares->acts": {"min_proposals": 100, "min_precision": 0.85, "max_severe_false": 0.02},
             "demote_below_precision": 0.50, "demote_window": 20}


@dataclass
class Lesson:
    lesson_id: str
    agent: str
    scope_type: str            # entity | agent
    scope_id: str              # dedupe prefix or entity id
    effect: dict               # {"suppress_days": n} or {"min_dollars_mult": x}
    created_on: date
    active: bool = True


@dataclass
class Governor:
    cooldown_days: dict = field(default_factory=lambda: dict(DEFAULT_COOLDOWN_DAYS))
    lessons: list = field(default_factory=list)
    last_raised: dict = field(default_factory=dict)       # dedupe_key -> (date, dollars)
    suppressed: list = field(default_factory=list)        # records of what governance held back, for the console

    def _cooldown(self, dedupe_key: str) -> int:
        prefix = dedupe_key.split("|")[0]
        return self.cooldown_days.get(prefix, 3)

    def _lesson_for(self, p, as_of: date):
        for ls in self.lessons:
            if not ls.active or ls.agent != p.agent:
                continue
            if ls.scope_id in (p.entity_id, p.dedupe_key):
                if "suppress_days" in ls.effect and (as_of - ls.created_on).days <= ls.effect["suppress_days"]:
                    return ls
                if "min_dollars_mult" in ls.effect:
                    return ls
        return None

    def commit(self, proposals: list, as_of: date) -> None:
        """Record proposals as raised, so cooldown applies from now on. Used when a budget decides what is really shown."""
        for p in proposals:
            self.last_raised[p.dedupe_key] = (as_of, p.dollars_at_stake)

    def filter(self, proposals: list, as_of: date, commit: bool = True) -> list:
        kept = []
        for p in proposals:
            ls = self._lesson_for(p, as_of)
            if ls:
                if "suppress_days" in ls.effect:
                    self.suppressed.append({"proposal_id": p.proposal_id, "reason": "lesson", "lesson_id": ls.lesson_id, "as_of": str(as_of), "key": p.dedupe_key})
                    continue
                if "min_dollars_mult" in ls.effect and p.dollars_at_stake < ls.effect["min_dollars_mult"] * p.dollars_at_stake * 0 + ls.effect.get("min_dollars", 0):
                    self.suppressed.append({"proposal_id": p.proposal_id, "reason": "lesson_threshold", "lesson_id": ls.lesson_id, "as_of": str(as_of), "key": p.dedupe_key})
                    continue
            last = self.last_raised.get(p.dedupe_key)
            cd = self._cooldown(p.dedupe_key)
            if last and cd > 0 and (as_of - last[0]).days < cd:
                # repeat only if materially worse (dollars up 50%+)
                if p.dollars_at_stake < last[1] * 1.5:
                    self.suppressed.append({"proposal_id": p.proposal_id, "reason": "cooldown", "as_of": str(as_of), "key": p.dedupe_key})
                    continue
            if commit:
                self.last_raised[p.dedupe_key] = (as_of, p.dollars_at_stake)
            kept.append(p)
        return kept


def ladder_status(agent: str, current: str, stats: dict) -> dict:
    """Where an agent stands on the autonomy ladder, with what it still needs to be promoted."""
    i = LADDER.index(current)
    nxt = LADDER[i + 1] if i + 1 < len(LADDER) else None
    out = {"agent": agent, "current": current, "label": LADDER_LABEL[current], "next": nxt,
           "next_label": LADDER_LABEL.get(nxt) if nxt else None, "stats": stats, "criteria": [], "eligible": False, "demote": False}
    key = f"{current}->{nxt}" if nxt else None
    n, prec, sev = stats.get("proposals", 0), stats.get("precision"), stats.get("severe_false_rate", 0.0)
    if key in PROMOTION:
        c = PROMOTION[key]
        checks = [("Proposals reviewed", n, c["min_proposals"], n >= c["min_proposals"]),
                  ("Useful (precision)", prec, c["min_precision"], prec is not None and prec >= c["min_precision"]),
                  ("High-severity false alarms", sev, c["max_severe_false"], sev is not None and sev <= c["max_severe_false"])]
        out["criteria"] = [{"label": a, "value": b, "target": t, "met": m, "direction": "max" if "false" in a else "min"} for a, b, t, m in checks]
        out["eligible"] = all(m for *_, m in checks)
    elif nxt:
        out["criteria"] = [{"label": "Needs a policy decision from AXION", "value": None, "target": None, "met": False, "direction": "min"}]
    if prec is not None and stats.get("recent_precision") is not None and stats["recent_precision"] < PROMOTION["demote_below_precision"] and i > 0:
        out["demote"] = True
    return out
