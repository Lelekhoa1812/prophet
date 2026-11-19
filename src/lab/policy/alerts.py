"""One score, one budget: decide which alerts a person sees, and which go to the digest.

Motivation vs Logic
Motivation: separate hand-set gates (probability at least 55%, dollars at least $150) interact, miss the efficient
frontier, and mean nothing to a client. Each client has different margins, staffing and appetite for alerts.
Logic: every proposal carries one score, the expected dollars at stake (already probability-weighted by the agent).
A tenant states three business facts: what it costs a person to review one alert, what share of the at-risk dollars
a reviewed alert usually saves, and how many alerts one owner can handle in a morning. From those:
  net    = effectiveness x expected dollars - review_cost - (1 - chance it happens) x false_alarm_cost
  shown only if net > 0; the floor at certainty 100% is review_cost / effectiveness
  shown  = the owner's top `budget_per_owner` alerts at or above the floor, ranked by score
  digest = everything else at or above the floor, grouped, so nothing is silently lost
Order-entry stops are exempt from the budget, because they only work if seen before picking.
A learned per-agent weight (bounded, logged) can scale the score, see `bounded_weight`.
"""
from __future__ import annotations

from dataclasses import dataclass, field

EXEMPT_ACTIONS = {"hold_order_line"}


@dataclass
class Economics:
    review_cost: float = 12.5            # dollars of staff time to read and act on one alert
    effectiveness: float = 0.5           # share of expected loss a reviewed alert typically recovers
    budget_per_owner: int = 8            # alerts one owner sees per morning
    false_alarm_cost: float = 200.0      # dollars lost when an alert that did not happen is acted on (wasted order, wasted call)
    weight_bounds: tuple = (0.5, 2.0)    # learning may scale an agent's score only within these
    agent_weight: dict = field(default_factory=dict)

    @property
    def floor(self) -> float:
        return self.review_cost / max(self.effectiveness, 0.05)

    def describe(self) -> dict:
        return {"review_cost": self.review_cost, "effectiveness": self.effectiveness, "budget_per_owner": self.budget_per_owner, "false_alarm_cost": self.false_alarm_cost,
                "score_floor": round(self.floor, 2), "agent_weight": dict(self.agent_weight)}


def from_dict(d: dict | None) -> Economics:
    d = d or {}
    allowed = {k: v for k, v in d.items() if k in Economics.__dataclass_fields__}
    return Economics(**allowed)


def bounded_weight(econ: Economics, agent: str, wanted: float) -> float:
    """Clamp a learned multiplier to the tenant's bounds, and record it."""
    lo, hi = econ.weight_bounds
    w = min(max(wanted, lo), hi)
    econ.agent_weight[agent] = w
    return w


def _get(p, k, default=None):
    return getattr(p, k, default) if hasattr(p, k) and not isinstance(p, dict) else (p.get(k, default) if isinstance(p, dict) else default)


def value_of(p, econ: Economics) -> float:
    """Expected dollars at stake, scaled by the agent's bounded learned weight. This is what the floor tests."""
    return float(_get(p, "dollars_at_stake", 0.0)) * econ.agent_weight.get(_get(p, "agent"), 1.0)


def net_value(p, econ: Economics) -> float:
    """Expected net value of showing this alert: what a reviewed alert recovers, minus the review time, minus the
    expected cost of acting on a false alarm. Alerts without a modelled chance (p_event) pay no false-alarm cost."""
    gain = econ.effectiveness * value_of(p, econ)
    pe = _get(p, "p_event")
    fa = (1.0 - float(pe)) * econ.false_alarm_cost if pe is not None else 0.0
    return gain - econ.review_cost - fa


score_of = net_value        # ranking score: one number per alert


def _action_type(p) -> str:
    a = p.action if hasattr(p, "action") else p.get("action", {})
    return (a or {}).get("type", "")


def _owner(p) -> str:
    return p.owner if hasattr(p, "owner") else p["owner"]


def select(proposals: list, econ: Economics) -> tuple[list, list, list]:
    """Return (shown, digest, below_floor). Works on Proposal objects or API dicts."""
    ranked = sorted(proposals, key=lambda p: -score_of(p, econ))
    shown, digest, below, per_owner = [], [], [], {}
    for p in ranked:
        if _action_type(p) in EXEMPT_ACTIONS:
            shown.append(p)
            continue
        if net_value(p, econ) <= 0:
            below.append(p)
            continue
        o = _owner(p)
        per_owner[o] = per_owner.get(o, 0) + 1
        (shown if per_owner[o] <= econ.budget_per_owner else digest).append(p)
    return shown, digest, below


def agent_params(econ: Economics, params: dict | None = None) -> dict:
    """Open the agents' own gates so the policy decides what is shown. Shortfall and freshness emit every candidate
    that could clear the floor (the chance gate is gone: false-alarm cost prices it). Customer and margin keep their own
    rules for now; they are not yet probability-based."""
    params = dict(params or {})
    return {**params,
            "shortfall": {"risk_threshold": 0.0, "min_dollars": econ.floor, **params.get("shortfall", {})},
            "freshness": {"min_dollars": econ.floor, "min_share_at_risk": 0.05, **params.get("freshness", {})}}


def digest_groups(items: list) -> list[dict]:
    """Group digested alerts by owner and agent: count and expected dollars."""
    g: dict[tuple, dict] = {}
    for p in items:
        o, a = _owner(p), _get(p, "agent")
        row = g.setdefault((o, a), {"owner": o, "agent": a, "count": 0, "dollars": 0.0})
        row["count"] += 1
        row["dollars"] = round(row["dollars"] + float(_get(p, "dollars_at_stake", 0.0)), 2)
    return sorted(g.values(), key=lambda r: -r["dollars"])
