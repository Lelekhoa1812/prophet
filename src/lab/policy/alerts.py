"""One score, one budget: decide which alerts a person sees, and which go to the digest.

Motivation vs Logic
Motivation: separate hand-set gates (probability at least 55%, dollars at least $150) interact, miss the efficient
frontier, and mean nothing to a client. Each client has different margins, staffing and appetite for alerts.
Logic: every proposal carries one score, the expected dollars at stake (already probability-weighted by the agent).
A tenant states three business facts: what it costs a person to review one alert, what share of the at-risk dollars
a reviewed alert usually saves, and how many alerts one owner can handle in a morning. From those:
  floor  = review_cost / effectiveness      (below this an alert costs more to read than it saves)
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
    review_cost: float = 12.0            # dollars of staff time to read and act on one alert
    effectiveness: float = 0.5           # share of expected loss a reviewed alert typically recovers
    budget_per_owner: int = 5            # alerts one owner sees per morning
    weight_bounds: tuple = (0.5, 2.0)    # learning may scale an agent's score only within these
    agent_weight: dict = field(default_factory=dict)

    @property
    def floor(self) -> float:
        return self.review_cost / max(self.effectiveness, 0.05)

    def describe(self) -> dict:
        return {"review_cost": self.review_cost, "effectiveness": self.effectiveness, "budget_per_owner": self.budget_per_owner,
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


def score_of(p, econ: Economics) -> float:
    return float(p.dollars_at_stake if hasattr(p, "dollars_at_stake") else p["dollars_at_stake"]) * econ.agent_weight.get(
        p.agent if hasattr(p, "agent") else p["agent"], 1.0)


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
        if score_of(p, econ) < econ.floor:
            below.append(p)
            continue
        o = _owner(p)
        per_owner[o] = per_owner.get(o, 0) + 1
        (shown if per_owner[o] <= econ.budget_per_owner else digest).append(p)
    return shown, digest, below
