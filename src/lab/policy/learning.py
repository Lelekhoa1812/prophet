"""Segment learning: which kinds of alert this client finds useful.

Motivation vs Logic
Motivation: on the twin, shortfall alerts for pork and sausages were useful 17 to 18% of the time while seafood and produce
reached 68 to 77%. A single per-agent weight cannot see that, because each owner already gets one agent's alerts. A client
needs the system to learn, from its own approvals and dismissals, which segments to trust.
Logic: count useful and not-useful outcomes per (agent, segment). A segment's weight is its precision relative to the agent's
overall precision, shrunk toward 1 with `prior` pseudo-alerts so thin evidence moves nothing, and clamped to the tenant's
bounds. Weights multiply expected dollars before the floor and the ranking, so a distrusted segment drops down the list and
may fall into the digest, never disappears. Every weight can be listed and reset.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class SegmentLearner:
    prior: float = 10.0
    bounds: tuple = (0.5, 2.0)
    good: dict = field(default_factory=dict)
    total: dict = field(default_factory=dict)
    agent_good: dict = field(default_factory=dict)
    agent_total: dict = field(default_factory=dict)

    def observe(self, agent: str, segment: str | None, useful: bool) -> None:
        self.agent_total[agent] = self.agent_total.get(agent, 0) + 1
        self.agent_good[agent] = self.agent_good.get(agent, 0) + int(useful)
        if segment is None:
            return
        k = f"{agent}|{segment}"
        self.total[k] = self.total.get(k, 0) + 1
        self.good[k] = self.good.get(k, 0) + int(useful)

    def weights(self) -> dict[str, float]:
        out = {}
        lo, hi = self.bounds
        for k, n in self.total.items():
            agent = k.split("|")[0]
            p0 = max(self.agent_good.get(agent, 0) / max(self.agent_total.get(agent, 1), 1), 0.05)
            p = (self.good.get(k, 0) + self.prior * p0) / (n + self.prior)
            out[k] = round(min(max(p / p0, lo), hi), 3)
        return out
