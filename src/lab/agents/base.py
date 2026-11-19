"""Agent contract: an agent is a pure function from (business state at as_of) to proposals with evidence."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

from ..engine.state import ForecastService, WorldData


@dataclass
class AgentContext:
    world: WorldData
    as_of: pd.Timestamp
    fs: ForecastService
    params: dict
    lessons: list = field(default_factory=list)


@dataclass
class Proposal:
    agent: str
    as_of: str
    entity_type: str
    entity_id: str
    title: str
    severity: str                      # high / medium / low
    dollars_at_stake: float
    confidence: float
    evidence: dict
    action: dict
    dedupe_key: str
    owner: str = "buyer"
    autonomy_level: str = "prepares"   # explains / suggests / prepares / acts / runs
    proposal_id: str = ""

    def finalise(self) -> "Proposal":
        raw = f"{self.agent}|{self.dedupe_key}|{self.as_of}"
        self.proposal_id = "PR-" + hashlib.sha1(raw.encode()).hexdigest()[:10].upper()
        return self


def severity_for(dollars: float, confidence: float, hi: float = 2000, mid: float = 500) -> str:
    score = dollars * (0.5 + confidence / 2)
    return "high" if score >= hi else "medium" if score >= mid else "low"


def interp_cdf(value: float, q10: float, q50: float, q90: float) -> float:
    """P(demand <= value) from three quantile points, linear in between, tails tapered."""
    if q90 <= q10:
        return 1.0 if value >= q50 else 0.0
    xs = [max(0.0, q10 - (q50 - q10) * 0.6), q10, q50, q90, q90 + (q90 - q50) * 0.6]
    ys = [0.0, 0.1, 0.5, 0.9, 1.0]
    return float(np.interp(value, xs, ys))


def inv_cdf(p: float, q10: float, q50: float, q90: float) -> float:
    xs = [max(0.0, q10 - (q50 - q10) * 0.6), q10, q50, q90, q90 + (q90 - q50) * 0.6]
    ys = [0.0, 0.1, 0.5, 0.9, 1.0]
    return float(np.interp(p, ys, xs))


def series_evidence(hist_x: list, hist_y: list, fc_x: list, q10: list, q50: list, q90: list, unit: str, label: str) -> dict:
    return {"kind": "series", "label": label, "unit": unit,
            "history": {"x": hist_x, "y": [round(float(v), 2) for v in hist_y]},
            "forecast": {"x": fc_x, "q10": [round(float(v), 2) for v in q10], "q50": [round(float(v), 2) for v in q50],
                         "q90": [round(float(v), 2) for v in q90]}}


def dstr(ts) -> str:
    return pd.Timestamp(ts).strftime("%Y-%m-%d")


Agent = Callable[[AgentContext], list[Proposal]]
