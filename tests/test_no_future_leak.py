"""End to end: every agent must raise the same proposals on a world with everything after the as-of day removed.

Needs the local twin (data/ is git-ignored), so it is skipped where the twin has not been built.
"""
from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest

from lab.config import DATASETS

pytestmark = pytest.mark.skipif(not Path(DATASETS["twin"]).exists(), reason="twin dataset not built")


def truncated(w, as_of):
    keep_days = w.days <= as_of
    demand = w.demand.copy()
    demand[:, ~keep_days] = 0.0
    return replace(
        w, lines=w.lines[w.lines.delivery_date <= as_of], lots=w.lots[w.lots.received_date <= as_of],
        moves=w.moves[w.moves.moved_on <= as_of], pos=w.pos[w.pos.created_on <= as_of], demand=demand)


def key(ps):
    return sorted((p.agent, p.entity_id, round(p.dollars_at_stake, 4), p.dedupe_key, p.severity) for p in ps)


def test_agents_do_not_use_data_after_the_as_of_day():
    from lab.agents import REGISTRY
    from lab.agents.base import AgentContext
    from lab.engine.state import ForecastService, get_world, week_start
    from lab.sim.replay import scored_window
    import numpy as np
    w = get_world("twin")
    lo, hi = scored_window(w)
    cached = [d for d in pd.date_range(lo, hi, freq="D")
              if (Path(DATASETS["twin"]).parent / "cache" / f"twin_wk_{week_start(d):%Y%m%d}_4.pkl").exists()]
    mondays = [d for d in cached if d.dayofweek == 0]          # customer and margin checks only run on Mondays
    dates = [mondays[i] for i in np.linspace(0, len(mondays) - 1, 8).astype(int)]
    seen = {name: 0 for name in REGISTRY}
    for d in dates:
        wt = truncated(w, d)
        for name, agent in REGISTRY.items():
            full = agent.run(AgentContext(w, d, ForecastService(w), {}, []))
            cut = agent.run(AgentContext(wt, d, ForecastService(wt), {}, []))
            assert key(full) == key(cut), f"{name} on {d.date()} changed when the future was removed"
            seen[name] += len(full)
    # Power check: a test that never saw a proposal proves nothing. Margin raised only 1 on these dates (weak).
    # Negative control, done by hand: cutting 10 days too early changed shortfall (3 vs 7) and freshness (38 vs 44).
    assert all(seen[a] > 0 for a in ("shortfall", "freshness", "customer")), seen
