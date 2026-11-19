"""The leakage guard on a small seeded twin built inside the test, so CI runs it without the local data or TimesFM."""
import numpy as np
import pandas as pd
import pytest

from lab import store
from lab.engine import models, state
from lab.engine.models import TSB, Baseline, Router
from lab.twin import build
from lab.twin.generator import TwinConfig

from test_no_future_leak import key, truncated


def test_agents_do_not_use_data_after_the_as_of_day_mini(mini):
    from lab.agents import REGISTRY
    from lab.agents.base import AgentContext
    w = mini
    mondays = [d for d in pd.date_range(w.days[int(len(w.days) * 0.55)], w.days[-1] - pd.Timedelta(days=14), freq="D") if d.dayofweek == 0]
    dates = [mondays[i] for i in np.linspace(0, len(mondays) - 1, 6).astype(int)]
    seen = {n: 0 for n in REGISTRY}
    for d in dates:
        wt = truncated(w, d)
        for name, agent in REGISTRY.items():
            full = agent.run(AgentContext(w, d, state.ForecastService(w), {}, []))
            cut = agent.run(AgentContext(wt, d, state.ForecastService(wt), {}, []))
            assert key(full) == key(cut), f"{name} on {d.date()} changed when the future was removed"
            seen[name] += len(full)
    assert sum(seen.values()) > 0, "no agent raised anything: the test has no power"


def test_planted_leak_is_detected(mini):
    """Negative control: a world cut ten days too early must give different proposals for at least one agent."""
    from lab.agents import REGISTRY
    from lab.agents.base import AgentContext
    w = mini
    d = [x for x in pd.date_range(w.days[int(len(w.days) * 0.55)], w.days[-1] - pd.Timedelta(days=14), freq="D") if x.dayofweek == 0][5]
    wt = truncated(w, d - pd.Timedelta(days=10))
    diff = [n for n, a in REGISTRY.items() if key(a.run(AgentContext(w, d, state.ForecastService(w), {}, []))) != key(a.run(AgentContext(wt, d, state.ForecastService(wt), {}, [])))]
    assert diff
