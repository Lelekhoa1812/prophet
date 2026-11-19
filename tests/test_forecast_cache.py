"""Forecast cache must never serve a forecast made by another model, route or history; and the past cannot see the future."""
import numpy as np
import pandas as pd
import pytest

from lab.engine import models, state
from lab.engine.forecast import Forecast
from lab.engine.models import Router
from lab.engine.state import ForecastService, WorldData


class Const:
    licence = "own"; min_history = 4

    def __init__(self, name, value):
        self.name, self.value = name, value

    def predict(self, contexts, horizon, season=52):
        q = np.tile(np.linspace(self.value * 0.8, self.value * 1.2, 9), (len(contexts), horizon, 1))
        return Forecast(q[:, :, 4].copy(), q)


def world(weeks=40, bump_from=None):
    first = pd.Timestamp("2026-01-05")                     # a Monday
    days = pd.date_range(first, periods=weeks * 7, freq="D")
    demand = np.full((2, len(days)), 10.0)
    if bump_from is not None:
        demand[:, (pd.Timestamp(bump_from) - first).days:] = 999.0
    prods = pd.DataFrame({"product_id": ["p0", "p1"], "unit_cost": [1.0, 1.0]})
    empty = pd.DataFrame()
    return WorldData("t", prods, empty, empty, empty, empty, empty, days, {"p0": 0, "p1": 1}, demand, first, days[-1])


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    monkeypatch.setattr(state, "CACHE_DIR", tmp_path)
    yield
    models.set_router(None)


def router(value, default="a", route=None):
    return Router(registry={"a": Const("a", value), "b": Const("b", value * 3)}, default=default, route=route or {})


AS_OF = pd.Timestamp("2026-09-07")


def median(fs):
    return float(fs.weekly(AS_OF, 4)["fc"].q[0, 0, 4])


def test_route_change_is_not_served_from_cache():
    models.set_router(router(10, default="a"))
    fs = ForecastService(world())
    assert median(fs) == pytest.approx(10.0)
    models.set_router(router(10, default="b"))               # same data, different model
    assert median(ForecastService(world())) == pytest.approx(30.0)


def test_changed_history_is_not_served_from_cache():
    models.set_router(router(10))
    first = ForecastService(world()); median(first)
    revised = ForecastService(world())
    revised.w.demand[:, 5] += 1.0                             # a late correction inside the history window
    assert not np.array_equal(first.weekly(AS_OF, 4)["mat"], revised.weekly(AS_OF, 4)["mat"])


def test_future_data_does_not_change_the_forecast():
    models.set_router(router(10))
    a = ForecastService(world(weeks=40)).weekly(AS_OF, 4)
    b = ForecastService(world(weeks=40, bump_from="2026-09-07")).weekly(AS_OF, 4)   # everything from as_of on differs
    assert np.array_equal(a["mat"], b["mat"])
    assert np.array_equal(a["fc"].q, b["fc"].q)
