"""Shared fixtures. `mini` is a 60-product seeded twin built once per test module with a cheap router (no TimesFM, no local data)."""
import pytest

from lab import store
from lab.engine import models, state
from lab.engine.models import TSB, Baseline, Router
from lab.twin import build
from lab.twin.generator import TwinConfig


@pytest.fixture(scope="module")
def mini(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("mini")
    mp = pytest.MonkeyPatch()
    mp.setattr(store, "DATASETS", {"mini": tmp / "mini.duckdb"})
    (tmp / "cache").mkdir()
    mp.setattr(state, "CACHE_DIR", tmp / "cache")
    state._WORLDS.pop("mini", None)
    models.set_router(Router(registry={"ma8": Baseline("ma8"), "ma4": Baseline("ma4"), "tsb": TSB()}, default="ma8", route={"intermittent": "tsb"}))
    build.main(name="mini", cfg=TwinConfig(seed=11, start="2024-01-01", end="2025-09-30", n_products=60, n_customers=40, n_suppliers=5))
    yield state.get_world("mini")
    models.set_router(None)
    state._WORLDS.pop("mini", None)
    mp.undo()
