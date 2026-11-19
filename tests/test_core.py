"""Core rules: pseudonyms, governance, metrics, narration number check, read-only guard, twin determinism."""
from __future__ import annotations

import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lab.engine import metrics  # noqa: E402
from lab.engine.baselines import _band, moving_average  # noqa: E402
from lab.narration import bedrock  # noqa: E402
from lab.policy.governor import Governor, Lesson  # noqa: E402
from lab.source.pseudo import CodeBook, stable_hash  # noqa: E402
from lab.store import open_dataset  # noqa: E402
from lab.agents.base import Proposal  # noqa: E402


# ---- pseudonyms are stable across runs and carry no real ordering
def test_codebook_is_deterministic_and_dense():
    ids = [f"real-{i}" for i in range(50)]
    a = CodeBook("customer", "C", ids)
    b = CodeBook("customer", "C", list(reversed(ids)))
    assert a.codes == b.codes
    assert sorted(a.codes.values()) == [f"C-{i:04d}" for i in range(1, 51)]


def test_stable_hash_differs_by_kind():
    assert stable_hash("customer", "1") != stable_hash("product", "1")


# ---- governor: dedupe, cooldown, lesson suppression, material repeats still raised
def _p(key="shortfall|P-1", dollars=500.0, agent="shortfall", entity="P-1"):
    return Proposal(agent, "2026-10-01", "product", entity, "t", "medium", dollars, 0.7, {}, {}, key).finalise()


def test_cooldown_blocks_repeat_unless_materially_worse():
    g = Governor()
    d0 = date(2026, 10, 1)
    assert len(g.filter([_p(dollars=500)], d0)) == 1
    assert len(g.filter([_p(dollars=520)], d0 + timedelta(days=1))) == 0          # inside cooldown, not worse
    assert len(g.filter([_p(dollars=900)], d0 + timedelta(days=1))) == 1          # 50%+ worse: raised again


def test_lesson_suppresses_then_expires():
    g = Governor()
    g.lessons.append(Lesson("LS-1", "shortfall", "entity", "P-1", {"suppress_days": 14}, date(2026, 10, 1)))
    assert g.filter([_p()], date(2026, 10, 5)) == []
    assert len(g.filter([_p()], date(2026, 10, 20))) == 1


def test_lesson_for_other_agent_does_not_apply():
    g = Governor()
    g.lessons.append(Lesson("LS-2", "margin", "entity", "P-1", {"suppress_days": 30}, date(2026, 10, 1)))
    assert len(g.filter([_p()], date(2026, 10, 2))) == 1


# ---- metrics
def test_wape_and_coverage_basics():
    a = np.array([[10.0, 10.0]])
    assert metrics.wape(a, a) == 0.0
    assert metrics.wape(a, a * 1.1) == pytest.approx(0.1)
    q = np.stack([np.full((1, 2), v) for v in np.arange(0, 10)], axis=-1)   # quantile k is value k
    assert metrics.coverage(np.array([[4.0, 4.0]]), q, lo=0, hi=8) == 1.0


def test_pinball_zero_for_perfect_point_mass():
    a = np.array([[5.0]])
    q = np.full((1, 1, 9), 5.0)
    assert metrics.pinball(a, q) == 0.0


def test_band_widens_with_horizon_and_is_nonnegative():
    b = _band(np.array([1.0, 1.0, 1.0]), 2.0)
    assert (b >= 0).all()
    assert (b[-1, -1] - b[-1, 0]) > (b[0, -1] - b[0, 0])


def test_moving_average_flat_series():
    p, q = moving_average(np.full(20, 7.0), 3, 4)
    assert np.allclose(p, 7.0)


# ---- narration: a number that is not in the evidence is rejected
def test_validator_accepts_sourced_numbers_and_rejects_invented():
    p = {"title": "Chicken: short", "evidence": {"why": "Stock is 15 units against 113 forecast.", "facts": [{"label": "x", "value": 15}]},
         "action": {"qty": 263}, "dollars_at_stake": 214.06}
    ok, bad = bedrock.validate("Chicken has 15 units against 113 forecast; order 263 for about $214.", p)
    assert ok and bad == []
    ok, bad = bedrock.validate("Chicken will sell 999 units next week.", p)
    assert not ok and "999" in bad


# ---- read-only guard on the staging snapshot path
def test_open_dataset_read_only_refuses_writes(tmp_path, monkeypatch):
    import lab.config as cfg
    import lab.store as store
    db = tmp_path / "t.duckdb"
    monkeypatch.setitem(cfg.DATASETS, "tmp", db)
    monkeypatch.setitem(store.DATASETS, "tmp", db)
    con = open_dataset("tmp")
    con.execute("insert into meta values ('k','v')")
    con.close()
    ro = open_dataset("tmp", read_only=True)
    with pytest.raises(Exception):
        ro.execute("insert into meta values ('x','y')")
    ro.close()


# ---- twin determinism: same seed, same world
def test_twin_is_deterministic_for_a_seed():
    from lab.twin.generator import TwinConfig, build_world
    a = build_world(TwinConfig(seed=3, n_products=40, n_customers=30))
    b = build_world(TwinConfig(seed=3, n_products=40, n_customers=30))
    assert np.array_equal(a.demand, b.demand)
    assert [t["kind"] for t in a.truth] == [t["kind"] for t in b.truth]
