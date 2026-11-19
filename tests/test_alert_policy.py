from lab.agents.base import Proposal
from lab.policy.alerts import Economics, bounded_weight, from_dict, select
from lab.policy.governor import Governor
from datetime import date


def prop(i, dollars, owner="buyer", action="raise_purchase_order", agent="shortfall"):
    return Proposal(agent, "2026-01-01", "product", f"P{i}", "t", "low", dollars, 0.8, {}, {"type": action}, f"shortfall|P{i}", owner=owner).finalise()


def test_floor_is_review_cost_over_effectiveness():
    assert Economics(review_cost=10, effectiveness=0.5).floor == 20


def test_budget_ranks_by_score_and_sends_rest_to_digest():
    econ = Economics(review_cost=10, effectiveness=0.5, budget_per_owner=2)
    ps = [prop(i, d) for i, d in enumerate([500, 50, 300, 10, 100])]
    shown, digest, below = select(ps, econ)
    assert [p.dollars_at_stake for p in shown] == [500, 300]
    assert [p.dollars_at_stake for p in digest] == [100, 50]
    assert [p.dollars_at_stake for p in below] == [10]


def test_budget_is_per_owner_and_order_stops_are_exempt():
    econ = Economics(review_cost=10, effectiveness=0.5, budget_per_owner=1)
    ps = [prop(1, 400, "buyer"), prop(2, 300, "buyer"), prop(3, 200, "sales"), prop(4, 5, "buyer", "hold_order_line")]
    shown, digest, _ = select(ps, econ)
    ids = {p.entity_id for p in shown}
    assert ids == {"P1", "P3", "P4"} and [p.entity_id for p in digest] == ["P2"]


def test_learned_weight_is_clamped_and_logged():
    econ = from_dict({"review_cost": 10})
    assert bounded_weight(econ, "shortfall", 9.0) == econ.weight_bounds[1]
    assert econ.describe()["agent_weight"]["shortfall"] == econ.weight_bounds[1]


def test_governor_commit_false_does_not_start_cooldown():
    g = Governor()
    p = prop(1, 300)
    assert len(g.filter([p], date(2026, 1, 1), commit=False)) == 1
    assert len(g.filter([p], date(2026, 1, 2), commit=False)) == 1      # not recorded, so no cooldown
    g.commit([p], date(2026, 1, 2))
    assert g.filter([p], date(2026, 1, 3), commit=False) == []
