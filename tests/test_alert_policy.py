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


def test_false_alarm_cost_demotes_uncertain_alerts_of_equal_size():
    econ = Economics(review_cost=10, effectiveness=0.5, budget_per_owner=1, false_alarm_cost=200, ranking_false_alarm_cost={})
    sure = prop(1, 300); sure.p_event = 0.95
    unsure = prop(2, 300); unsure.p_event = 0.55
    shown, digest, below = select([unsure, sure], econ)
    assert [p.entity_id for p in shown] == ["P1"]
    # 0.5 x 300 - 10 - 0.45 x 200 = 50 > 0, so the unsure one is above water but over budget
    assert [p.entity_id for p in digest] == ["P2"] and below == []


def test_false_alarm_cost_can_push_an_alert_below_zero():
    econ = Economics(review_cost=10, effectiveness=0.5, budget_per_owner=5, false_alarm_cost=400, ranking_false_alarm_cost={})
    weak = prop(3, 150); weak.p_event = 0.6          # 75 - 10 - 160 < 0
    shown, digest, below = select([weak], econ)
    assert shown == [] and digest == [] and len(below) == 1


def test_agent_ranking_cost_changes_only_that_agents_score():
    import pytest
    from types import SimpleNamespace as N
    from lab.policy.alerts import Economics, net_value
    mk = lambda agent: N(agent=agent, dollars_at_stake=1000.0, p_event=0.7, segment=None)
    base = Economics(ranking_false_alarm_cost={})
    tuned = Economics(ranking_false_alarm_cost={"shortfall": 1000.0})
    assert net_value(mk("margin"), tuned) == net_value(mk("margin"), base)
    assert net_value(mk("shortfall"), tuned) == pytest.approx(net_value(mk("shortfall"), base) - 0.3 * (1000.0 - 200.0))


def test_default_economics_rank_shortfall_with_the_tuned_cost_and_state_it():
    from lab.policy.alerts import Economics, from_dict
    d = Economics().describe()
    assert d["ranking_false_alarm_cost"] == {"shortfall": 1500.0, "freshness": 1500.0} and d["false_alarm_cost"] == 200.0
    assert from_dict(d).ranking_false_alarm_cost == {"shortfall": 1500.0, "freshness": 1500.0}
    assert from_dict({"review_cost": 10}).ranking_false_alarm_cost == {"shortfall": 1500.0, "freshness": 1500.0}     # old tenant files get the default
