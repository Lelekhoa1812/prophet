from lab.policy.learning import SegmentLearner
from lab.policy.alerts import Economics, select
from lab.agents.base import Proposal


def test_segment_weight_follows_precision_and_is_shrunk_and_bounded():
    L = SegmentLearner()
    for _ in range(60):
        L.observe("shortfall", "SEAFOOD", True)
    for _ in range(40):
        L.observe("shortfall", "PORK", False)
    L.observe("shortfall", "PORK", True)
    for _ in range(3):
        L.observe("shortfall", "LAMB", False)         # thin evidence
    w = L.weights()
    assert w["shortfall|SEAFOOD"] > 1.0 and w["shortfall|PORK"] == 0.5
    assert 0.5 <= w["shortfall|LAMB"] <= 1.0 and w["shortfall|LAMB"] > w["shortfall|PORK"]
    assert max(w.values()) <= 2.0


def test_distrusted_segment_ranks_below_equal_trusted_one():
    econ = Economics(budget_per_owner=1, segment_weight={"shortfall|PORK": 0.5})
    a = Proposal("shortfall", "d", "product", "A", "t", "low", 300, 0.8, {}, {"type": "x"}, "k1", owner="buyer", segment="PORK").finalise()
    b = Proposal("shortfall", "d", "product", "B", "t", "low", 300, 0.8, {}, {"type": "x"}, "k2", owner="buyer", segment="SEAFOOD").finalise()
    shown, digest, _ = select([a, b], econ)
    assert [p.entity_id for p in shown] == ["B"] and [p.entity_id for p in digest] == ["A"]
