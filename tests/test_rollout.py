import json

from lab.policy import rollout
from lab.policy.alerts import Economics


def test_phase_demo_shadow_live(tmp_path, monkeypatch):
    cfg = {"go_live": None, "shadow_days": 14}
    assert rollout.phase(cfg, "2026-01-01")["phase"] == "demo"
    cfg["go_live"] = "2026-01-10"
    assert rollout.phase(cfg, "2026-01-05")["phase"] == "setup"
    assert rollout.phase(cfg, "2026-01-20")["phase"] == "shadow"
    assert rollout.phase(cfg, "2026-01-24")["phase"] == "live"


def test_learning_is_bounded_logged_and_needs_enough_decisions(tmp_path, monkeypatch):
    monkeypatch.setattr(rollout, "DATA", tmp_path)
    cfg = rollout.load("t")
    few = [("shortfall", "approve", None, "P-1", f"a{i}") for i in range(5)]
    assert rollout.learn("t", few, cfg) == []
    # 30 useful, 0 wrong: precision 1.0, so a step up capped at +10% per batch
    good = lambda tag: [("shortfall", "approve", None, "P-1", f"{tag}{i}") for i in range(30)]
    ch = rollout.learn("t", good("b"), rollout.load("t"))
    assert ch and [c for c in ch if c["kind"] == "agent"][0]["new"] == 1.1
    for k in range(20):                                 # new evidence each time: moves, but stays inside the bound
        rollout.learn("t", good(f"c{k}") , rollout.load("t"))
    assert rollout.economics_of(rollout.load("t")).agent_weight["shortfall"] == Economics().weight_bounds[1]
    assert json.loads((tmp_path / "tenant_t.json").read_text())["learning_log"]


def test_same_decisions_are_not_learned_twice(tmp_path, monkeypatch):
    """Bug: every call re-applied the same decisions, so pressing Learn repeatedly ratcheted the weight to its bound."""
    monkeypatch.setattr(rollout, "DATA", tmp_path)
    rows = [("shortfall", "approve", None, "P-1", f"d{i}") for i in range(30)]
    first = rollout.learn("t", rows, rollout.load("t"))
    assert first
    w = rollout.economics_of(rollout.load("t")).agent_weight["shortfall"]
    for _ in range(5):
        assert rollout.learn("t", rows, rollout.load("t")) == []
    assert rollout.economics_of(rollout.load("t")).agent_weight["shortfall"] == w


def test_a_few_new_decisions_do_not_trigger_another_step(tmp_path, monkeypatch):
    monkeypatch.setattr(rollout, "DATA", tmp_path)
    rows = [("shortfall", "approve", None, "P-1", f"d{i}") for i in range(30)]
    rollout.learn("t", rows, rollout.load("t"))
    more = rows + [("shortfall", "approve", None, "P-1", f"n{i}") for i in range(3)]
    assert rollout.learn("t", more, rollout.load("t")) == []


def test_operator_context_dismissals_do_not_count_against_an_agent(tmp_path, monkeypatch):
    monkeypatch.setattr(rollout, "DATA", tmp_path)
    rows = [("freshness", "approve", None, "P-1", f"a{i}") for i in range(10)] + [("freshness", "dismiss", "one_off", "P-1", f"b{i}") for i in range(40)]
    assert rollout.learn("t", rows, rollout.load("t")) == [] or rollout.learn("t", rows, rollout.load("t"))[0]["basis"]["not_useful"] == 0
