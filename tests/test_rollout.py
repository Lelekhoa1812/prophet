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
    few = [("shortfall", "approve", None, "P-1")] * 5
    assert rollout.learn("t", few, cfg) == []
    # 30 useful, 0 wrong: precision 1.0, so a step up capped at +10% per call
    good = [("shortfall", "approve", None, "P-1")] * 30
    ch = rollout.learn("t", good, rollout.load("t"))
    assert ch and [c for c in ch if c["kind"] == "agent"][0]["new"] == 1.1
    for _ in range(20):
        rollout.learn("t", good, rollout.load("t"))
    assert rollout.economics_of(rollout.load("t")).agent_weight["shortfall"] == Economics().weight_bounds[1]
    assert json.loads((tmp_path / "tenant_t.json").read_text())["learning_log"]


def test_operator_context_dismissals_do_not_count_against_an_agent(tmp_path, monkeypatch):
    monkeypatch.setattr(rollout, "DATA", tmp_path)
    rows = [("freshness", "approve", None, "P-1")] * 10 + [("freshness", "dismiss", "one_off", "P-1")] * 40
    assert rollout.learn("t", rows, rollout.load("t")) == [] or rollout.learn("t", rows, rollout.load("t"))[0]["basis"]["not_useful"] == 0
