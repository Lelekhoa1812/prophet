import pytest
from fastapi.testclient import TestClient

from lab.api import main
from lab.policy.alerts import Economics
from lab.policy.whatif import whatif

AG = {"shortfall": {"proposals": 100, "useful": 50, "dollars_truth_caught": 10_000.0, "dollars_truth_total": 20_000.0, "dollars_at_stake_total": 70_000.0},
      "freshness": {"proposals": 200, "useful": 160, "dollars_truth_caught": 40_000.0, "dollars_truth_total": 50_000.0, "dollars_at_stake_total": 60_000.0}}
ECON = Economics(review_cost=10.0, false_alarm_cost=100.0)


def test_arithmetic_per_agent_and_total():
    r = whatif(AG, ECON, acceptance=0.5, effectiveness=0.4)
    s = {a["agent"]: a for a in r["agents"]}["shortfall"]
    assert s["gross"] == pytest.approx(10_000 * 0.5 * 0.4)
    assert s["review_cost"] == pytest.approx(100 * 10)
    assert s["false_alarm_cost"] == pytest.approx(50 * 0.5 * 100)
    assert s["net"] == pytest.approx(2000 - 1000 - 2500)
    assert r["total"]["net"] == pytest.approx(sum(a["net"] for a in r["agents"]))


def test_net_rises_with_acceptance_when_alerts_are_worth_it_and_break_even_is_found():
    lo = whatif(AG, ECON, 0.1, 0.5)["total"]["net"]
    hi = whatif(AG, ECON, 0.9, 0.5)["total"]["net"]
    assert hi > lo
    be = whatif(AG, ECON, 0.5, 0.5)["break_even_acceptance"]
    assert be is not None and whatif(AG, ECON, be, 0.5)["total"]["net"] == pytest.approx(0, abs=1e-6)


def test_no_break_even_when_every_acted_alert_loses_money():
    bad = {"a": {"proposals": 100, "useful": 1, "dollars_truth_caught": 100.0, "dollars_truth_total": 100.0}}
    assert whatif(bad, ECON, 0.5, 0.5)["break_even_acceptance"] is None


def test_labels_say_expected_not_observed():
    r = whatif(AG, ECON, 0.5, 0.5)
    assert r["evidence"] == "Synthetic replay" and "not observed" in r["caveat"].lower()


def test_endpoint_uses_the_stored_scorecard(monkeypatch):
    monkeypatch.setattr(main, "_available", lambda: ["twin"])
    monkeypatch.setattr(main, "_scorecard", lambda ds: {"agents": AG, "window": ["2025-01-01", "2025-02-01"]})
    c = TestClient(main.app, base_url="http://localhost:8000")
    r = c.get("/api/policy/whatif", params={"dataset": "twin", "acceptance": 0.5, "effectiveness": 0.5}).json()
    assert r["total"]["gross"] == pytest.approx(50_000 * 0.25) and r["evidence"] == "Synthetic replay"
    assert len(r["grid"]) >= 9


def test_agents_without_dollar_truth_are_listed_as_unmeasured_not_netted():
    ag = {**AG, "customer": {"proposals": 50, "useful": 10, "dollars_truth_caught": 0.0, "dollars_truth_total": 0.0}}
    r = whatif(ag, ECON, 0.5, 0.5)
    assert r["unmeasured_agents"] == ["customer"]
    assert r["total"]["net"] == whatif(AG, ECON, 0.5, 0.5)["total"]["net"]
    assert [x for x in r["agents"] if x["agent"] == "customer"][0]["value_measured"] is False


def test_claimed_dollars_are_shown_next_to_matched_truth():
    s = {a["agent"]: a for a in whatif(AG, ECON, 0.5, 0.5)["agents"]}["shortfall"]
    assert s["claimed_dollars"] == 70_000.0 and s["truth_dollars_total"] == 20_000.0


def test_problem_level_collapses_repeat_alerts_on_one_problem():
    from lab.policy.whatif import problem_level
    rows = [("shortfall", "P1", "2026-01-01", 100.0), ("shortfall", "P1", "2026-01-03", 300.0), ("shortfall", "P1", "2026-01-09", 200.0),   # one problem, 3 alerts
            ("shortfall", "P1", "2026-02-20", 50.0),                                                                                      # a new problem
            ("shortfall", "P2", "2026-01-02", 40.0), ("freshness", "L1", "2026-01-02", 500.0)]
    r = problem_level(rows, gap_days=10)
    assert r["shortfall"] == {"alerts": 5, "problems": 3, "claimed_dollars": 300.0 + 50.0 + 40.0}
    assert r["freshness"]["problems"] == 1


def test_estimated_dollars_stay_out_of_the_headline():
    # Motivation vs Logic: customer and margin dollars are baseline estimates; they must not enter the headline net.
    from lab.policy.whatif import measured
    assert measured({"dollars_truth_total": 500_000.0, "dollar_basis": "estimate"}) is False
    assert measured({"dollars_truth_total": 500_000.0, "dollar_basis": "recorded"}) is True
    assert measured({"dollars_truth_total": 500_000.0}) is True
