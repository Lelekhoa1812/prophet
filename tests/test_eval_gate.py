"""Decision-quality gate (HTML rule 7, 'prove it continuously'): a replay on the seeded mini twin must keep its alert quality.

Floors come from the first measured run (recorded in docs/GOAL_EVIDENCE.md) minus a margin, so ordinary changes pass and a
real regression in what the agents catch or how often they are right fails the build. If a deliberate change moves a
number, re-measure, change the floor in the same commit and say why in GOAL_CHANGELOG.
"""
import pandas as pd

from lab.policy.alerts import Economics
from lab.sim.replay import replay, scored_window

# Measured 19 Nov 2026 on the 90-day mini replay (deterministic, identical on two runs):
#   shortfall 15 alerts, precision 0.600, recall 0.143 (2 of 14)   freshness 241, 0.867, 0.811 (120 of 148)
#   customer 17, 0.529, 0.400 (2 of 5)                              margin 3, 1.000, 0.750 (3 of 4)
# Floors sit near 80% of those. Counts are small, so shortfall, customer and margin floors are coarse tripwires.
FLOORS = {   # agent: (min useful share of alerts, min share of recorded problems caught, min alerts)
    "shortfall": (0.45, 0.10, 10),
    "freshness": (0.78, 0.70, 180),
    "customer": (0.40, 0.30, 10),
    "margin": (0.60, 0.50, 2),
}


def window(mini):
    lo, hi = scored_window(mini)
    return str((hi - pd.Timedelta(days=89)).date()), str(hi.date())


def test_alert_quality_stays_above_the_recorded_floors(mini):
    start, end = window(mini)
    r = replay("mini", start=start, end=end, policy=Economics().describe(), persist=False)
    for agent, (prec, rec, n) in FLOORS.items():
        a = r["agents"][agent]
        assert a["proposals"] >= n, f"{agent}: only {a['proposals']} alerts"
        assert a["precision"] >= prec, f"{agent} precision {a['precision']:.3f} < {prec}"
        assert a["recall"] >= rec, f"{agent} recall {a['recall']:.3f} < {rec}"


def test_the_gate_fires_when_shortfall_is_silenced(mini):
    """Control: a deliberately broken shortfall setting must fail the same floors."""
    start, end = window(mini)
    r = replay("mini", start=start, end=end, agents=["shortfall"], params={"shortfall": {"min_dollars": 1e9}}, persist=False)
    a = r["agents"].get("shortfall")
    prec, rec, n = FLOORS["shortfall"]
    assert a is None or a["proposals"] < n or (a["recall"] or 0) < rec
