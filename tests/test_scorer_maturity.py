"""Freshness alerts are scored on whether their lots expired. An alert whose 21-day horizon runs past the data end cannot be
judged yet, so it must be left out of precision rather than counted as a false alarm.

Bug: score() cut expiry truth at the window end, although the twin has data beyond it. Alerts made in the last weeks of
the window were marked false even when right, which showed up as a late-window precision collapse.
Fix: expiry truth runs to the data end, and an alert is scored only when its lots' use-by dates fall inside the data.
"""
import pandas as pd

from lab.sim.replay import scored_window
from lab.sim.scorer import score
from lab.store import open_dataset


def _data_end(name):
    con = open_dataset(name, read_only=True)
    end = pd.Timestamp(con.execute("select max(delivery_date) from order_lines").fetchone()[0])
    con.close()
    return end


def _fresh(pid, as_of, days_left):
    return {"agent": "freshness", "entity_type": "product", "entity_id": pid, "as_of": str(as_of.date()),
            "dollars_at_stake": 500.0, "severity": "low", "dedupe_key": f"freshness|{pid}", "proposal_id": f"PR-{pid}",
            "action": {"lots": [{"lot_id": f"L-{pid}", "days_left": days_left, "at_risk_qty": 5.0}]}}


def test_alerts_too_recent_to_mature_are_excluded_from_freshness_precision(mini):
    end = _data_end("mini")
    lo, hi = scored_window(mini)
    proposals = [
        _fresh("MATURE", end - pd.Timedelta(days=60), days_left=10),   # horizon ends well inside the data
        _fresh("IMMATURE", end - pd.Timedelta(days=3), days_left=10),  # horizon runs 7 days past the data end
    ]
    res = score("mini", proposals, (lo, hi))
    fresh = res["agents"]["freshness"]
    assert fresh["proposals"] == 1
    assert fresh["immature_excluded"] == 1
    assert "PR-MATURE" in res["flags"]["freshness"]
    assert "PR-IMMATURE" not in res["flags"]["freshness"]
