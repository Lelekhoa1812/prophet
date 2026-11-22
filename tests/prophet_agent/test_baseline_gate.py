"""Audit 2026-11-22, item 4: the baseline gate must cover the delivered horizon and block a loss to a simple rule on
TimesFM's own centre line. Before, any successful backtest passed the gate."""
from prophet_agent.runtime.contracts import _baseline_gate


def _bt(horizon, timesfm_mase, naive_mase, flagged=False):
    out = {"horizon": horizon, "best_baseline": "naive",
           "models": {"timesfm": {"mase": timesfm_mase}, "naive": {"mase": naive_mase}}}
    if flagged:
        out["flags"] = ["a flagged backtest"]
    return out


def test_a_backtest_shorter_than_the_delivered_horizon_does_not_vouch_for_it():
    gate = _baseline_gate({"horizon": 60}, [_bt(14, 0.8, 1.2)])
    assert gate["ok"] is False and "no backtest covers that horizon" in gate["detail"]


def test_a_backtest_at_or_beyond_the_delivered_horizon_passes_when_timesfm_wins():
    gate = _baseline_gate({"horizon": 14}, [_bt(14, 0.8, 1.2), _bt(30, 0.9, 1.1)])
    assert gate["ok"] is True


def test_timesfm_losing_to_a_simple_rule_on_its_own_centre_is_blocked():
    gate = _baseline_gate({"horizon": 14}, [_bt(14, 1.3, 1.1)])
    assert gate["ok"] is False and gate["severity"] == "block" and "switch the centre" in gate["detail"]


def test_the_same_loss_is_allowed_once_the_centre_is_switched_to_the_rule():
    delivered = {"horizon": 14, "centre": {"rule": "naive", "reason": "the backtest found this simple rule more accurate"}}
    assert _baseline_gate(delivered, [_bt(14, 1.3, 1.1)])["ok"] is True


def test_a_flagged_result_is_not_blocked_by_this_gate():
    assert _baseline_gate({"horizon": 14}, [_bt(14, 1.3, 1.1, flagged=True)])["ok"] is True
