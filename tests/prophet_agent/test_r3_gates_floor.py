"""Round 3 gate B4, goal 1 (A1-04, A4-02, A3-14, A7-03): a sales series with a few refund days must not forecast negative.

Live evidence (measured): retail rep3, obs 16 and 30. The history held two genuine negative days (returns). One returns day
switched off the zero floor, so Saturdays came out at -3,125 and -3,606 and the 14-day total included them. The stub engines
below stand in for TimesFM 3; the real model is never loaded here.
"""
import threading
from types import SimpleNamespace

import numpy as np

from prophet_agent.forecast.timesfm import TimesFMEngine
from prophet_agent.runtime.contracts import check_evidence, check_forecast


def _engine(predict_batch, global_context=400):
    eng = object.__new__(TimesFMEngine)  # bypass __init__, which loads the real model
    eng.model = SimpleNamespace(global_context=global_context, predict_batch=predict_batch)
    eng._infer_lock = threading.Lock()
    eng.checkpoint = "stub"
    return eng


def _sales_with_refunds(n=300, refund_days=(40, 200)):
    """Positive daily sales with a weekly cycle, and two refund days below zero (as in the live retail file)."""
    t = np.arange(n)
    x = 1000.0 + 300.0 * np.sin(2 * np.pi * t / 7)
    for d in refund_days:
        x[d] = -1683.5
    return x


def _negative_forecast_stub(horizon_value=-3606.0):
    """A stub that forecasts a negative Saturday, as the live model did."""
    def predict_batch(contexts, horizon, return_quantiles):
        out = []
        for _ in contexts:
            point = np.array([500.0, horizon_value, 800.0, 900.0][:horizon], dtype=np.float32)
            q = np.stack([point - 100.0 * k for k in range(1, 10)], axis=-1)  # band reaches below zero too
            out.append(SimpleNamespace(forecast=point, quantiles=q))
        return out
    return predict_batch


def test_two_refund_days_do_not_switch_off_the_floor_for_a_sales_series():
    # Before the fix: any negative in the history skipped the floor, so the point and band came back negative.
    fc = _engine(_negative_forecast_stub()).forecast(_sales_with_refunds(), horizon=4)
    assert (fc.point >= 0).all(), f"a sales series forecast below zero: {fc.point}"
    assert (fc.q >= 0).all(), "the band must be floored with the point"


def test_a_genuinely_signed_series_keeps_its_negative_forecast():
    from prophet_agent.forecast.timesfm import nonnegative_domain
    # Returns, temperatures and balances: half the history is negative, so the floor must stay off.
    returns = np.sin(np.arange(300) * 0.7) * 2.0
    assert not nonnegative_domain(returns)
    fc = _engine(_negative_forecast_stub(-3.0)).forecast(returns, horizon=4)
    assert fc.point[1] == -3.0


def test_nonnegative_domain_allows_a_small_share_of_isolated_negatives_only():
    from prophet_agent.forecast.timesfm import nonnegative_domain
    assert nonnegative_domain(np.array([1.0, 2.0, 3.0]))
    assert nonnegative_domain(_sales_with_refunds())          # 2 negatives in 300: 0.7 per cent
    assert not nonnegative_domain(np.array([-1.0, 2.0]))      # 1 in 2
    assert not nonnegative_domain(np.linspace(-5, 5, 100))    # signed series
    assert not nonnegative_domain(np.array([np.nan, 1.0]))    # not finite: never treated as a sales series


def test_controlled_forecast_also_floors_a_sales_series_with_refund_days():
    from prophet_agent.forecast.control import Control
    eng = _engine(_negative_forecast_stub())
    eng.model = SimpleNamespace(global_context=400)

    def decode(tgt, po, pf, tmask, horizon):   # the controlled path decodes through model.model, so stub that call
        base = np.array([500.0, -3606.0, 800.0, 900.0][:horizon], dtype=np.float32)
        out = np.stack([base + 100.0 * (k - 4) for k in range(9)], axis=-1)   # (horizon, 9), reaches below zero
        return out[None]
    eng._decode = decode
    fc = eng.forecast_controlled(_sales_with_refunds(), 4, Control())
    assert (fc.point >= 0).all() and (fc.q >= 0).all()


def test_contract_blocks_a_negative_forecast_on_a_history_that_is_essentially_non_negative():
    # Live retail rep3 day 9: -3,606 on a history whose only negatives are two refund days. Before the fix this check was
    # skipped because the history had a negative value.
    hist = list(_sales_with_refunds())
    checks = check_forecast([500.0, -3606.0, 800.0, 900.0], history=hist)
    named = [c for c in checks if c["name"] == "non_negative_when_history_is"]
    assert len(named) == 1 and not named[0]["ok"] and named[0]["severity"] == "block"
    assert "-3606" not in named[0]["detail"] and "1 forecast" in named[0]["detail"]


def test_contract_blocks_a_negative_band_on_an_essentially_non_negative_history():
    hist = list(_sales_with_refunds())
    checks = check_forecast([500.0, 600.0], q10=[-20.0, 100.0], q90=[700.0, 800.0], history=hist)
    c = next(c for c in checks if c["name"] == "non_negative_when_history_is")
    assert not c["ok"] and c["severity"] == "block" and "band" in c["detail"]


def test_contract_does_not_floor_a_signed_history():
    hist = list(np.sin(np.arange(300) * 0.7) * 2.0)
    assert all(c["name"] != "non_negative_when_history_is" for c in check_forecast([-1.0, -2.0], history=hist))


def test_a_blocked_negative_forecast_fails_the_evidence_check_with_a_plain_reason():
    fc_obs = {"kind": "tool", "tool": "timesfm_forecast", "args": {"values": list(_sales_with_refunds()), "horizon": 4},
              "result": {"ok": True, "config": {"horizon": 4}, "health": {"verdict": "ok"},
                         "point": [500.0, -3606.0, 800.0, 900.0], "q10": [400.0, -3700.0, 700.0, 800.0],
                         "q90": [600.0, -3500.0, 900.0, 1000.0]}}
    inspect = {"kind": "tool", "tool": "inspect_data", "args": {}, "result": {"ok": True}}
    blocked = [c for c in check_evidence([inspect, fc_obs]) if not c["ok"] and c["severity"] == "block"]
    assert any(c["name"] == "non_negative_when_history_is" for c in blocked)
    assert any("below zero" in c["detail"] for c in blocked)
