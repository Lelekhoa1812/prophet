"""The backtest_series tool's new arguments: combine must be a boolean, and n_origins defaults to twelve. The model is not
loaded: argument errors are raised before any engine call."""
from __future__ import annotations

import pytest

from prophet_agent.tools import data_tools as D


def test_combine_must_be_true_or_false():
    with pytest.raises(ValueError, match="combine must be true or false"):
        D._backtest({"values": [float(i % 5) for i in range(120)], "horizon": 4, "combine": "yes"}, "k")


def test_combine_and_n_origins_are_documented_in_the_tool_spec():
    assert "combine" in D.BACKTEST_SPEC["args"] and "default 12" in D.BACKTEST_SPEC["args"]["n_origins"]


def test_a_panel_is_backtested_in_one_call_with_a_summary(monkeypatch):
    import numpy as np
    from prophet_agent.runtime import contracts

    def fake(series, horizon, **kw):
        return {"models": {"naive": {"mase": 1.0}, "timesfm": {"mase": 0.9}}, "best_baseline": "naive",
                "verdict": "timesfm_better", "timesfm_range_coverage_80": 0.8, "origins": 12, "horizon": horizon}
    monkeypatch.setattr(D, "rolling_backtest", fake)
    monkeypatch.setattr(D, "resolve_series", lambda a: [("a", np.ones(100)), ("b", np.ones(100))])
    out = D._backtest({"source": {"run_id": "r", "file": "out/series.json"}, "horizon": 7}, "k")
    assert out["n_series"] == 2 and out["verdict_counts"] == {"timesfm_better": 2}
    assert set(out["summary"]) == {"a", "b"} and "series" not in D.observe_backtest(out)
    assert not contracts.blocking(contracts.check_backtest(out))
    with pytest.raises(ValueError, match="control options apply to one series"):
        D._backtest({"source": {"run_id": "r", "file": "out/series.json"}, "horizon": 7, "znorm": True}, "k")
