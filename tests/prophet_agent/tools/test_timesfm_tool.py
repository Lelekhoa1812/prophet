"""TimesFM runtime tool (GOAL G9). The offline tests stub the engine; the marked test runs the real model end-to-end
through the durable runtime with a scripted planner, so a real forecast is recorded in the ledger."""
from __future__ import annotations

import numpy as np
import pytest

from prophet_agent.forecast import timesfm as timesfm_mod
from prophet_agent.forecast.types import Forecast
from prophet_agent.runtime.goal import goal_program
from prophet_agent.runtime.runner import Runtime
from prophet_agent.runtime.store import Store
from prophet_agent.tools import timesfm_tool


class _StubEngine:
    checkpoint = "stub-checkpoint"
    global_context = 16

    def forecast_batch(self, series_list, horizon):
        n = len(series_list)
        point = np.stack([np.full(horizon, float(np.asarray(x)[-1])) for x in series_list])
        q = np.tile(np.linspace(0, 8, 9), (n, horizon, 1))
        return Forecast(point, q)


@pytest.fixture()
def stub_engine(monkeypatch):
    monkeypatch.setattr(timesfm_mod.TimesFMEngine, "get", classmethod(lambda cls: _StubEngine()))


def test_tool_records_configuration_and_context_used(stub_engine):
    from prophet_agent.tools.timesfm_tool import _run
    out = _run({"values": list(range(40)), "horizon": 3}, "k")
    assert out["config"]["global_context"] == 16
    assert out["config"]["context_used"] == 16
    assert out["config"]["history_points"] == 40
    assert out["config"]["horizon"] == 3
    assert len(out["point"]) == 3
    assert sorted(out["quantiles"]) == ["0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9"]
    assert "licence" not in out["config"] and "research use" not in str(out).lower()  # licence terms live in NOTICE.md only


def test_intermittent_demand_reports_expected_demand_as_the_point(stub_engine):
    from prophet_agent.forecast.tsb import expected_volume
    from prophet_agent.tools.timesfm_tool import _run
    x = [0.0] * 20 + [4.0, 0.0, 0.0, 6.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 5.0]   # sells in 3 of 32 periods
    out = _run({"values": x, "horizon": 3}, "k")
    assert out["point"] == [round(float(v), 4) for v in expected_volume(np.array(x), 3)]
    assert out["median_point"] == [5.0, 5.0, 5.0]       # the stub's median, kept visible
    assert out["config"]["point_basis"].startswith("expected demand")


def test_continuous_demand_keeps_the_median_as_the_point(stub_engine):
    from prophet_agent.tools.timesfm_tool import _run
    out = _run({"values": [10.0 + i for i in range(40)], "horizon": 3}, "k")   # sells every period
    assert out["point"] == [49.0, 49.0, 49.0]
    assert "median_point" not in out
    assert out["config"]["point_basis"] == "median"


@pytest.mark.parametrize("args", [
    {"values": [], "horizon": 3},
    {"values": [1.0], "horizon": 0},
    {"values": [1.0], "horizon": 10_000},
    {"values": "abc", "horizon": 3},
])
def test_tool_rejects_bad_input_before_loading_anything(args):
    from prophet_agent.tools.timesfm_tool import _run
    with pytest.raises(ValueError):
        _run(args, "k")


def test_tool_is_declared_idempotent():
    assert timesfm_tool().idempotent is True


@pytest.mark.timesfm
def test_real_timesfm_forecast_through_the_durable_runtime(tmp_path):
    rng = np.random.default_rng(1)
    t = np.arange(156)
    values = (20 + 4 * np.sin(2 * np.pi * t / 52) + 0.05 * t + rng.normal(0, 0.7, t.size)).round(3).tolist()

    class PlannerOnce:
        def __init__(self):
            self.plans = [
                {"action": "call_tool", "tool": "timesfm_forecast", "args": {"values": values, "horizon": 12},
                 "rationale": "forecast the next 12 weeks"},
                {"action": "finish", "rationale": "enough"},
            ]
            self.evals = [{"verdict": "met", "reason": "forecast returned", "answer": "12-week forecast ready"}]

        def decide(self, messages, schema, name):
            return (self.plans if name == "plan" else self.evals).pop(0)

    store = Store(tmp_path / "real.sqlite")
    try:
        mid = store.create_mission("forecast weekly demand for 12 weeks", {"max_tool_calls": 5})
        out = Runtime(store, owner="real").run(
            mid, goal_program("forecast weekly demand", {"timesfm_forecast": timesfm_tool()}, PlannerOnce()))
        assert out["status"] == "completed"
        evidence = out["result"]["evidence"][0]
        assert "result" in evidence, evidence
        cfg = evidence["result"]["config"]
        assert cfg["global_context"] == 15360
        assert cfg["horizon"] == 12
        assert len(evidence["result"]["point"]) == 12
        assert all(v >= 0 for v in evidence["result"]["point"])
        tool_rows = store.conn.execute("SELECT status FROM tool_calls WHERE mission_id=?", (mid,)).fetchall()
        assert [r["status"] for r in tool_rows] == ["done"]
    finally:
        store.close()

