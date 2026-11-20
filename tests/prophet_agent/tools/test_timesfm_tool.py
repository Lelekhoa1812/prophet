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
from prophet_agent.tools import LICENCE_NOTE, timesfm_tool


class _StubEngine:
    checkpoint = "stub-checkpoint"
    global_context = 16

    def forecast(self, series, horizon):
        point = np.full(horizon, float(series[-1]))
        q = np.tile(np.linspace(0, 8, 9), (horizon, 1))
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
    assert "licence" not in out  # audit wording belongs in NOTICE.md, not in results the planner repeats


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
