"""Plan horizon against the requested horizon (audit item 10, research 1, 2026-11-22).

A goal that asks for "the next 20 trading days" is answered with a forecast over that span. Before this check the loop never
compared the forecast's horizon with the request, so a plan of 14 steps passed as "met". These tests pin the parse of the
request, the comparison with the forecast's own horizon, and the two visible outcomes: a block that sends the finish back,
and a note the answer must carry when the comparison cannot be made.

The scripted brain follows test_goal_loop.py. Every scenario here is offline: the tools are stubs that return the shapes the
real tools return.
"""
from __future__ import annotations

import json

import pytest

from prophet_agent.runtime import goal as goal_mod
from prophet_agent.runtime.goal import GoalConfig, goal_program
from prophet_agent.runtime.runner import Runtime, Tool
from prophet_agent.runtime.store import Store

TRADING = "Forecast the next 20 trading days of the sales series."


class ScriptedBrain:
    def __init__(self, plans=(), evals=()):
        self.plans = list(plans)
        self.evals = list(evals)
        self.calls = {"plan": 0, "eval": 0}

    def decide(self, messages, schema, name):
        self.calls[name] += 1
        queue = self.plans if name == "plan" else self.evals
        if not queue:
            raise AssertionError(f"brain asked for {name} but the script is exhausted")
        return queue.pop(0)


def call(tool, args=None, why="gather"):
    return {"action": "call_tool", "tool": tool, "args": args or {}, "rationale": why}


def finish(why="done"):
    return {"action": "finish", "rationale": why}


def met(answer="The forecast is below."):
    return {"verdict": "met", "reason": "enough", "answer": answer}


@pytest.fixture()
def store(tmp_path):
    s = Store(tmp_path / "horizon.sqlite")
    yield s
    s.close()


def _tools(step=None):
    """Stub tools. The forecast returns as many points as it was asked for, and records that horizon in its config, as
    the real timesfm_forecast does. ``step`` is the series step the forecast record carries, when it carries one."""
    def forecast(args, key):
        h = args["horizon"]
        config = {"horizon": h, "history_points": 60}
        if step is not None:
            config["series_step"] = step
        return {"point": [10.0] * h, "config": config}

    def backtest(args, key):
        return {"models": {"naive": {"mase": 1.2}, "timesfm": {"mase": 0.9}}, "best_baseline": "naive",
                "horizon": args["horizon"], "origins": 4}

    # The sandbox step stands for the scan and the ledger that a forecast needs (the contract checks read the files it names).
    sandbox = {"run_id": "run1", "outputs": ["out/scan.json", "out/ledger.json"]}
    return {"sandbox_python": Tool("sandbox_python", lambda a, k: dict(sandbox), idempotent=True),
            "backtest_series": Tool("backtest_series", backtest, idempotent=True),
            "timesfm_forecast": Tool("timesfm_forecast", forecast, idempotent=True)}


@pytest.fixture(autouse=True)
def sandbox_root(tmp_path, monkeypatch):
    """A run folder with the scan and a four-stage ledger, where the checks look for them. Each test gets its own."""
    root = tmp_path / "sandbox"
    out = root / "run1" / "out"
    out.mkdir(parents=True)
    (out / "scan.json").write_text('{"findings": []}', encoding="utf-8")
    stages = ["time index", "coverage", "zeros and censoring", "outliers and level shifts"]
    (out / "ledger.json").write_text(json.dumps([{"stage": s, "finding": "checked", "decision": "kept"} for s in stages]),
                                     encoding="utf-8")
    monkeypatch.setattr(goal_mod, "SANDBOX_ROOT", root)
    return root


def _run(store, goal, brain, step=None):
    program = goal_program(goal, _tools(step), brain, GoalConfig())
    mid = store.create_mission("forecast", {})
    return Runtime(store, owner="h1").run(mid, program)


# ---------------------------------------------------------------- regression: the mismatch is caught end to end

def test_plan_horizon_shorter_than_requested_is_not_met(store):
    """The goal asks for 20 trading days; the plan forecasts 14. Before the fix the evaluator's "met" passed unchallenged.
    After the fix the finish is refused (a block named horizon_matches_request), and after the allowed retries the
    answer leaves as partial with that check named."""
    brain = ScriptedBrain(
        plans=[call("sandbox_python"), call("backtest_series", {"horizon": 20}),
               call("timesfm_forecast", {"horizon": 14}), finish(), finish(), finish()],
        evals=[met(), met(), met()],
    )
    result = _run(store, TRADING, brain, step="daily")["result"]
    assert result["status"] == "partial", f"a 14-step plan for a 20-day request was accepted as {result['status']!r}"
    assert "horizon_matches_request" in result.get("contract_failures", [])
    assert "20" in result["reason"] and "14" in result["reason"]


def test_plan_horizon_matching_the_request_is_met(store):
    """Same goal, the plan forecasts 20 trading days of a daily series: the comparison passes and the answer is met."""
    brain = ScriptedBrain(
        plans=[call("sandbox_python"), call("backtest_series", {"horizon": 20}),
               call("timesfm_forecast", {"horizon": 20}), finish()],
        evals=[met()],
    )
    out = _run(store, TRADING, brain, step="daily")
    assert out["result"]["status"] == "met"
    assert "Horizon check" not in out["result"]["answer"], "a passing comparison needs no note"


# ---------------------------------------------------------------- parsing the request

@pytest.mark.parametrize("goal, expected", [
    ("Forecast the next 20 trading days.", (20, "trading_day")),
    ("Forecast the next 14 days of demand.", (14, "day")),
    ("Give me a forecast for the next 6 weeks.", (6, "week")),
    ("Project the next 3 months ahead.", (3, "month")),
    ("Forecast 60 days ahead.", (60, "day")),
    ("Forecast over a horizon of 12 periods.", (12, "period")),
    ("Run a 5-day horizon forecast.", (5, "day")),
    ("Forecast the next 2 years.", (2, "year")),
    ("Forecast the next 3 quarters.", (3, "quarter")),
    ("Forecast the next business days, 10 business days ahead.", (10, "trading_day")),
])
def test_requested_horizon_is_parsed_from_the_goal(goal, expected):
    assert goal_mod.requested_horizon(goal) == expected


@pytest.mark.parametrize("goal", [
    "Forecast the demand.",                                   # no horizon stated
    "Forecast the next 20 trading days and the next 4 weeks.",  # two different horizons: not unambiguous
    "The sales were 5 days late last year.",                  # a number of days, but not a forecast horizon
    "Forecast the next 0 days.",                              # not a horizon
])
def test_requested_horizon_refuses_what_it_cannot_read(goal):
    assert goal_mod.requested_horizon(goal) is None


# ---------------------------------------------------------------- the comparison itself

def _forecast_history(horizon, step=None, args_horizon=None):
    config = {"horizon": horizon}
    if step is not None:
        config["series_step"] = step
    return [{"kind": "tool", "tool": "timesfm_forecast", "args": {"horizon": args_horizon or horizon},
             "result": {"point": [1.0] * horizon, "config": config}}]


def _check(checks, name):
    matches = [c for c in checks if c["name"] == name]
    assert len(matches) == 1, f"expected one {name} check, got {checks}"
    return matches[0]


def test_period_request_must_equal_the_plan_horizon():
    ok = goal_mod.horizon_checks("Forecast the next 12 periods.", _forecast_history(12))
    bad = goal_mod.horizon_checks("Forecast the next 12 periods.", _forecast_history(9))
    assert _check(ok, "horizon_matches_request")["ok"] is True
    failed = _check(bad, "horizon_matches_request")
    assert failed["ok"] is False and failed["severity"] == "block"
    assert "12" in failed["detail"] and "9" in failed["detail"]


@pytest.mark.parametrize("goal, horizon, step, matches", [
    ("Forecast the next 6 weeks.", 6, "weekly", True),
    ("Forecast the next 6 weeks.", 42, "daily", True),         # 42 days of a daily series
    ("Forecast the next 6 weeks.", 20, "weekly", False),
    ("Forecast the next 20 trading days.", 20, "daily", True),   # a daily series is one period per trading day
    ("Forecast the next 20 trading days.", 4, "weekly", True),   # 20 trading days is 28 calendar days, 4 weeks
    ("Forecast the next 20 trading days.", 14, "daily", False),
    ("Forecast the next 3 months.", 3, "monthly", True),
    ("Forecast the next 3 months.", 4, "monthly", False),
    ("Forecast the next 14 days.", 14, "1 days 00:00:00", True),  # the profile's own text for a one-day step
    ("Forecast the next 14 days.", 14, 7, False),                # a numeric step is a number of days
])
def test_calendar_request_is_converted_with_the_series_step(goal, horizon, step, matches):
    checks = goal_mod.horizon_checks(goal, _forecast_history(horizon, step=step))
    check = _check(checks, "horizon_matches_request")
    assert check["ok"] is matches, check["detail"]
    if not matches:
        assert check["severity"] == "block"


def test_unknown_series_step_is_a_visible_warning_not_a_block():
    """Until the forecast record carries its series step, a calendar request cannot be converted. The check says so, as a
    warning, so the loop does not refuse every calendar forecast, and the answer carries the note."""
    checks = goal_mod.horizon_checks("Forecast the next 6 weeks.", _forecast_history(6))
    check = _check(checks, "horizon_matches_request")
    assert check["ok"] is False and check["severity"] == "warn"
    assert "step" in check["detail"]


def test_no_forecast_or_no_request_leaves_the_check_out():
    assert goal_mod.horizon_checks("Forecast the next 6 weeks.", []) == []
    assert goal_mod.horizon_checks("Forecast the demand.", _forecast_history(6)) == []


def test_warning_reaches_the_answer(store):
    """A calendar request with no recorded step: the answer leaves met, and carries the computed note the person reads."""
    brain = ScriptedBrain(
        plans=[call("sandbox_python"), call("backtest_series", {"horizon": 14}),
               call("timesfm_forecast", {"horizon": 14}), finish()],
        evals=[met("The forecast is below.")],
    )
    out = _run(store, "Forecast the next 14 days of the sales series.", brain)
    result = out["result"]
    assert result["status"] == "met"
    assert "Horizon check (computed)" in result["answer"]
    assert "horizon_matches_request" in result["answer"]
