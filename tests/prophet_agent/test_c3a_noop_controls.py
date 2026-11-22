"""C3a, A6-08 (contract side): a no-op setting is not a treatment, and a search over many configurations is reported.

Live evidence (measured, audit round 3, A6): retail rep4 used context_points=504 on a 504-point history. configured_vs_default_pct
was 0.0, the treatment changed nothing, and the contract still counted it as tested. Separately, a configured backtest is scored
on the same windows it was chosen on; with several configurations tried, that selection is not reported anywhere.
"""
from prophet_agent.runtime import contracts


def _args(context=None, n=504):
    args = {"values": [1.0] * n, "horizon": 14}
    if context is not None:
        args["context_points"] = context
    return args


def _record(n=504):
    return {"config": {"horizon": 14, "history_points": n}}


def _check(checks, name):
    found = [c for c in checks if c["name"] == name]
    assert len(found) == 1, checks
    return found[0]


def test_a_context_window_that_covers_the_whole_history_is_not_a_treatment():
    assert contracts.used_controls({"context_points": 504}, 504) == []
    assert contracts.used_controls(_args(context=504)) == []          # the length is read from the values


def test_a_shorter_context_window_is_still_a_treatment():
    assert contracts.used_controls({"context_points": 200}, 504) == ["context_points"]


def test_a_forecast_whose_only_setting_is_a_no_op_needs_no_lab():
    assert contracts.check_controls([], _args(context=504), _record()) is None


def test_a_forecast_with_a_real_context_window_still_needs_a_lab():
    checks = contracts.check_evidence([{"kind": "tool", "tool": "timesfm_forecast", "args": _args(context=200),
                                        "result": {"point": [1.0] * 14, "config": {"horizon": 14, "history_points": 504},
                                                   "health": {"verdict": "ok", "period": 7}}}])
    controls = _check(checks, "controls_tested")
    assert controls["ok"] is False


def _backtest(context, ignore=None):
    args = {"horizon": 14, "context_points": context}
    if ignore:
        args["ignore_ranges"] = ignore
    return {"kind": "tool", "tool": "backtest_series", "args": args,
            "result": {"ok": True, "horizon": 14, "models": {}, "configured_vs_default_pct": 1.0}}


def test_more_than_three_configurations_tried_on_one_series_is_reported_as_an_optimistic_selection():
    history = [_backtest(c) for c in (100, 200, 300, 400)]
    assert _check(contracts.check_evidence(history), "controls_selection")["ok"] is False


def test_three_configurations_or_fewer_are_not_reported():
    history = [_backtest(c) for c in (100, 200, 300)]
    assert _check(contracts.check_evidence(history), "controls_selection")["ok"] is True


def test_repeating_the_same_configuration_is_not_counted_as_a_new_one():
    history = [_backtest(100), _backtest(100), _backtest(200), _backtest(200), _backtest(300)]
    assert _check(contracts.check_evidence(history), "controls_selection")["ok"] is True
