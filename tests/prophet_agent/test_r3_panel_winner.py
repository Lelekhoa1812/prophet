"""A2-07: a panel's best model is named only with a paired test against the runner-up and against TimesFM. The test is an
exact sign-flip on per-entity MASE differences (enumerated for few entities, Monte Carlo for many)."""
import numpy as np
import pytest

from prophet_agent.forecast import global_gbm as gg
from prophet_agent.tools import panel_tool as pt


def test_two_entities_half_a_percent_apart_are_no_clear_winner():
    """Regression: two entities whose MASE differs by 0.5 percent must not produce a named winner. Before the fix,
    panel_backtest named the lowest mean with no test at all (A2-07, measured on the electricity panel)."""
    per_entity = {"e1": {"naive": 1.005, "global_gbm": 1.000}, "e2": {"naive": 1.005, "global_gbm": 1.000}}
    res = gg.winner_test(per_entity, ["naive", "global_gbm"])
    assert res["verdict"] == "no clear winner"
    assert res["against"]["naive"]["verdict"] == "no_clear_difference"
    assert res["against"]["naive"]["p_value"] == pytest.approx(0.5)
    assert res["against"]["naive"]["method"] == "exact"


def test_a_consistent_lead_over_many_entities_is_a_clear_winner():
    rng = np.random.default_rng(4)
    per_entity = {f"e{i}": {"timesfm": 1.0 + rng.normal(0, 0.1), "naive": 0.0} for i in range(60)}
    for e in per_entity:
        per_entity[e]["naive"] = per_entity[e]["timesfm"] + 0.05 + rng.normal(0, 0.005)
    res = gg.winner_test(per_entity, ["timesfm", "naive"])
    assert res["best"] == "timesfm" and res["verdict"] == "clear winner: timesfm"
    assert res["against"]["naive"]["p_value"] < 0.05
    assert res["against"]["naive"]["method"] == "monte_carlo" and res["against"]["naive"]["entities"] == 60


def test_the_best_is_compared_with_timesfm_too_when_it_is_not_timesfm():
    rng = np.random.default_rng(9)
    per_entity = {f"e{i}": {"naive": 1.0 + rng.normal(0, 0.02), "timesfm": 1.2 + rng.normal(0, 0.02),
                            "global_gbm": 1.1 + rng.normal(0, 0.02)} for i in range(12)}
    res = gg.winner_test(per_entity, ["naive", "timesfm", "global_gbm"])
    assert res["best"] == "naive" and res["runner_up"] == "global_gbm"
    assert set(res["against"]) == {"global_gbm", "timesfm"}
    assert res["against"]["timesfm"]["verdict"] == "naive_better"
    assert res["verdict"] == "clear winner: naive"


def test_the_monte_carlo_p_value_is_reproducible_and_exact_when_few_entities():
    diffs = np.array([0.3, -0.1, 0.2, 0.4, 0.05, 0.25])
    a = gg.sign_flip_test(diffs, seed=1)
    b = gg.sign_flip_test(diffs, seed=1)
    assert a == b and a["method"] == "exact" and a["draws"] == 2 ** 6
    many = np.concatenate([diffs] * 10)
    mc = gg.sign_flip_test(many, seed=1, draws=2000)
    assert mc["method"] == "monte_carlo" and mc["draws"] == 2000
    assert gg.sign_flip_test(many, seed=1, draws=2000) == mc


def test_the_panel_backtest_keeps_its_fields_and_the_extended_view_adds_the_test():
    panel = {f"s{i}": (5 + i) * (100 + 10 * np.sin(2 * np.pi * np.arange(120) / 7) + np.arange(120) * 0.1)
             for i in range(4)}
    plain = gg.panel_backtest(panel, 5, n_origins=3, season=7)
    assert "best_overall_test" not in plain
    full = gg.panel_backtest(panel, 5, n_origins=3, season=7, extended=True)
    assert set(plain) <= set(full)
    test = full["best_overall_test"]
    assert test["best"] == full["best_overall"] and test["alpha"] == 0.05
    assert test["verdict"] in ("no clear winner",) or test["verdict"].startswith("clear winner: ")


def test_reading_says_no_clear_winner_when_the_lead_is_not_significant():
    res = {"overall": {"naive": 1.7, "global_gbm": 1.8}, "wins": {"naive": 3, "global_gbm": 1},
           "best_overall": "naive", "per_entity": {"a": {}, "b": {}, "c": {}, "d": {}}, "origins": 6, "horizon": 7,
           "coverage_80": {}, "best_overall_test": {"verdict": "no clear winner", "best": "naive", "alpha": 0.05,
                                                     "against": {"global_gbm": {"p_value": 0.5}}}}
    assert "no clear winner" in pt.reading(res)
