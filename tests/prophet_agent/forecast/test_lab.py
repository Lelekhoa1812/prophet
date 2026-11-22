"""Control lab: decisions, causality, budget, failures and the tool surface. The engine is a fake whose behaviour is set by
the test, so every number here is a property of the fake, not a measurement of TimesFM."""
from __future__ import annotations

import json

import numpy as np
import pytest

from prophet_agent.forecast import lab as L
from prophet_agent.forecast.control import Control
from prophet_agent.forecast.types import Forecast
from prophet_agent.tools import lab_tool as T

H = 4
N = 120


class MeanEngine:
    """Stands in for TimesFM 3. The forecast is the mean of the last 8 usable points of the history it is given, so a
    treatment that changes the usable points changes the forecast. Records every call it receives."""

    WINDOW = 8

    def __init__(self, respect_ignore=True, znorm_bias=0.0, fail_on_znorm=False, fail_on=None):
        self.respect_ignore = respect_ignore
        self.znorm_bias = znorm_bias
        self.fail_on_znorm = fail_on_znorm
        self.fail_on = fail_on
        self.calls = []

    def forecast_controlled(self, series, horizon, control):
        series = np.asarray(series, dtype=float)
        self.calls.append((series.copy(), control))
        if self.fail_on_znorm and control.znorm:
            raise RuntimeError("model refused znorm")
        if self.fail_on is not None and self.fail_on(series, control):
            raise RuntimeError("model failed on this fold")
        usable = ~control.ignore_mask(len(series)) if self.respect_ignore else np.ones(len(series), dtype=bool)
        level = float(series[usable][-self.WINDOW:].mean())
        if control.znorm:
            level += self.znorm_bias
        point = np.full(horizon, level)
        q = point[:, None] + np.linspace(-1, 1, 9)[None, :] * 5.0
        return Forecast(point, q)


def noise(n=N, seed=1):
    return 100.0 + np.random.default_rng(seed).normal(0.0, 1.0, n)


def spiky(n=N, h=H, folds=8, seed=3):
    """Noise around 100 with a spike of 500 on the last point before every fold origin, so the default model reads each
    spike in the last window it averages."""
    x = noise(n, seed)
    for o in L.plan_folds(n, h, folds)[0]:
        x[o - 1] = 500.0
    return x


def log_of(engine):
    return [(s.tolist(), c.describe(), {k: v.tolist() for k, v in c.past.items()},
             {k: v.tolist() for k, v in c.future.items()}) for s, c in engine.calls]


# ------------------------------------------------------------------------------------- folds and refusals --

def test_folds_are_time_ordered_with_one_horizon_gap_and_end_at_the_series_end():
    origins, min_hist, fit = L.plan_folds(N, H, 8)
    assert origins[-1] == N - H
    assert np.all(np.diff(origins) == 2 * H)          # one test window plus one horizon of gap between origins
    assert origins[0] >= min_hist and fit >= len(origins)


def test_history_too_short_is_refused_in_plain_words():
    with pytest.raises(ValueError, match=r"the series has 30 points; the lab needs at least 60 for 6 walk-forward folds"):
        L.run_lab(noise(30), H, MeanEngine())


def test_folds_below_the_minimum_are_refused():
    with pytest.raises(ValueError, match="folds must be a whole number from 6 to 30"):
        L.run_lab(noise(), H, MeanEngine(), folds=3)


def test_flat_history_folds_are_dropped_and_reported():
    x = np.concatenate([np.full(16, 5.0), noise(52, seed=4)])   # n = 68: seven origins, the first has a flat history
    out = L.run_lab(x, H, MeanEngine())
    assert out["fold_count"] == 6 and 16 not in out["folds"]
    assert any("flat" in c for c in out["caveats"])


def test_flat_history_leaving_too_few_folds_is_refused():
    x = np.concatenate([np.full(16, 5.0), noise(44, seed=4)])   # n = 60: six origins, one flat, five left
    with pytest.raises(ValueError, match="only 5 of the 6 folds have history that varies"):
        L.run_lab(x, H, MeanEngine())


# ------------------------------------------------------------------------------------------ outlier logic --

def test_outlier_flags_find_an_injected_spike_and_not_clean_noise():
    x = noise(80)
    x[50] = 500.0
    flags = L.robust_flags(x)
    assert flags[50] and flags.sum() == 1
    assert not L.robust_flags(noise(80, seed=9)).any()


def test_carry_forward_reads_only_earlier_points():
    hist = np.array([1.0, 2.0, 99.0, 4.0, 99.0, 6.0])
    flags = np.array([False, False, True, False, True, False])
    out, replaced = L._carry_forward(hist, flags)
    assert list(out) == [1.0, 2.0, 2.0, 4.0, 4.0, 6.0] and replaced == 2
    changed = hist.copy()
    changed[5] = -1000.0                              # a later value must not alter an earlier replacement
    out2, _ = L._carry_forward(changed, flags)
    assert np.array_equal(out2[:5], out[:5])


def test_scale_effect_rule_needs_zero_to_win_on_mase_and_lose_on_pinball():
    assert L.is_scale_effect(zero_mase=0.6, default_mase=1.0, zero_pinball=2.0, default_pinball=1.0)
    assert not L.is_scale_effect(zero_mase=0.6, default_mase=1.0, zero_pinball=0.5, default_pinball=1.0)
    assert not L.is_scale_effect(zero_mase=1.2, default_mase=1.0, zero_pinball=2.0, default_pinball=1.0)


# ---------------------------------------------------------------------------------------- decision rules --

def _pair(gain, wins, lo, hi, p=0.03):
    return {"gain": gain, "wins": wins, "lo": lo, "hi": hi, "p": p}


def test_adopt_needs_gain_wins_interval_and_pinball_together():
    n_folds = 8
    assert L._decide(_pair(0.10, 8, 0.02, 0.2), 0.05, n_folds)[0] == "adopt"
    assert L._decide(_pair(L.MIN_GAIN - 0.001, 8, 0.0, 0.2), 0.05, n_folds)[0] == "no evidence"     # gain just short
    assert L._decide(_pair(0.10, 4, 0.02, 0.2), 0.05, n_folds)[0] == "no evidence"                  # 4/8 < 60%
    assert L._decide(_pair(0.10, 8, 0.0, 0.2), 0.05, n_folds)[0] == "no evidence"                   # interval has zero
    assert L._decide(_pair(0.10, 8, 0.02, 0.2), -0.01, n_folds)[0] == "no evidence"                 # pinball worse
    assert L._decide(_pair(0.10, 8, 0.02, 0.2), None, n_folds)[0] == "no evidence"                  # pinball unknown


def test_pinball_tolerance_is_the_only_pinball_threshold():
    assert L.PINBALL_TOLERANCE == 0.0
    assert L._decide(_pair(0.10, 8, 0.02, 0.2), 0.0, 8)[0] == "adopt"      # equal pinball is not worse


def test_harmful_only_when_the_mase_interval_lies_entirely_below_zero():
    assert L._decide(_pair(-0.2, 0, -0.3, -0.1), -0.2, 8)[0] == "harmful"
    assert L._decide(_pair(-0.2, 1, -0.3, 0.05), -0.2, 8)[0] == "no evidence"


def test_wilcoxon_is_skipped_below_the_minimum_folds_and_when_all_differences_are_zero():
    assert L._wilcoxon(np.array([0.1, 0.2, 0.3, 0.4, 0.5])) is None           # five folds
    assert L._wilcoxon(np.zeros(8)) is None
    p = L._wilcoxon(np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, -0.05, 0.7]))
    assert p is not None and 0.0 < p <= 1.0


# ------------------------------------------------------------------------------------ the decisions run --

def test_adopts_a_treatment_that_removes_injected_outliers():
    x = spiky()
    out = L.run_lab(x, H, MeanEngine(), folds=8)
    assert out["decisions"]["ignore_outliers"]["decision"] == "adopt"
    assert out["decisions"]["replace_outliers"]["decision"] == "adopt"
    assert out["recommended"] != "keep the default"
    assert out["recommended"]["candidate"] in {"ignore_outliers", "replace_outliers"}
    row = next(r for r in out["table"] if r["name"] == "ignore_outliers")
    assert row["wins"] == out["fold_count"] and row["pinball_gain_pct"] > 0


def test_keeps_the_default_when_nothing_matters():
    out = L.run_lab(noise(), H, MeanEngine(respect_ignore=False), folds=8)
    assert out["recommended"] == "keep the default"
    assert all(d["decision"] != "adopt" for d in out["decisions"].values())
    assert out["decisions"]["znorm"]["decision"] == "no evidence"


def test_marks_a_treatment_that_hurts_as_harmful():
    out = L.run_lab(noise(), H, MeanEngine(znorm_bias=30.0), folds=8)
    assert out["decisions"]["znorm"]["decision"] == "harmful"
    assert out["recommended"] == "keep the default"


def test_a_failing_treatment_is_excluded_and_does_not_stop_the_run():
    out = L.run_lab(spiky(), H, MeanEngine(fail_on_znorm=True), folds=8)
    assert [f["candidate"] for f in out["failed"]] == ["znorm"]
    assert "znorm" not in out["decisions"]
    assert out["recommended"]["candidate"] != "znorm"


def test_a_failing_reference_is_fatal_because_nothing_can_be_compared_with_it():
    with pytest.raises(RuntimeError, match="no reference to compare with"):
        L.run_lab(noise(), H, MeanEngine(fail_on=lambda s, c: c.is_default), folds=8)


def test_identical_treatments_cost_no_calls_and_are_reported():
    eng = MeanEngine(respect_ignore=False)
    out = L.run_lab(noise(), H, eng, candidates=["ignore_outliers"], folds=8)
    assert any(s["candidate"] == "ignore_outliers" and "identical" in s["reason"] for s in out["skipped"])
    assert out["calls_spent"] == out["fold_count"]              # only the reference was called


def test_scale_effect_is_reported_on_the_result():
    out = L.run_lab(noise(), H, MeanEngine(), folds=8)
    assert set(out["reference"]) >= {"zero_forecast_mase", "scale_effect", "best_simple_baseline"}
    assert out["reference"]["best_simple_baseline"] != "zero"


# ----------------------------------------------------------------------------------- budget and calls --

def test_budget_cut_is_reported_and_never_exceeded():
    out = L.run_lab(spiky(), H, MeanEngine(), folds=8, max_calls=12)
    assert out["calls_spent"] == 8 <= out["max_calls"] == 12
    assert out["skipped"] and all("budget" in s["reason"] for s in out["skipped"])
    assert any("were not run" in c for c in out["caveats"])
    assert any(r["name"] == "default" for r in out["table"])


def test_a_budget_below_the_reference_is_refused():
    with pytest.raises(ValueError, match="cannot cover the 8 folds the reference needs"):
        L.run_lab(spiky(), H, MeanEngine(), folds=8, max_calls=5)


def test_every_engine_call_sees_only_history_before_its_origin():
    eng = MeanEngine()
    out = L.run_lab(spiky(), H, eng, folds=8, period=4)
    origins = set(out["folds"])
    assert eng.calls
    for series, control in eng.calls:
        assert len(series) in origins
        assert all(e <= len(series) for _, e in control.ignore)
        assert all(len(v) <= len(series) + H for v in control.future.values())
        assert all(len(v) <= len(series) for v in control.past.values())


# ------------------------------------------------------------------------------------------- causality --

def _covariates(n=N, h=H, seed=7):
    rng = np.random.default_rng(seed)
    return {"p": rng.normal(size=n)}, {"f": rng.normal(size=n + h)}


def test_every_default_grid_candidate_is_unchanged_when_the_future_is_poisoned_after_each_origin():
    x = spiky()
    past, future = _covariates()
    grid = L._grid(period=4, has_covariates=True)
    origins = L.plan_folds(N, H, 8, 4)[0]
    checked = 0
    for o in origins:
        xp = x.copy()
        xp[o:] = np.nan                                           # the values after the origin, scored later
        pp = {k: np.where(np.arange(len(v)) >= o, np.nan, v) for k, v in past.items()}
        fp = {k: np.where(np.arange(len(v)) >= o + H, np.nan, v) for k, v in future.items()}
        clean = L._fold_context(x, past, future, o, H, 4)
        poisoned = L._fold_context(xp, pp, fp, o, H, 4)
        for cand in grid:
            a, b = cand.prepare(clean), cand.prepare(poisoned)
            assert np.array_equal(a.history, b.history), cand.name
            assert np.isfinite(b.history).all(), cand.name
            assert a.control.describe() == b.control.describe(), cand.name
            assert a.control.is_default == b.control.is_default, cand.name
            for k in a.control.past:
                assert np.array_equal(a.control.past[k], b.control.past[k]), cand.name
            for k in a.control.future:
                assert np.array_equal(a.control.future[k], b.control.future[k]), cand.name
            checked += 1
    assert checked == len(origins) * len(grid)


def test_reference_baselines_are_unchanged_when_the_future_is_poisoned():
    x = spiky()
    for o in L.plan_folds(N, H, 8, 4)[0]:
        xp = x.copy()
        xp[o:] = np.nan
        for name in ("naive", "mean_last_8", "seasonal_naive", "tsb", "zero"):
            a = L._baseline(name, x[:o].copy(), H, 4)
            b = L._baseline(name, xp[:o].copy(), H, 4)
            assert np.array_equal(a.point, b.point) and np.array_equal(a.q, b.q), name


def test_the_whole_run_gives_the_engine_the_same_inputs_whatever_the_tail_holds():
    x = spiky()
    past, future = _covariates()
    xp = x.copy()
    xp[N - H:] = 1e6                                             # the last test window only
    pp = {"p": past["p"].copy()}
    pp["p"][N - H:] = 1e6
    fp = {"f": future["f"].copy()}
    fp["f"][N:] = 1e6                                            # the covariate values for the final horizon only
    e1, e2 = MeanEngine(), MeanEngine()
    L.run_lab(x, H, e1, folds=8, past_covariates=past, future_covariates=future)
    L.run_lab(xp, H, e2, folds=8, past_covariates=pp, future_covariates=fp)
    assert log_of(e1) == log_of(e2)


def test_replace_outliers_changes_the_input_and_leaves_the_control_default():
    eng = MeanEngine()
    L.run_lab(spiky(), H, eng, candidates=["replace_outliers"], folds=8)
    replaced = [s for s, c in eng.calls if c.is_default and s.max() < 200]
    assert replaced, "the replaced history should reach the engine with the spikes gone"


# ---------------------------------------------------------------------------- determinism and reporting --

def test_the_same_seed_gives_the_same_result():
    a = L.run_lab(spiky(), H, MeanEngine(), folds=8, seed=11)
    b = L.run_lab(spiky(), H, MeanEngine(), folds=8, seed=11)
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


def test_table_is_sorted_by_gain_and_has_every_reported_column():
    out = L.run_lab(spiky(), H, MeanEngine(), folds=8, period=4, past_covariates=_covariates()[0],
                    future_covariates=_covariates()[1])
    gains = [r["gain_pct"] for r in out["table"] if r["gain_pct"] is not None]
    assert gains == sorted(gains, reverse=True)
    names = {r["name"] for r in out["table"]}
    assert {"default", "zero", "seasonal_naive", "naive", "mean_last_8"} <= names
    for r in out["table"]:
        for key in ("mean_mase", "mean_rmsse", "mean_pinball", "pinball_gain_pct", "wilcoxon_p", "decision", "kind"):
            assert key in r
    assert out["reference"]["best_simple_baseline"] != "zero"


def test_seasonal_window_and_covariate_candidates_only_appear_when_they_apply():
    names_plain = [c.name for c in L._grid(None, False)]
    assert "ctx_season" not in names_plain and "with_covariates" not in names_plain
    assert "ctx_season" in [c.name for c in L._grid(4, False)]
    assert "with_covariates" in [c.name for c in L._grid(None, True)]
    with pytest.raises(ValueError, match="with_covariates: it needs past_covariates or future_covariates"):
        L.run_lab(spiky(), H, MeanEngine(), candidates=["with_covariates"], folds=8)
    with pytest.raises(ValueError, match="ctx_season: it needs period"):
        L.run_lab(spiky(), H, MeanEngine(), candidates=["ctx_season"], folds=8)


def test_unknown_candidate_names_are_listed_in_the_refusal():
    with pytest.raises(ValueError, match="unknown candidates \\['nope'\\]; choose from"):
        L.run_lab(spiky(), H, MeanEngine(), candidates=["nope"], folds=8)


def test_requested_candidates_run_with_the_reference_only():
    out = L.run_lab(spiky(), H, MeanEngine(), candidates=["znorm"], folds=8)
    assert set(out["decisions"]) == {"znorm"}


# ------------------------------------------------------------------------------------------------ tool --

def test_the_tool_spec_has_the_same_shape_as_the_forecast_spec():
    spec = T.LAB_TOOL_SPEC
    assert spec["name"] == "control_lab"
    assert {"title", "activity", "description", "args"} <= set(spec)
    assert {"values", "source", "horizon", "period", "candidates", "past_covariates", "future_covariates"} == set(spec["args"])


def test_the_tool_is_idempotent_and_its_observe_is_short():
    tool = T.lab_tool(lambda: MeanEngine())
    assert tool.name == "control_lab" and tool.idempotent is True
    result = tool.fn({"values": spiky().tolist(), "horizon": H, "candidates": ["ignore_outliers", "znorm"]}, "k")
    seen = tool.observe(result)
    assert len(seen["table"]) <= 25
    assert len(seen["table"]) == len(result["table"])
    assert seen["recommended"] == result["recommended"]


@pytest.mark.parametrize("args, message", [
    ({"values": [1.0] * 60, "horizon": H, "model": "x"}, "unknown arguments \\['model'\\]"),
    ({"values": [1.0] * 60, "horizon": 0}, "horizon must be an integer from 1 to 512"),
    ({"values": [1.0] * 60, "horizon": True}, "horizon must be an integer"),
    ({"values": [1.0] * 60, "horizon": H, "period": 0}, "period must be a whole number from 1 to 1000"),
    ({"values": [1.0] * 60, "horizon": H, "candidates": "znorm"}, "candidates must be a list of names"),
    ({"values": [1.0] * 60, "horizon": H, "candidates": ["nope"]}, "unknown candidates"),
    ({"values": [1.0] * 60, "horizon": H, "past_covariates": {"p": [1.0, 2.0]}}, "needs exactly 60"),
    ({"values": [1.0] * 60, "horizon": H, "future_covariates": {"f": [1.0] * 60}}, "PLUS the horizon"),
    ({"values": [1.0] * 60, "source": {"run_id": "x"}, "horizon": H}, "give exactly one of values or source"),
    ({"source": {"run_id": "x"}, "horizon": H}, "source needs run_id"),
])
def test_argument_errors_are_readable(args, message):
    tool = T.lab_tool(lambda: MeanEngine())
    with pytest.raises(ValueError, match=message):
        tool.fn(args, "k")


def test_the_model_is_not_loaded_when_the_experiment_is_refused():
    def no_model():
        raise AssertionError("the model must not load for a refused experiment")
    tool = T.lab_tool(no_model)
    with pytest.raises(ValueError, match="needs at least"):
        tool.fn({"values": noise(30).tolist(), "horizon": H}, "k")


# ------------------------------------------------------------------------- the baseline gate and multiplicity --
class LastValueEngine:
    """Repeats the last value with a fixed band. On white noise a mean of the last eight points beats it, so the default
    is the worse reference here and the lab has to say so. Behaviour is set by the test, not measured on TimesFM."""

    def forecast_controlled(self, series, horizon, control):
        series = np.asarray(series, dtype=float)
        point = np.full(horizon, float(series[-1]))
        return Forecast(point, point[:, None] + np.linspace(-1, 1, 9)[None, :] * 5.0)


def test_a_simple_baseline_that_beats_the_default_is_named_and_the_recommendation_says_so():
    out = L.run_lab(noise(), H, LastValueEngine(), folds=8)
    bb = out["baseline_beats_default"]
    assert bb is not None and bb["name"] == "mean_last_8" and bb["gain"] > L.BASELINE_GATE_PCT
    assert out["recommended"] == L.BASELINE_NOTE
    assert "a simple baseline beats default TimesFM on these folds; consider it or a combination" == out["recommended"]


def test_no_baseline_gate_when_the_default_is_as_good_as_the_best_rule():
    out = L.run_lab(noise(), H, MeanEngine(respect_ignore=False), folds=8)
    assert out["baseline_beats_default"] is None
    assert out["recommended"] == "keep the default"


def test_holm_step_down_adjusts_the_smallest_p_by_the_family_size_and_never_decreases():
    adj = L._holm({"a": 0.01, "b": 0.04, "c": 0.03, "d": None})
    assert adj["a"] == pytest.approx(0.03)          # 3 * 0.01
    assert adj["c"] == pytest.approx(0.06)          # 2 * 0.03, then carried up to the running maximum
    assert adj["b"] == pytest.approx(0.06)          # 1 * 0.04 is below the running maximum
    assert "d" not in adj                           # no p-value, so not in the family
    assert L._holm({"x": 0.9, "y": 0.9})["x"] == pytest.approx(1.0)


def test_an_adoption_must_survive_holm_correction_across_the_treatments_tested():
    out = L.run_lab(spiky(), H, MeanEngine(), folds=8)
    adopted = [nm for nm, d in out["decisions"].items() if d["decision"] == "adopt"]
    assert adopted, "the injected spikes should still be adopted after correction"
    for nm in adopted:
        assert out["decisions"][nm]["holm_p"] is not None and out["decisions"][nm]["holm_p"] <= L.HOLM_ALPHA
