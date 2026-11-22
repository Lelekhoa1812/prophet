"""Global gradient-boosted panel model: causal features, no leakage across origins, shapes, and a speed budget."""
import time

import numpy as np
import pytest

from prophet_agent.forecast import global_gbm as gg
from prophet_agent.forecast.types import Forecast

Z = np.array([-1.2816, -0.8416, -0.5244, -0.2533, 0.0, 0.2533, 0.5244, 0.8416, 1.2816])


class Engine:
    """Stands in for TimesFM: forecasts the last value with a normal band. It has no forecast_batch, so the per-series path runs."""
    def __init__(self, sd=1.0):
        self.sd = sd

    def forecast(self, series, horizon):
        last = float(series[-1])
        q = last + self.sd * Z[None, :] * np.ones((horizon, 1))
        return Forecast(np.full(horizon, last, dtype=np.float32), q.astype(np.float32))


def make_panel(entities=8, n=400, seed=0):
    """Eight entities on one calendar: a shared weekly cycle and trend, each with its own level (5 to 40 times)."""
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    shape = 100 + 10 * np.sin(2 * np.pi * t / 7) + 0.05 * t
    return {f"sku{i}": (5 * (i + 1)) * (shape + rng.normal(0, 3, n)) for i in range(entities)}


def test_features_at_an_origin_do_not_depend_on_later_values():
    """Row o of the feature matrix (and the scale) is built from x[:o] only, so changing x from index 120 on must leave
    rows 0 to 120 exactly as they were."""
    rng = np.random.default_rng(2)
    x = 50 + np.cumsum(rng.normal(0, 1, 200))
    y = x.copy()
    y[120:] += 1000.0
    f1, s1 = gg._feature_matrix(x, 14, [7, 14])
    f2, s2 = gg._feature_matrix(y, 14, [7, 14])
    np.testing.assert_array_equal(f1[:121], f2[:121])
    np.testing.assert_array_equal(s1[:121], s2[:121])
    assert not np.array_equal(f1[121:], f2[121:]), "later origins do see the changed values, as they should"


def test_each_origin_model_is_refitted_on_history_before_that_origin_only(monkeypatch):
    """Instrument the global model: at each origin it must receive exactly the prefix of every series before the origin."""
    panel = make_panel()
    seen = []
    real = gg.global_forecast

    def spy(hist, horizon, season=None, **kw):
        seen.append({k: np.asarray(v).copy() for k, v in hist.items()})
        return real(hist, horizon, season=season, **kw)

    monkeypatch.setattr(gg, "global_forecast", spy)
    gg.panel_backtest(panel, 7, n_origins=3, season=7)
    n = 400
    cuts = [n - (3 - k) * 7 for k in range(3)]
    assert len(seen) == 3
    for hist, cut in zip(seen, cuts):
        assert set(hist) == set(panel)
        for name, values in panel.items():
            assert len(hist[name]) == cut, "the training data must stop at the origin"
            np.testing.assert_array_equal(hist[name], np.asarray(values)[:cut])


def test_the_global_model_beats_naive_on_a_seasonal_panel():
    out = gg.panel_backtest(make_panel(), 7, n_origins=3, season=7)
    assert out["overall"]["global_gbm"] < out["overall"]["naive"]
    assert out["overall"]["global_gbm"] < out["overall"]["seasonal_naive"]
    assert out["best_overall"] == "global_gbm"
    assert out["wins"]["global_gbm"] >= 6


def test_global_forecast_returns_one_band_per_entity_with_the_right_shape():
    panel = make_panel(entities=3, n=120)
    out = gg.global_forecast(panel, 5, season=7)
    assert list(out) == list(panel)
    for name, band in out.items():
        assert set(band) == {"point", "q10", "q90"}
        assert len(band["point"]) == len(band["q10"]) == len(band["q90"]) == 5
        lo, mid, hi = (np.asarray(band[k]) for k in ("q10", "point", "q90"))
        assert (lo <= mid + 1e-9).all() and (mid <= hi + 1e-9).all(), "quantiles must not cross"
        assert (np.asarray(band["point"]) >= 0).all(), "a non-negative history gives a non-negative forecast"
    # the scale of each entity is carried through: the biggest entity forecasts a bigger level
    assert np.mean(out["sku2"]["point"]) > np.mean(out["sku0"]["point"]) * 2


def test_panel_backtest_keys_and_counts_without_an_engine():
    out = gg.panel_backtest(make_panel(), 6, n_origins=2, season=7)
    assert set(out) == {"per_entity", "overall", "wins", "best_overall", "coverage_80", "origins", "horizon", "season", "notes"}
    assert set(out["overall"]) == {"naive", "seasonal_naive", "global_gbm"}
    assert sum(out["wins"].values()) == 8 and set(out["per_entity"]) == set(make_panel())
    assert set(out["coverage_80"]) == {"global_gbm"} and 0 <= out["coverage_80"]["global_gbm"] <= 1
    assert out["origins"] == 2 and out["horizon"] == 6 and out["season"] == 7
    assert any("TimesFM not run" in note for note in out["notes"])


def test_panel_backtest_with_an_engine_adds_timesfm_and_the_blend():
    out = gg.panel_backtest(make_panel(), 6, n_origins=2, season=7, engine=Engine(sd=1.0))
    assert set(out["overall"]) == {"naive", "seasonal_naive", "global_gbm", "timesfm", "blend"}
    assert set(out["coverage_80"]) == {"global_gbm", "timesfm"}
    assert set(out["per_entity"]["sku0"]) == set(out["overall"])
    assert sum(out["wins"].values()) == 8
    assert out["overall"]["blend"] == pytest.approx(
        np.mean([(out["per_entity"][k]["blend"]) for k in out["per_entity"]]), abs=1e-4)


def test_mase_is_scaled_by_each_entitys_own_history():
    """Scaling one entity's series by a constant must not change its MASE, because MASE divides by the same scale."""
    panel = make_panel(entities=2, n=200)
    base = gg.panel_backtest(panel, 5, n_origins=2, season=7)["per_entity"]["sku0"]["naive"]
    scaled = dict(panel)
    scaled["sku0"] = panel["sku0"] * 1000.0
    again = gg.panel_backtest(scaled, 5, n_origins=2, season=7)["per_entity"]["sku0"]["naive"]
    assert again == pytest.approx(base, rel=1e-3)


def test_bad_inputs_are_refused_with_the_reason():
    panel = make_panel(entities=2, n=60)
    with pytest.raises(ValueError, match="at least 2"):
        gg.panel_backtest({"a": panel["sku0"]}, 3)
    with pytest.raises(ValueError, match="n_origins"):
        gg.panel_backtest(panel, 3, n_origins=0)
    with pytest.raises(ValueError, match="origins"):
        gg.panel_backtest(panel, 40, n_origins=6)
    with pytest.raises(ValueError, match="finite"):
        gg.global_forecast({"a": [1.0, float("nan")] * 20}, 3)
    with pytest.raises(ValueError, match="quantiles"):
        gg.global_forecast(panel, 3, quantiles=(0.5, 0.1, 0.9))
    with pytest.raises(ValueError, match="season"):
        gg.global_forecast(panel, 3, season=1)


def test_eight_entities_by_400_points_run_in_under_ten_seconds():
    panel = make_panel(entities=8, n=400)
    start = time.perf_counter()
    gg.panel_backtest(panel, 12, n_origins=3, season=7, engine=Engine(sd=1.0))
    elapsed = time.perf_counter() - start
    assert elapsed < 10.0, f"took {elapsed:.1f} s"
