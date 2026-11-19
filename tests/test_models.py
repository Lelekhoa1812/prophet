import numpy as np

from lab.engine.forecast import Forecast
from lab.engine.models import TSB, Baseline, Ensemble, Router, fit_routing, series_class


class Const:
    licence = "own"; min_history = 4

    def __init__(self, name, value):
        self.name, self.value = name, value

    def predict(self, contexts, horizon, season=52):
        q = np.tile(np.linspace(self.value * 0.8, self.value * 1.2, 9), (len(contexts), horizon, 1))
        return Forecast(q[:, :, 4].copy(), q)


def test_classes():
    assert series_class(np.ones(10)) == "short"
    assert series_class(np.ones(60)) == "steady"
    assert series_class(np.array([0, 0, 5] * 30)) == "intermittent"


def test_tsb_shapes_sorted_nonneg_and_lumpy_median_low():
    lumpy = np.array([0, 0, 0, 40, 0, 0, 0, 0, 35, 0, 0, 0, 0, 0, 50, 0, 0, 0, 0, 0] * 3, dtype=float)
    f = TSB().predict([lumpy, np.ones(30) * 10], 4)
    assert f.q.shape == (2, 4, 9) and (np.diff(f.q, axis=-1) >= 0).all() and (f.q >= 0).all()
    assert f.q[0, 0, 4] < f.q[0, 0, 8]
    assert abs(f.q[1, 0, 4] - 10) < 1e-6


def test_baseline_and_ensemble_return_forecast():
    s = np.arange(1, 60, dtype=float)
    b = Baseline("ma8")
    e = Ensemble([b, Const("c", 30)])
    assert e.predict([s], 4).q.shape == (1, 4, 9)


def test_router_dispatches_by_class_and_falls_back_for_short_history():
    reg = {"a": Const("a", 10), "b": Const("b", 100), "ma4": Const("ma4", 1)}
    r = Router(registry=reg, default="a", route={"intermittent": "b"})
    steady = np.ones(60); lumpy = np.array([0, 0, 5] * 30, dtype=float)
    f = r.predict([steady, lumpy], 3)
    assert f.point[0, 0] == 10 and f.point[1, 0] == 100


def test_licence_fallback_never_serves_non_commercial():
    class NC(Const):
        licence = "non-commercial"
    reg = {"timesfm": NC("timesfm", 10), "ma8": Const("ma8", 7), "ma4": Const("ma4", 1)}
    r = Router(registry=reg, default="timesfm", allow_non_commercial=False, licence_fallback="ma8")
    assert r.predict([np.ones(60)], 2).point[0, 0] == 7


def test_fit_routing_needs_a_clear_win():
    mat = np.tile(np.array([0, 0, 50, 0, 0, 0, 60, 0, 0, 0, 0, 40], dtype=float), (12, 6))
    good = TSB(); bad = Const("flat", 500)
    out = fit_routing(mat, [bad, good], origins=list(range(30, 60, 3)), default="flat", min_series=5)
    assert out["route"]["intermittent"] == "tsb"
