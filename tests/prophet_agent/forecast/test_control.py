"""Control of a TimesFM forecast: validation, leak-free slicing at a past origin, the engine's use of the model, the
tool surface, and (marked) the real model. Numbers from generated series are synthetic."""
from __future__ import annotations

import threading

import numpy as np
import pytest

from prophet_agent.forecast import timesfm as timesfm_mod
from prophet_agent.forecast.control import Control, build_control
from prophet_agent.forecast.rolling import rolling_backtest
from prophet_agent.forecast.timesfm import TimesFMEngine
from prophet_agent.forecast.types import Forecast


# ------------------------------------------------------------------------------------------------ validation --

def test_ranges_are_checked_merged_and_must_leave_data():
    c = build_control({"ignore_ranges": [[10, 12], [11, 15], [3, 4]]}, n=50, horizon=5)
    assert c.ignore == ((3, 4), (10, 15))
    assert c.describe()["ignored_points"] == 6
    for bad in ([[5, 5]], [[-1, 3]], [[0, 51]], [[1]], [["a", 2]], "x"):
        with pytest.raises(ValueError, match="ignore_ranges"):
            build_control({"ignore_ranges": bad}, n=50, horizon=5)
    with pytest.raises(ValueError, match="usable points"):
        build_control({"ignore_ranges": [[0, 45]]}, n=50, horizon=5)


def test_future_covariate_must_cover_history_plus_horizon_and_says_so():
    ok = build_control({"future_covariates": {"promo": [0.0] * 55}}, n=50, horizon=5)
    assert ok.future["promo"].shape == (55,)
    with pytest.raises(ValueError, match="needs exactly 55.*PLUS the horizon"):
        build_control({"future_covariates": {"promo": [0.0] * 50}}, n=50, horizon=5)
    with pytest.raises(ValueError, match="needs exactly 50"):
        build_control({"past_covariates": {"price": [1.0] * 55}}, n=50, horizon=5)


def test_covariates_by_name_come_from_the_source_file_pool():
    pool = {"promo": [0.0] * 55}
    c = build_control({"future_covariates": ["promo"]}, 50, 5, pool)
    assert list(c.future) == ["promo"]
    with pytest.raises(ValueError, match="not found.*available: \\['promo'\\]"):
        build_control({"future_covariates": ["holiday"]}, 50, 5, pool)


@pytest.mark.parametrize("bad", [{"future_covariates": {"x": [float("nan")] * 55}}, {"future_covariates": {"x": "abc"}},
                                 {"future_covariates": {f"c{i}": [0.0] * 55 for i in range(9)}}, {"znorm": "yes"},
                                 {"context_points": 3}, {"context_points": True}])
def test_bad_control_is_refused(bad):
    with pytest.raises(ValueError):
        build_control(bad, 50, 5)


def test_default_control_is_default_and_describes_nothing():
    c = build_control({}, 50, 5)
    assert c.is_default and c.describe() == {}


# ------------------------------------------------------------------------------------------- no future leaks --

def test_at_origin_cuts_every_array_to_what_was_knowable():
    n, h = 60, 4
    c = Control(ignore=((10, 12), (50, 55)), past={"p": np.arange(n, dtype=float)}, future={"f": np.arange(n + h, dtype=float)})
    o = c.at_origin(40, h)
    assert o.ignore == ((10, 12),)                         # a range that starts after the origin is not yet known
    assert len(o.past["p"]) == 40 and len(o.future["f"]) == 44
    assert o.future["f"][-1] == 43                         # the future covariate runs to the end of the test window
    o2 = c.at_origin(52, h)
    assert o2.ignore == ((10, 12), (50, 52))               # a range that straddles the origin is clipped to it


# ---------------------------------------------------------------------------------- engine, model stubbed --

class _Rec:
    def __init__(self):
        self.calls = []


def _engine(rec: _Rec, global_context=64):
    eng = object.__new__(TimesFMEngine)
    eng.model = type("M", (), {"global_context": global_context})()
    eng._infer_lock = threading.Lock()
    eng.checkpoint = "stub"

    def decode(tgt, po, pf, tmask, horizon):
        rec.calls.append(dict(tgt=tgt.copy(), po=po, pf=pf, tmask=tmask, horizon=horizon))
        base = float(np.mean(tgt[0][~tmask[0]] if tmask is not None else tgt[0]))
        out = np.zeros((tgt.shape[0] + (0 if po is None else po.shape[0]) + (0 if pf is None else pf.shape[0]), horizon, 9))
        out[0] = base + np.linspace(-1, 1, 9)[None, :]
        return out
    eng._decode = decode
    return eng


def test_context_window_slices_history_and_all_covariates_together():
    rec = _Rec()
    n, h = 100, 5
    x = np.arange(n, dtype=float) + 10
    c = Control(context_points=20, past={"p": np.arange(n, dtype=float)}, future={"f": np.arange(n + h, dtype=float)})
    _engine(rec).forecast_controlled(x, h, c)
    call = rec.calls[0]
    assert call["tgt"].shape == (1, 20) and call["po"].shape == (1, 20) and call["pf"].shape == (1, 25)
    assert call["pf"][0, 0] == 80 and call["pf"][0, -1] == 104       # aligned with the target window, runs h past it


def test_ignored_points_are_masked_and_their_values_never_reach_the_model():
    rec = _Rec()
    x = np.full(50, 10.0)
    x[20] = 9_999.0
    f = _engine(rec).forecast_controlled(x, 3, Control(ignore=((20, 21),)))
    call = rec.calls[0]
    assert bool(call["tmask"][0, 20]) and call["tmask"].sum() == 1
    assert call["tgt"][0, 20] == 0.0
    assert abs(float(f.point[0]) - 10.0) < 1.5                       # the spike did not leak into the level


def test_znorm_round_trips_and_symmetric_runs_twice_and_averages():
    rec = _Rec()
    x = np.linspace(100, 200, 40)
    f = _engine(rec).forecast_controlled(x, 3, Control(znorm=True))
    assert abs(rec.calls[0]["tgt"].mean()) < 1e-4 and abs(rec.calls[0]["tgt"].std() - 1) < 1e-3
    assert abs(float(f.point[0]) - x.mean()) < 5                     # un-normalised back to the original scale
    rec.calls.clear()
    _engine(rec).forecast_controlled(x, 3, Control(symmetric=True))
    assert len(rec.calls) == 2 and np.allclose(rec.calls[1]["tgt"], -rec.calls[0]["tgt"])


def test_floor_only_when_history_never_negative():
    rec = _Rec()
    pos = _engine(rec).forecast_controlled(np.full(30, 0.2), 2, Control(context_points=10))
    assert (pos.q >= 0).all()
    neg = _engine(_Rec()).forecast_controlled(np.linspace(-5, -1, 30), 2, Control(context_points=10))
    assert (neg.q < 0).any()


def test_too_few_usable_points_and_bad_future_length_are_refused():
    with pytest.raises(ValueError, match="fewer than 8"):
        _engine(_Rec()).forecast_controlled(np.arange(20.0), 2, Control(ignore=((0, 15),)))
    with pytest.raises(ValueError, match="span the context plus the horizon"):
        _engine(_Rec()).forecast_controlled(np.arange(20.0), 2, Control(future={"f": np.zeros(21)}))


# -------------------------------------------------------------------------------- backtest, engine stubbed --

class _BtEngine:
    """Default model repeats the last value; the controlled model repeats the last NON-ignored value, so ignoring spikes
    helps, and every call records how much future it could see."""
    checkpoint = "stub"
    global_context = 1000

    def __init__(self):
        self.seen = []

    def forecast_batch(self, series_list, horizon):
        point = np.stack([np.full(horizon, float(s[-1])) for s in series_list])
        return Forecast(point, np.repeat(point[:, :, None], 9, axis=2) + np.linspace(-1, 1, 9))

    def forecast_controlled(self, series, horizon, control):
        n = len(series)
        self.seen.append((n, {k: len(v) for k, v in control.future.items()}, {k: len(v) for k, v in control.past.items()}))
        ok = ~control.ignore_mask(n)
        last = float(np.asarray(series)[ok][-1])
        point = np.full(horizon, last)
        return Forecast(point, np.repeat(point[:, None], 9, axis=1) + np.linspace(-1, 1, 9))


def test_backtest_scores_configured_against_default_without_leaking_the_future():
    n, h = 80, 4
    x = np.full(n, 50.0) + np.random.default_rng(0).normal(0, 0.5, n)
    origins = [n - h * (8 - j) for j in range(8)]
    bad = origins[1::3]                # three origins whose last observed point is a keying error
    for o in bad:
        x[o - 1] = 500.0               # the default model repeats it
    ctl = Control(ignore=tuple((o - 1, o) for o in bad), future={"f": np.zeros(n + h)}, past={"p": np.zeros(n)})
    eng = _BtEngine()
    out = rolling_backtest(x, h, n_origins=8, season=1, engine=eng, control=ctl)
    assert out["models"]["timesfm_default"]["mase"] > out["models"]["timesfm"]["mase"]
    assert out["configured_vs_default_pct"] > 0
    assert out["control"]["ignored_points"] == 3
    for origin_n, fut, past in eng.seen:
        assert fut["f"] == origin_n + h and past["p"] == origin_n      # nothing beyond the test window, no later history
    assert "best_baseline" in out and out["best_baseline"] not in ("timesfm", "timesfm_default")


def test_backtest_without_control_is_unchanged():
    x = np.sin(np.arange(80) / 3) + 5
    out = rolling_backtest(x, 4, n_origins=6, season=1, engine=_BtEngine())
    assert "timesfm_default" not in out["models"] and "configured_vs_default_pct" not in out


# ----------------------------------------------------------------------------------------------- tool layer --

class _ToolEngine(_BtEngine):
    def forecast_controlled(self, series, horizon, control):
        self.last_control = control
        return super().forecast_controlled(series, horizon, control)


@pytest.fixture()
def tool_engine(monkeypatch):
    eng = _ToolEngine()
    monkeypatch.setattr(timesfm_mod.TimesFMEngine, "get", classmethod(lambda cls: eng))
    return eng


def test_tool_applies_control_and_records_it(tool_engine):
    from prophet_agent.tools.timesfm_tool import _run
    out = _run({"values": list(range(40)), "horizon": 3, "ignore_ranges": [[5, 7]], "znorm": True,
                "future_covariates": {"promo": [0.0] * 43}}, "k")
    assert out["config"]["control"] == {"ignored_ranges": [[5, 7]], "ignored_points": 2, "future_covariates": ["promo"], "znorm": True}
    assert tool_engine.last_control.znorm


def test_tool_default_path_has_no_control_in_config(tool_engine):
    from prophet_agent.tools.timesfm_tool import _run
    assert "control" not in _run({"values": list(range(40)), "horizon": 3}, "k")["config"]


def test_tool_refuses_per_series_control_on_many_series(tool_engine, monkeypatch):
    import importlib
    tt = importlib.import_module("prophet_agent.tools.timesfm_tool")
    monkeypatch.setattr(tt, "resolve_series", lambda a: [("a", np.arange(30.0)), ("b", np.arange(30.0))])
    with pytest.raises(ValueError, match="one series"):
        tt._run({"source": {}, "horizon": 3, "ignore_ranges": [[1, 2]]}, "k")


def test_backtest_tool_accepts_control_and_reports_comparison(monkeypatch):
    from prophet_agent.tools import data_tools
    eng = _BtEngine()
    monkeypatch.setattr(timesfm_mod.TimesFMEngine, "get", classmethod(lambda cls: eng))
    x = [50.0 + (i % 5) for i in range(80)]
    out = data_tools._backtest({"values": x, "horizon": 4, "n_origins": 4, "season": 1, "ignore_ranges": [[3, 4]]}, "k")
    assert "timesfm_default" in out["models"] and out["control"]["ignored_points"] == 1


# ---------------------------------------------------------------------------------------------- real model --

def _series(seed=7, n=300, h=28):
    rng = np.random.default_rng(seed)
    t = np.arange(n + h)
    promo = np.zeros(n + h)
    for s in range(20, n + h, 45):
        promo[s:s + 4] = 1
    full = (100 + 0.1 * t + 20 * np.sin(2 * np.pi * t / 7)) * (1 + 0.8 * promo) + rng.normal(0, 4, n + h)
    return full[:n], full[n:], promo, h


@pytest.mark.timesfm
def test_real_default_control_matches_the_library_exactly():
    y, _, _, h = _series()
    e = TimesFMEngine.get()
    lib, ctl = e.forecast(y, h), e.forecast_controlled(y, h, Control())
    assert np.array_equal(lib.point, ctl.point) and np.array_equal(lib.q, ctl.q)


@pytest.mark.timesfm
def test_real_masking_a_keying_error_removes_its_influence_and_covariate_changes_the_output():
    y, truth, promo, h = _series()
    y = y.copy()
    y[[100, 205]] *= 10                                           # synthetic keying errors
    e = TimesFMEngine.get()
    raw = e.forecast(y, h)
    masked = e.forecast_controlled(y, h, Control(ignore=((100, 101), (205, 206))))
    assert np.mean(np.abs(masked.point - truth)) < np.mean(np.abs(raw.point - truth))
    withcov = e.forecast_controlled(y, h, Control(ignore=((100, 101), (205, 206)), future={"promo": promo}))
    assert not np.allclose(withcov.point, masked.point)           # the covariate is read by the model
    assert (np.diff(withcov.q, axis=-1) >= 0).all() and withcov.q.shape == (h, 9)
    sym = e.forecast_controlled(y, h, Control(symmetric=True, znorm=True, context_points=128))
    assert np.isfinite(sym.q).all()


# ------------------------------------------------------------------------------------------- report path --

def test_report_note_states_what_was_applied():
    import importlib
    rt = importlib.import_module("prophet_agent.tools.report_tool")
    assert "default settings" in rt._method_note({})
    note = rt._method_note({"control": {"ignored_points": 3, "future_covariates": ["promo"], "znorm": True}})
    assert "3 history points ignored" in note and "promo" in note and "research use only" in note


def test_report_forwards_controls_to_the_forecast(tool_engine, monkeypatch):
    import importlib
    rt = importlib.import_module("prophet_agent.tools.report_tool")
    seen = {}
    real = rt.timesfm_run
    monkeypatch.setattr(rt, "timesfm_run", lambda a, k: seen.update(a) or real(a, k))
    monkeypatch.setattr(rt, "build_forecast_pack", lambda **kw: seen.update(note=kw["note"]) or type(
        "P", (), {"ok": True, "run_id": "r", "workdir": "w", "outputs": [], "stderr": ""})())
    out = rt._run({"values": [float(i % 7) + 10 for i in range(60)], "horizon": 4, "title": "t", "ignore_ranges": [[3, 5]],
                   "future_covariates": {"promo": [0.0] * 64}}, "k")
    assert seen["ignore_ranges"] == [[3, 5]] and "title" not in seen
    assert out["forecast"]["config"]["control"]["future_covariates"] == ["promo"]
    assert "promo" in seen["note"]
