"""A6-12 (code part): quantile crossings are counted before the engine sorts them, and the count is readable by the caller.

Motivation vs Logic
Motivation: the engine sorts TimesFM's nine quantiles silently (audit A6-12), so nobody knows how often the model's raw
quantiles cross. The crossing rate is the evidence for whether the sort is a rounding fix or a real defect.
Logic: count adjacent descents (q[k+1] < q[k]) in the raw output, per series, before any clip or sort. The count is kept in
thread-local state so the tool that made the call reads its own count, not another thread's. The model is stubbed.
"""
from __future__ import annotations

import threading
from types import SimpleNamespace

import numpy as np

from prophet_agent.forecast.control import Control
from prophet_agent.forecast.timesfm import TimesFMEngine


def _engine(predict_batch=None, global_context=16):
    eng = object.__new__(TimesFMEngine)  # bypass __init__, which loads the real model
    eng.model = SimpleNamespace(global_context=global_context, predict_batch=predict_batch or (lambda *a, **k: []))
    eng._infer_lock = threading.Lock()
    eng.checkpoint = "stub"
    return eng


def _one_swap_per_step(horizon: int) -> np.ndarray:
    """Nine quantiles per step with the 0.3 and 0.4 quantiles swapped: exactly one descent per step."""
    q = np.tile(np.arange(9, dtype=float) + 1.0, (horizon, 1))
    q[:, [2, 3]] = q[:, [3, 2]]
    return q


def test_batch_counts_one_crossing_per_step_before_sorting():
    horizon = 3

    def predict_batch(contexts, horizon, return_quantiles):
        return [SimpleNamespace(forecast=np.full(horizon, 5.0), quantiles=_one_swap_per_step(horizon))]

    eng = _engine(predict_batch)
    fc = eng.forecast_batch([np.arange(30, dtype=float)], horizon)
    assert eng.last_quantile_crossings() == [horizon]
    assert (np.diff(fc.q, axis=-1) >= 0).all()          # still sorted for the caller


def test_monotone_quantiles_count_zero_crossings():
    def predict_batch(contexts, horizon, return_quantiles):
        return [SimpleNamespace(forecast=np.full(horizon, 5.0), quantiles=np.tile(np.arange(9, dtype=float), (horizon, 1)))]

    eng = _engine(predict_batch)
    eng.forecast_batch([np.arange(30, dtype=float)], 4)
    assert eng.last_quantile_crossings() == [0]


def test_batch_counts_are_per_series():
    def predict_batch(contexts, horizon, return_quantiles):
        ok = np.tile(np.arange(9, dtype=float), (horizon, 1))
        return [SimpleNamespace(forecast=np.zeros(horizon), quantiles=_one_swap_per_step(horizon)),
                SimpleNamespace(forecast=np.zeros(horizon), quantiles=ok)]

    eng = _engine(predict_batch)
    eng.forecast_batch([np.arange(30, dtype=float), np.arange(30, dtype=float)], 2)
    assert eng.last_quantile_crossings() == [2, 0]


def test_controlled_path_counts_the_raw_model_output(monkeypatch):
    horizon = 3
    eng = _engine()
    raw = _one_swap_per_step(horizon)[None, :, :]          # (variates, horizon, quantiles) as _decode returns it

    monkeypatch.setattr(TimesFMEngine, "_decode", lambda self, tgt, po, pf, tmask, h: raw.copy())
    fc = eng.forecast_controlled(np.arange(30, dtype=float), horizon, Control())
    assert eng.last_quantile_crossings() == [horizon]
    assert (np.diff(fc.q, axis=-1) >= 0).all()


def test_no_call_yet_reports_none():
    assert _engine().last_quantile_crossings() is None


def test_counts_do_not_leak_across_threads():
    def predict_batch(contexts, horizon, return_quantiles):
        return [SimpleNamespace(forecast=np.zeros(horizon), quantiles=_one_swap_per_step(horizon))]

    eng = _engine(predict_batch)
    seen = {}

    def other():
        seen["other_before"] = eng.last_quantile_crossings()

    t = threading.Thread(target=other)
    eng.forecast_batch([np.arange(30, dtype=float)], 2)
    t.start()
    t.join()
    assert seen["other_before"] is None
    assert eng.last_quantile_crossings() == [2]
