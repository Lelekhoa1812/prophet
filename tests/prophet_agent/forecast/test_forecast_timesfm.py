"""TimesFM 3 wrapper checks (ported from TimesFMEngine in src/lab/engine/forecast.py).

The offline tests stub the model. Only the marked test loads the real checkpoint.
"""
import threading
from types import SimpleNamespace

import numpy as np
import pytest

from prophet_agent.forecast.timesfm import TimesFMEngine


def _engine_with_stub(predict_batch, global_context=16):
    eng = object.__new__(TimesFMEngine)  # bypass __init__, which loads the real model
    eng.model = SimpleNamespace(global_context=global_context, predict_batch=predict_batch)
    eng._infer_lock = threading.Lock()
    eng.checkpoint = "stub"
    return eng


def test_stub_output_is_clipped_sorted_and_context_capped():
    seen_lengths = []

    def predict_batch(contexts, horizon, return_quantiles):
        assert return_quantiles is True
        seen_lengths.extend(len(c) for c in contexts)
        raw_q = np.linspace(-1.0, 7.0, 9)[::-1]  # descending and partly negative, as a bad output might be
        q = np.tile(raw_q, (horizon, 1))
        return [SimpleNamespace(forecast=np.full(horizon, -2.0), quantiles=q)]

    eng = _engine_with_stub(predict_batch, global_context=16)
    fc = eng.forecast(np.arange(40, dtype=float), horizon=3)
    assert seen_lengths == [16]
    assert fc.point.shape == (3,)
    assert fc.q.shape == (3, 9)
    assert (fc.point >= 0).all() and (fc.q >= 0).all()
    assert (np.diff(fc.q, axis=-1) >= 0).all()


def test_rejects_non_finite_and_empty_input():
    eng = _engine_with_stub(lambda *a, **k: [])
    with pytest.raises(ValueError):
        eng.forecast(np.array([1.0, np.nan]), horizon=2)
    with pytest.raises(ValueError):
        eng.forecast(np.array([]), horizon=2)


@pytest.mark.timesfm
def test_real_timesfm_shapes_quantile_order_and_non_negativity():
    rng = np.random.default_rng(0)
    series = 10 + 3 * np.sin(np.arange(120) * 2 * np.pi / 52) + rng.normal(0, 1, 120)
    engine = TimesFMEngine.get()
    fc = engine.forecast(series, horizon=8)
    assert fc.point.shape == (8,)
    assert fc.q.shape == (8, 9)
    assert (fc.point >= 0).all() and (fc.q >= 0).all()
    assert (np.diff(fc.q, axis=-1) >= 0).all()
    assert TimesFMEngine.get() is engine  # one instance per process
