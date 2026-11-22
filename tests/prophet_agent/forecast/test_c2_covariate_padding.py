"""A6-10 (measurement path): the covariate horizon padding is a choice the agent can make and the lab can measure.

Motivation vs Logic
Motivation: the TimesFM 3 package pads a future covariate to the patch-rounded horizon by repeating its last value when
padding_mode is "edge" (timesfm3/mlx/timesfm3_forecaster.py, lines 318-327, code-read). Prophet calls the model's decode
directly and never set the mode, so the lever was never tested (audit A6-10). Live runs used no covariates at all.
Logic: Control carries covariate_padding ("none" or "edge"). With "edge" the engine appends the last future value up to
ceil(horizon / 64) * 64 before decoding, as the package does, and trims the output to the horizon. The lab gets a candidate
that measures it. The model is stubbed; these tests check the inputs the model is given, not a numeric target.
"""
from __future__ import annotations

import threading
from types import SimpleNamespace

import numpy as np
import pytest

from prophet_agent.forecast.control import CONTROL_ARGS, Control, build_control
from prophet_agent.forecast.timesfm import OUTPUT_PATCH, TimesFMEngine, edge_pad_future


def _engine(global_context=1024):
    eng = object.__new__(TimesFMEngine)  # bypass __init__, which loads the real model
    eng.model = SimpleNamespace(global_context=global_context, predict_batch=lambda *a, **k: [])
    eng._infer_lock = threading.Lock()
    eng.checkpoint = "stub"
    return eng


def test_edge_padding_is_a_named_control_argument_and_off_by_default():
    assert "covariate_padding" in CONTROL_ARGS
    n, h = 120, 20
    promo = [float(i % 3) for i in range(n + h)]
    off = build_control({"future_covariates": {"promo": promo}}, n, h)
    on = build_control({"future_covariates": {"promo": promo}, "covariate_padding": "edge"}, n, h)
    assert off.padding_mode == "none" and "covariate_padding" not in off.describe()
    assert on.padding_mode == "edge" and on.describe()["covariate_padding"] == "edge"
    assert not on.is_default


def test_unknown_padding_and_padding_without_covariates_are_refused():
    with pytest.raises(ValueError, match="covariate_padding"):
        build_control({"future_covariates": {"p": [0.0] * 25}, "covariate_padding": "zeros"}, 5, 20)
    with pytest.raises(ValueError, match="needs past_covariates or future_covariates"):
        build_control({"covariate_padding": "edge"}, 120, 20)


def test_the_padding_survives_backtest_origins():
    n, h = 120, 20
    c = build_control({"future_covariates": {"p": [1.0] * (n + h)}, "covariate_padding": "edge"}, n, h)
    assert c.at_origin(80, h).padding_mode == "edge"


def test_edge_pad_extends_by_the_last_value_to_the_patch_rounded_horizon():
    assert OUTPUT_PATCH == 64
    pf = np.arange(140, dtype=np.float32)[None, :]          # context 120 plus horizon 20
    padded = edge_pad_future(pf, 20)
    assert padded.shape == (1, 120 + 64)                    # ceil(20 / 64) * 64 = 64 decoded steps after the context
    np.testing.assert_array_equal(padded[:, :140], pf)
    assert (padded[:, 140:] == pf[:, -1:]).all()


def test_a_horizon_that_is_already_a_patch_multiple_is_not_padded():
    pf = np.ones((2, 128), dtype=np.float32)
    assert edge_pad_future(pf, 64).shape == (2, 128)


def test_the_engine_gives_the_model_the_edge_padded_covariate(monkeypatch):
    n, h = 120, 20
    promo = np.array([float(i % 4) for i in range(n + h)])
    seen = []

    def fake_decode(self, tgt, po, pf, tmask, horizon):
        seen.append(None if pf is None else np.array(pf))
        return np.tile(np.linspace(0, 8, 9), (1, horizon, 1)).astype(np.float32)

    monkeypatch.setattr(TimesFMEngine, "_decode", fake_decode)
    eng = _engine()
    control = build_control({"future_covariates": {"promo": list(promo)}, "covariate_padding": "edge"}, n, h)
    fc = eng.forecast_controlled(np.arange(n, dtype=float) + 10, h, control)
    assert fc.point.shape == (h,)
    assert seen[0].shape == (1, n + h + (OUTPUT_PATCH - h))   # the context plus the 64 decoded steps
    np.testing.assert_array_equal(seen[0][0, :n + h], promo)
    assert (seen[0][0, n + h:] == promo[-1]).all()


def test_without_padding_the_model_gets_the_covariate_as_given(monkeypatch):
    n, h = 120, 20
    promo = [float(i % 4) for i in range(n + h)]
    seen = []

    def fake_decode(self, tgt, po, pf, tmask, horizon):
        seen.append(np.array(pf))
        return np.tile(np.linspace(0, 8, 9), (1, horizon, 1)).astype(np.float32)

    monkeypatch.setattr(TimesFMEngine, "_decode", fake_decode)
    control = build_control({"future_covariates": {"promo": promo}}, n, h)
    _engine().forecast_controlled(np.arange(n, dtype=float) + 10, h, control)
    assert seen[0].shape == (1, n + h)


def test_the_lab_measures_the_padding_as_its_own_candidate():
    from prophet_agent.forecast.lab import CANDIDATE_NAMES, FoldContext, _grid
    assert "covariates_edge_pad" in CANDIDATE_NAMES
    names = [c.name for c in _grid(None, True)]
    assert names[-2:] == ["with_covariates", "covariates_edge_pad"]
    assert "covariates_edge_pad" not in [c.name for c in _grid(None, False)]
    cand = {c.name: c for c in _grid(None, True)}["covariates_edge_pad"]
    ctx = FoldContext(hist=np.arange(60, dtype=float), origin=60, horizon=20, period=None,
                      past={}, future={"promo": np.arange(80, dtype=float)})
    assert cand.prepare(ctx).control.padding_mode == "edge"
    assert isinstance(cand.prepare(ctx).control, Control)
