"""TimesFM 3 wrapper (MLX backend, Apple silicon). Ported from the TimesFMEngine in src/lab/engine/forecast.py.

Motivation vs Logic
Motivation: loading the model takes seconds and MLX inference is not thread-safe, so every caller in a process must
share one loaded model and run inference one call at a time.
Logic: a process-wide singleton, created under a lock, with inference serialised by a second lock. The full official
context (the forecaster's global_context) is used. The legacy 512-point cap was a lab choice and is not kept.
"""
from __future__ import annotations

import threading
import time

import numpy as np

from .types import Forecast

CHECKPOINT = "google/timesfm-3.0-pytorch"
N_QUANTILES = 9  # quantiles 0.1 .. 0.9; the median is index 4
# A sales, count or volume history with a few refund days below zero is still a non-negative measure. Live retail rep3
# (measured): two refund days (-1,683 and -1,566) in about 1,000 days switched the floor off, and Saturdays came out at
# -3,125 and -3,606. Up to this share of points below zero, the series is treated as non-negative and floored.
NEG_SHARE_MAX = 0.02


def nonnegative_domain(x) -> bool:
    """True when the series is a non-negative measure: finite, and at most NEG_SHARE_MAX of its points below zero. A signed
    series (returns, temperatures, balances) has far more negative points and keeps its negative forecasts."""
    a = np.asarray(x, dtype=float).reshape(-1)
    if a.size == 0 or not np.isfinite(a).all():
        return False
    return float(np.mean(a < 0)) <= NEG_SHARE_MAX


class TimesFMEngine:
    """One loaded TimesFM 3 forecaster. Use `TimesFMEngine.get()`, not the constructor."""

    _instance: "TimesFMEngine | None" = None
    _create_lock = threading.Lock()

    def __init__(self, checkpoint: str = CHECKPOINT):
        from timesfm3.mlx import TimesFM3Forecaster  # heavy import, deferred until a model is needed

        t = time.perf_counter()
        self.model = TimesFM3Forecaster.from_pretrained(checkpoint)
        self.load_seconds = round(time.perf_counter() - t, 2)
        self.checkpoint = checkpoint
        self._infer_lock = threading.Lock()

    @classmethod
    def get(cls) -> "TimesFMEngine":
        with cls._create_lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    @property
    def global_context(self) -> int:
        return int(self.model.global_context)

    def forecast(self, series: np.ndarray, horizon: int) -> Forecast:
        """Forecast one univariate series. Returns point (horizon,) and q (horizon, 9), clipped at 0 and sorted."""
        batch = self.forecast_batch([series], horizon)
        return Forecast(batch.point[0], batch.q[0])

    def forecast_batch(self, series_list, horizon: int) -> Forecast:
        """Forecast many series in one model call. Returns point (n, horizon) and q (n, horizon, 9). Forecasts are floored at zero only for
        non-negative series (see nonnegative_domain)."""
        if horizon < 1:
            raise ValueError("horizon must be at least 1")
        if len(series_list) == 0:
            raise ValueError("no series given")
        batch = []
        for series in series_list:
            x = np.asarray(series, dtype=np.float32).reshape(-1)
            if x.size == 0:
                raise ValueError("series is empty")
            if not np.isfinite(x).all():
                raise ValueError("series contains NaN or infinite values")
            batch.append(x[-self.global_context:])
        with self._infer_lock:
            outs = list(self.model.predict_batch(batch, horizon=horizon, return_quantiles=True))
        if len(outs) != len(batch):
            raise RuntimeError(f"TimesFM returned {len(outs)} forecasts for {len(batch)} series")
        point = np.stack([np.asarray(o.forecast, dtype=np.float32) for o in outs])
        q = np.stack([np.asarray(o.quantiles, dtype=np.float32) for o in outs])
        if q.shape != (len(batch), horizon, N_QUANTILES):
            raise RuntimeError(f"unexpected quantile shape {q.shape}, expected {(len(batch), horizon, N_QUANTILES)}")
        # A non-negative measure (counts, sales, a few refund days) cannot be forecast below zero. A signed series (returns,
        # temperatures, balances) is left alone: clipping it would bend a legitimate forecast. See NEG_SHARE_MAX.
        nonneg = np.array([nonnegative_domain(x) for x in batch])
        point = np.where(nonneg[:, None], np.clip(point, 0, None), point)
        q = np.where(nonneg[:, None, None], np.clip(q, 0, None), q)
        q = np.sort(q, axis=-1)  # guard against quantile crossing
        return Forecast(point, q)


    def _decode(self, tgt, po, pf, tmask, horizon: int) -> np.ndarray:
        import mlx.core as mx
        out = self.model.model.decode(
            mx.array(tgt)[None], horizon,
            past_only_covariates=mx.array(po)[None] if po is not None else None,
            past_future_covariates=mx.array(pf)[None] if pf is not None else None,
            target_mask=mx.array(tmask)[None] if tmask is not None else None)
        return np.array(out)[0]          # (variates, horizon, quantiles); the first row is the target

    def forecast_controlled(self, series, horizon: int, control) -> Forecast:
        """One series with the agent's choices applied (see forecast/control.py). Ignored points are masked, so the model
        does not read them at all. With a default control this equals ``forecast``. Returns point (horizon,), q (horizon, 9)."""
        if horizon < 1:
            raise ValueError("horizon must be at least 1")
        x = np.asarray(series, dtype=np.float32).reshape(-1)
        n = x.size
        if n == 0 or not np.isfinite(x).all():
            raise ValueError("series is empty or contains NaN or infinite values")
        keep = self.global_context if not control.context_points else min(control.context_points, self.global_context)
        start = max(0, n - keep)
        tgt = x[start:][None, :]
        ign = control.ignore_mask(n)[start:]
        if int((~ign).sum()) < 8:
            raise ValueError("fewer than 8 usable points remain in the context after ignoring")
        po = np.stack([v[start:] for _, v in sorted(control.past.items())]).astype(np.float32) if control.past else None
        pf = np.stack([v[start:] for _, v in sorted(control.future.items())]).astype(np.float32) if control.future else None
        if pf is not None and pf.shape[-1] != tgt.shape[-1] + horizon:
            raise ValueError(f"future covariates must span the context plus the horizon ({tgt.shape[-1] + horizon}), got {pf.shape[-1]}")
        # The mask and the placeholder values under it. The model replaces masked values by zero after its own scaling.
        tmask = ign[None, :] if ign.any() else None
        stats = None
        if control.znorm:
            def z(a, usable=None):
                v = a if usable is None else a[:, usable]
                mu = v.mean(axis=1, keepdims=True)
                sd = v.std(axis=1, keepdims=True)
                sd = np.where(sd < 1e-7, 1.0, sd)
                return ((a - mu) / sd).astype(np.float32), mu, sd
            tgt, mu, sd = z(tgt, ~ign)
            stats = (mu[0, 0], sd[0, 0])
            po = z(po)[0] if po is not None else None
            pf = z(pf)[0] if pf is not None else None
        if tmask is not None:
            tgt = np.where(tmask, 0.0, tgt).astype(np.float32)
        with self._infer_lock:
            out = self._decode(tgt, po, pf, tmask, horizon)
            if control.symmetric:
                neg = self._decode(-tgt, -po if po is not None else None, -pf if pf is not None else None, tmask, horizon)
                out = (np.sort(out, axis=-1) - np.sort(neg, axis=-1)[..., ::-1]) / 2
        out = np.sort(out, axis=-1)[0, :horizon, :]
        if stats is not None:
            out = out * stats[1] + stats[0]
        if nonnegative_domain(x[start:][~ign]):
            out = np.clip(out, 0, None)
        if out.shape != (horizon, N_QUANTILES):
            raise RuntimeError(f"unexpected quantile shape {out.shape}, expected {(horizon, N_QUANTILES)}")
        return Forecast(out[:, 4].astype(np.float32), out.astype(np.float32))


def forecast(series: np.ndarray, horizon: int) -> Forecast:
    """Forecast one series with the process-wide TimesFM 3 instance."""
    return TimesFMEngine.get().forecast(series, horizon)
