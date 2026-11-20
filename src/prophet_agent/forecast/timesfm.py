"""TimesFM 3 wrapper (MLX backend, Apple silicon). Ported from the TimesFMEngine in src/lab/engine/forecast.py.

Motivation vs Logic
Motivation: loading the model takes seconds and MLX inference is not thread-safe, so every caller in a process must
share one loaded model and run inference one call at a time.
Logic: a process-wide singleton, created under a lock, with inference serialised by a second lock. The full official
context (the forecaster's global_context) is used. The legacy 512-point cap was a lab choice and is not kept.
Licence: the TimesFM 3 weights are non-commercial. This module is for research use only.
"""
from __future__ import annotations

import threading
import time

import numpy as np

from .types import Forecast

CHECKPOINT = "google/timesfm-3.0-pytorch"
N_QUANTILES = 9  # quantiles 0.1 .. 0.9; the median is index 4


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
        """Forecast many series in one model call. Returns point (n, horizon) and q (n, horizon, 9)."""
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
        point = np.clip(np.stack([np.asarray(o.forecast, dtype=np.float32) for o in outs]), 0, None)
        q = np.stack([np.asarray(o.quantiles, dtype=np.float32) for o in outs])
        if q.shape != (len(batch), horizon, N_QUANTILES):
            raise RuntimeError(f"unexpected quantile shape {q.shape}, expected {(len(batch), horizon, N_QUANTILES)}")
        q = np.sort(np.clip(q, 0, None), axis=-1)  # guard against quantile crossing
        return Forecast(point, q)


def forecast(series: np.ndarray, horizon: int) -> Forecast:
    """Forecast one series with the process-wide TimesFM 3 instance."""
    return TimesFMEngine.get().forecast(series, horizon)
