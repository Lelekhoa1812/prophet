"""TimesFM 3 wrapper (MLX backend on Apple silicon). One shared instance per process."""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass

import numpy as np

from ..config import TIMESFM_CHECKPOINT

QUANTILES = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)


@dataclass
class Forecast:
    point: np.ndarray        # (n, h)
    q: np.ndarray            # (n, h, 9) quantiles 0.1 .. 0.9


class TimesFMEngine:
    _inst: "TimesFMEngine | None" = None
    _lock = threading.Lock()

    def __init__(self):
        t = time.time()
        from timesfm3.mlx import TimesFM3Forecaster  # noqa: WPS433 (heavy import, lazy)
        self.model = TimesFM3Forecaster.from_pretrained(TIMESFM_CHECKPOINT)
        self.load_seconds = round(time.time() - t, 1)
        self.checkpoint = TIMESFM_CHECKPOINT

    @classmethod
    def get(cls) -> "TimesFMEngine":
        with cls._lock:
            if cls._inst is None:
                cls._inst = TimesFMEngine()
            return cls._inst

    def predict(self, contexts: list[np.ndarray], horizon: int, max_context: int = 512,
                chunk: int = 128) -> Forecast:
        pts, qs = [], []
        ctx = [np.asarray(c[-max_context:], dtype=np.float32) for c in contexts]
        for i in range(0, len(ctx), chunk):
            for out in self.model.predict_batch(ctx[i:i + chunk], horizon=horizon, return_quantiles=True):
                pts.append(np.asarray(out.forecast, dtype=np.float32))
                qs.append(np.asarray(out.quantiles, dtype=np.float32))
        point = np.clip(np.stack(pts), 0, None)
        q = np.clip(np.stack(qs), 0, None)
        q = np.sort(q, axis=-1)           # guard against quantile crossing
        return Forecast(point, q)

    def predict_cov(self, contexts: list[np.ndarray], horizon: int, future_covs: list[np.ndarray],
                    max_context: int = 512, chunk: int = 128) -> Forecast:
        """Univariate targets with known-future covariates. future_covs[i] has shape (k, len(context_i)+horizon)."""
        pts, qs = [], []
        ctx = [np.asarray(c[-max_context:], dtype=np.float32)[None, :] for c in contexts]
        cov = [np.asarray(f[:, -(c.shape[1] + horizon):], dtype=np.float32) for f, c in zip(future_covs, ctx)]
        for i in range(0, len(ctx), chunk):
            for out in self.model.predict_batch(ctx[i:i + chunk], horizon=horizon, past_future_covariates=cov[i:i + chunk],
                                                return_quantiles=True):
                pts.append(np.asarray(out.forecast, dtype=np.float32)[0])
                qs.append(np.asarray(out.quantiles, dtype=np.float32)[0])
        q = np.sort(np.clip(np.stack(qs), 0, None), axis=-1)
        return Forecast(np.clip(np.stack(pts), 0, None), q)
