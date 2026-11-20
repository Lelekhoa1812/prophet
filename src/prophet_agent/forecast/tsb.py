"""Teunter-Syntetos-Babai (TSB) model for intermittent demand. Ported from the TSB class in src/lab/engine/models.py.

The Router, fit_routing and the licence switch are not ported. This module holds only the model.

Motivation vs Logic
Motivation: lumpy series (long runs of zeros, then a large order) get a misleading smooth average from most models.
Logic: forecast the chance of any demand times the size of a typical order. Quantiles come from a deterministic
Monte-Carlo of (occurs ? size : 0) summed over each step, so the median can sit near zero with a long right tail.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from .metrics import QS
from .types import Forecast


@dataclass
class TSB:
    """Chance of demand (smoothed by alpha) times the sizes of the last 52 non-zero orders, sampled per step."""

    alpha: float = 0.15
    n: int = 400
    seed: int = 5

    def predict(self, contexts: Sequence[np.ndarray], horizon: int) -> Forecast:
        rng = np.random.default_rng(self.seed)
        u = rng.random((self.n, horizon))
        pick = rng.random((self.n, horizon))
        qs = np.empty((len(contexts), horizon, len(QS)))
        for i, c in enumerate(contexts):
            s = np.nan_to_num(np.asarray(c, dtype=float))
            p = 0.5
            for v in s:
                p = p + self.alpha * ((1.0 if v > 0 else 0.0) - p)
            sizes = s[s > 0][-52:]
            if len(sizes) == 0:
                qs[i] = 0.0
                continue
            draw = sizes[(pick * len(sizes)).astype(int) % len(sizes)]
            samples = np.where(u < p, draw, 0.0)
            qs[i] = np.quantile(samples, QS, axis=0).T
        q = np.sort(qs, axis=-1)
        median_index = int(np.argmin(np.abs(QS - 0.5)))
        return Forecast(q[:, :, median_index].copy(), q)
