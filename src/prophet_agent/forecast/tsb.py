"""Teunter-Syntetos-Babai (TSB) model for intermittent demand. Ported from the TSB class in src/lab/engine/models.py.

The Router and fit_routing are not ported. This module holds only the model.

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


def expected_volume(context: Sequence[float] | np.ndarray, horizon: int, alpha: float = 0.15) -> np.ndarray:
    """The expected demand per period under TSB: chance of demand times the mean of the last 52 non-zero sizes.

    This is the mean, not the median. For lumpy demand the median is zero, so a point forecast taken from a median can
    understate the volume the business will actually need by most of its amount. The expectation is exact (the sampler in
    ``TSB.predict`` draws sizes uniformly from the same window), and it is constant over the horizon, as TSB is."""
    s = np.nan_to_num(np.asarray(context, dtype=float))
    p = 0.5
    for v in s:
        p = p + alpha * ((1.0 if v > 0 else 0.0) - p)
    sizes = s[s > 0][-52:]
    level = float(p * sizes.mean()) if len(sizes) else 0.0
    return np.full(horizon, level)


INTERMITTENT_ADI = 1.32     # Syntetos-Boylan cut-off, the same as tools/timesfm_tool.py (one definition should move here)


def is_intermittent(series: Sequence[float] | np.ndarray) -> bool:
    """True when demand is intermittent: the average gap between periods with demand is above INTERMITTENT_ADI.

    Motivation vs Logic
    Motivation: the backtest must score the forecast that is delivered, and the delivered forecast for lumpy demand is TSB's
    (audit A6-01). The backtest needs the same test that the delivery tool applies.
    Logic: the count of periods with demand against the length of the series (the inverse of the demand interval). Only a
    series that never goes below zero can be demand; a series with negative values is never called intermittent.
    """
    s = np.nan_to_num(np.asarray(series, dtype=float))
    if (s < 0).any():
        return False            # a series that goes below zero is not demand (a price, a balance): zeros there are not "no sale"
    sold = int(np.count_nonzero(s > 0))
    return sold > 0 and len(s) / sold > INTERMITTENT_ADI


def tsb_band(context: Sequence[float] | np.ndarray, horizon: int) -> np.ndarray:
    """The TSB quantiles (horizon, 9) that the delivered band for intermittent demand is made of.

    Motivation vs Logic
    Motivation: when the point is the expected volume, the band must be the one the same model draws it from; a TimesFM band
    around a near-zero median does not contain it (audit A6-01, A6-02).
    Logic: TSB's own sampler (TSB.predict) for one series, widened so that it always brackets the expected volume. The
    sampler's 90th quantile is zero whenever the chance of demand is below 10 percent, so the expected volume (a positive
    mean) can sit above the band; the band is then extended to the point, and the widening is reported by the caller as
    coverage above nominal. The band thus always contains the point it is delivered with.
    """
    ctx = np.asarray(context, dtype=float)
    q = TSB().predict([ctx], horizon).q[0].copy()
    point = expected_volume(ctx, horizon)
    q[:, 8] = np.maximum(q[:, 8], point)
    q[:, 0] = np.minimum(q[:, 0], point)
    return q


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
