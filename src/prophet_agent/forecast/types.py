"""Forecast container shared by the forecast modules. Ported from src/lab/engine/forecast.py (dataclass only)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Forecast:
    """Point and quantile forecast. Leading dimensions are the series, if any.

    One series: point (h,), q (h, 9). A batch: point (n, h), q (n, h, 9). q holds the quantiles 0.1 to 0.9 and is
    sorted along its last axis.
    """

    point: np.ndarray
    q: np.ndarray
