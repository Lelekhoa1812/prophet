"""R6m, audit item 2: numerical edge cases in the forecast maths (all inputs synthetic).

Each test reproduces one defect found by a small probe before the fix. The probe for coverage gave 0.5 for a band that
contained its only finite actual: a missing actual was counted as a miss.
"""
from __future__ import annotations

import numpy as np

from prophet_agent.forecast import metrics


def test_coverage_leaves_out_missing_actuals_instead_of_counting_them_as_misses():
    actual = np.array([[np.nan, 5.0]])
    q = np.array([[[0.0, 1, 2, 3, 4, 5, 6, 7, 8], [0.0, 1, 2, 3, 4, 5, 6, 7, 8]]])
    assert metrics.coverage(actual, q) == 1.0                       # the one finite actual is inside 0 to 8
    assert np.isnan(metrics.coverage(np.array([[np.nan]]), q[:, :1]))  # nothing to measure
