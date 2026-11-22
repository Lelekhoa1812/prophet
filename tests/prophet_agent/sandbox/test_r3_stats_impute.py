"""Round 3 (A2-03): two-sided imputation leaks future values into the history a backtest reads.

Motivation: a gap at 50 to 55 was filled by interpolating to index 56. Editing only index 56 changed the filled values at
50 and 51, which a backtest origin at 52 reads (measured on the synthetic series below: 7.14 with the two-sided
default, 0.0 causal). Logic: the causal path must stay future-blind, and the two-sided path must be shown to leak, so
the default matters. The default itself is pending: changing it breaks an existing test that pins the two-sided
default (tests/prophet_agent/sandbox/test_prophet_stats.py, test_impute_linear_fills_short_gap_and_reports_it), which is
outside the file set for this round. The strict xfail below turns into a failure once the default is flipped, so the
marker is removed with that change.
"""
from __future__ import annotations

import inspect

import numpy as np
import pytest

from prophet_agent.sandbox import prophet_stats as ps


def _gapped_series() -> tuple[np.ndarray, np.ndarray]:
    x = np.cumsum(np.random.default_rng(0).normal(0, 1, 80)) + 35.0
    x[50:56] = np.nan  # gap 50 to 55, origin at 52
    edited = x.copy()
    edited[56] += 25.0  # only a future point changes
    return x, edited


def test_causal_imputation_is_unchanged_by_a_future_edit():
    x, edited = _gapped_series()
    a, _ = ps.impute(x, method="linear", causal=True)
    b, _ = ps.impute(edited, method="linear", causal=True)
    assert np.array_equal(a.iloc[:52].to_numpy(), b.iloc[:52].to_numpy())


def test_two_sided_imputation_leaks_a_future_edit_into_history():
    # Documents the defect the causal path avoids: the same edit moves the history by several units.
    x, edited = _gapped_series()
    a, _ = ps.impute(x, method="linear", causal=False)
    b, _ = ps.impute(edited, method="linear", causal=False)
    assert float(np.max(np.abs(a.iloc[:52].to_numpy() - b.iloc[:52].to_numpy()))) > 1.0


@pytest.mark.xfail(strict=True, reason="A2-03 default flip pending: see module docstring")
def test_default_imputation_is_causal():
    assert inspect.signature(ps.impute).parameters["causal"].default is True
