"""MinT reconciliation: coherence, identity, hand-checked answers, a synthetic study, and no crash on short or singular input."""
from __future__ import annotations

import numpy as np
import pytest

from prophet_agent.forecast.reconcile import bottom_up, reconcile, shrinkage_covariance

# 2-level hierarchy: rows are [total, b1, b2]
S2 = np.array([[1, 1],
               [1, 0],
               [0, 1]], dtype=float)

# 3-level hierarchy: rows are [total, A, B, b1, b2, b3, b4], with A = b1 + b2 and B = b3 + b4
S3 = np.array([[1, 1, 1, 1],
               [1, 1, 0, 0],
               [0, 0, 1, 1],
               [1, 0, 0, 0],
               [0, 1, 0, 0],
               [0, 0, 1, 0],
               [0, 0, 0, 1]], dtype=float)


def _coherent_check(rec: np.ndarray, tol: float = 1e-9) -> None:
    """Explicit tree sums for the 3-level hierarchy: total = A + B, A = b1 + b2, B = b3 + b4."""
    total, a, b = rec[:, 0], rec[:, 1], rec[:, 2]
    assert np.allclose(total, a + b, atol=tol)
    assert np.allclose(a, rec[:, 3] + rec[:, 4], atol=tol)
    assert np.allclose(b, rec[:, 5] + rec[:, 6], atol=tol)


def _coherent_check_2level(rec: np.ndarray, tol: float = 1e-9) -> None:
    assert np.allclose(rec[:, 0], rec[:, 1] + rec[:, 2], atol=tol)


# (a) coherence ------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("method", ["mint_shrink", "wls_structural", "ols"])
def test_two_level_reconciled_forecasts_sum_exactly(method):
    rng = np.random.default_rng(1)
    base = rng.normal(10, 3, size=(4, 3))                    # independent, so incoherent on purpose
    resid = rng.normal(0, 2, size=(40, 3))
    out = reconcile(S2, base, resid, method=method)
    _coherent_check_2level(out["reconciled"])
    assert np.allclose(out["reconciled"], out["bottom"] @ S2.T, atol=1e-9)


@pytest.mark.parametrize("method", ["mint_shrink", "wls_structural", "ols"])
def test_three_level_reconciled_forecasts_sum_exactly(method):
    rng = np.random.default_rng(2)
    base = rng.normal(20, 4, size=(5, 7))
    resid = rng.normal(0, 3, size=(60, 7))
    out = reconcile(S3, base, resid, method=method)
    _coherent_check(out["reconciled"])


# (b) identity ------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("method", ["mint_shrink", "wls_structural", "ols"])
def test_coherent_base_forecasts_come_back_unchanged(method):
    rng = np.random.default_rng(3)
    bottom = rng.uniform(5, 15, size=(6, 4))
    base = bottom @ S3.T                                    # already coherent: every parent is its children's sum
    resid = rng.normal(0, 1, size=(50, 7))
    out = reconcile(S3, base, resid, method=method)
    assert np.allclose(out["reconciled"], base, atol=1e-9)


# (c) known answers ---------------------------------------------------------------------------------------------------

def test_ols_matches_a_hand_computed_two_level_answer():
    # S'S = [[2, 1], [1, 2]], S'yhat = (14, 15), so bottom = (13/3, 16/3) and total = 29/3
    base = np.array([[10.0, 4.0, 5.0]])
    out = reconcile(S2, base, method="ols")
    assert out["bottom"][0] == pytest.approx([13 / 3, 16 / 3], abs=1e-12)
    assert out["reconciled"][0] == pytest.approx([29 / 3, 13 / 3, 16 / 3], abs=1e-12)


def test_wls_structural_matches_a_hand_computed_two_level_answer():
    # W = diag(S 1) = diag(2, 1, 1). M = S'W^-1 S = [[1.5, 0.5], [0.5, 1.5]], S'W^-1 yhat = (9, 10), so bottom = (4.25, 5.25)
    base = np.array([[10.0, 4.0, 5.0]])
    out = reconcile(S2, base, method="wls_structural")
    assert out["bottom"][0] == pytest.approx([4.25, 5.25], abs=1e-12)
    assert out["reconciled"][0] == pytest.approx([9.5, 4.25, 5.25], abs=1e-12)


def test_mint_agrees_with_the_textbook_formula_on_a_small_case():
    # P = (S' W^-1 S)^-1 S' W^-1 with an explicit inverse, as a cross-check on the stable solver
    rng = np.random.default_rng(4)
    resid = rng.normal(0, 2, size=(30, 7))
    base = rng.normal(20, 4, size=(3, 7))
    W, _ = shrinkage_covariance(resid)
    Winv = np.linalg.inv(W)
    P = np.linalg.inv(S3.T @ Winv @ S3) @ S3.T @ Winv
    expected = (base @ (S3 @ P).T)
    out = reconcile(S3, base, resid, method="mint_shrink")
    # the module adds a ridge of 1e-6 times the mean variance, so the naive inverse agrees to a relative 1e-4, not exactly
    assert out["reconciled"] == pytest.approx(expected, rel=1e-4)


def test_bottom_up_keeps_the_bottom_base_forecasts():
    base = np.array([[10.0, 4.0, 5.0], [7.0, 2.0, 6.0]])
    out = bottom_up(S2, base)
    assert out["bottom"] == pytest.approx(np.array([[4.0, 5.0], [2.0, 6.0]]))
    assert out["reconciled"] == pytest.approx(np.array([[9.0, 4.0, 5.0], [8.0, 2.0, 6.0]]))


# (d) synthetic study -------------------------------------------------------------------------------------------------

def test_synthetic_study_mint_is_no_worse_than_bottom_up_on_total_squared_error():
    """SYNTHETIC replay, not observed data. A 3-level hierarchy with a known bottom mean, independent base forecasts
    whose bottom-level forecasts are biased (+3, -2, +4, -3) and whose upper-level forecasts are unbiased, with
    error sd 2 * sqrt(number of bottom series under the node). The in-sample residuals come from the same process.
    Fixed seed 20261122, 300 draws. Compared on the same draw: total squared error over all seven nodes."""
    rng = np.random.default_rng(20261122)
    draws, T_hist = 300, 60
    mu_bottom = np.array([30.0, 20.0, 15.0, 35.0])
    leaves = S3.sum(axis=1)
    sd = 2.0 * np.sqrt(leaves)
    bias = np.array([0.0, 0.0, 0.0, 3.0, -2.0, 4.0, -3.0])
    se_bu, se_mint = 0.0, 0.0
    for _ in range(draws):
        truth = S3 @ (mu_bottom + rng.normal(0, 1, size=4))          # coherent realised values
        base = truth + bias + sd * rng.normal(size=7)                 # biased, independent, incoherent
        history = bias + sd * rng.normal(size=(T_hist, 7))            # in-sample residuals, same process
        bu = bottom_up(S3, base[None, :])["reconciled"][0]
        mi = reconcile(S3, base[None, :], history, method="mint_shrink")["reconciled"][0]
        se_bu += float(np.sum((bu - truth) ** 2))
        se_mint += float(np.sum((mi - truth) ** 2))
    assert se_mint / draws <= se_bu / draws


# (e) short history, singular covariance, and validation --------------------------------------------------------------

@pytest.mark.parametrize("rows", [0, 1, 2])
def test_short_residual_history_warns_and_falls_back_without_raising(rows):
    base = np.array([[10.0, 4.0, 5.0]])
    resid = np.zeros((rows, 3)) if rows == 0 else np.arange(rows * 3, dtype=float).reshape(rows, 3)
    out = reconcile(S2, base, resid, method="mint_shrink")
    assert out["method_used"] == "wls_structural"
    assert any("residual" in w for w in out["warnings"])
    _coherent_check_2level(out["reconciled"])


def test_missing_residuals_warn_and_fall_back():
    base = np.array([[10.0, 4.0, 5.0]])
    out = reconcile(S2, base, None, method="mint_shrink")
    assert out["method_used"] == "wls_structural" and out["warnings"]
    _coherent_check_2level(out["reconciled"])


def test_singular_covariance_is_ridged_and_warned_not_raised():
    rng = np.random.default_rng(5)
    base = rng.normal(20, 4, size=(2, 7))
    resid = rng.normal(0, 2, size=(12, 7))
    resid[:, 1] = resid[:, 0]                 # two nodes with identical residuals: a singular covariance
    resid[:, 6] = 0.0                         # one node with zero residual variance
    out = reconcile(S3, base, resid, method="mint_shrink")
    assert out["method_used"] == "mint_shrink"
    assert np.isfinite(out["reconciled"]).all()
    _coherent_check(out["reconciled"])
    assert any("singular" in w or "zero" in w or "ridge" in w for w in out["warnings"])


def test_bad_inputs_are_refused():
    base = np.ones((1, 3))
    with pytest.raises(ValueError):
        reconcile(np.array([[1, 2], [1, 0], [0, 1.0]]), base, method="ols")   # a non 0/1 entry
    with pytest.raises(ValueError):
        reconcile(S2, base, method="nope")
    with pytest.raises(ValueError):
        reconcile(S2, np.ones((1, 4)), method="ols")                         # base does not cover every node
    with pytest.raises(ValueError):
        reconcile(S2, base, np.ones((5, 2)), method="mint_shrink")          # residuals with the wrong node count
