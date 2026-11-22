"""Hierarchical forecast reconciliation: MinT, with OLS and WLS-structural as simpler options and bottom-up as a baseline.

Motivation vs Logic
Motivation: a hierarchy (total, regions, products) is forecast node by node, so the parts rarely add up to the total.
A planner needs one coherent set of numbers. Reconciliation adjusts every node so each parent equals the sum of its
children, and it does so using all the base forecasts rather than trusting one level. Bottom-up keeps only the bottom
series, which is simple but discards the information at the upper levels and inherits the bottom-level bias.
Logic: write the hierarchy as a summing matrix S (n nodes by m bottom series), so every node is S times the bottom
values. Reconciled forecasts are S P yhat, where P maps the base forecasts yhat of all nodes to the bottom series. MinT
(Wickramasuriya, Athanasopoulos and Hyndman, 2019, JASA 114(526) 804-819) chooses P = (S' W^-1 S)^-1 S' W^-1, where W
is the covariance of the base forecast errors. Whatever P is, S P S = S, so every reconciled set is exactly coherent,
and forecasts that are already coherent are returned unchanged. The three choices of W are:
  - "mint_shrink" (default): W is the sample covariance of in-sample residuals, shrunk toward its diagonal with the
    Schafer and Strimmer (2005) intensity. The shrinkage keeps W positive definite when the history is short relative to
    the number of nodes. The intensity is reported as ``shrinkage``. The estimator was taken from the summary of
    Schafer and Strimmer (2005) in secondary sources; the original equations were not checked.
  - "wls_structural": W = diag(S 1), so a node that sums k bottom series has error variance k times a single series.
    This is the structural (WLS-s) scaling. It needs no residual history.
  - "ols": W = I. Simplest, and no error covariance is assumed.
The inverse is never formed. W = L L' (Cholesky) gives a whitening matrix Q = L^-1 with Q'Q = W^-1, and P is solved
as the least-squares problem (Q S) P = Q, which is stable when S has full column rank. A ridge of RIDGE times the mean
variance is added to W so near-singular covariances still factor. When the residual history is shorter than
MIN_RESIDUAL_ROWS, or is missing, MinT falls back to "wls_structural" and reports a warning rather than raising. The
caller decides which residuals are in-sample and suitable; this module only computes the reconciliation.
"""
from __future__ import annotations

from typing import Any

import numpy as np

METHODS = ("mint_shrink", "wls_structural", "ols")
DEFAULT_METHOD = "mint_shrink"
MIN_RESIDUAL_ROWS = 3      # fewer residual rows than this cannot give a usable covariance; MinT falls back with a warning
RIDGE = 1e-6               # added to W as RIDGE times the mean variance, so a singular covariance still factors


def _check_summing(S: Any) -> np.ndarray:
    S = np.asarray(S, dtype=float)
    if S.ndim != 2 or S.size == 0:
        raise ValueError("S must be a non-empty two-dimensional summing matrix")
    if not np.isin(S, (0.0, 1.0)).all():
        raise ValueError("S must hold only 0 and 1 entries")
    if (S.sum(axis=1) < 1).any():
        raise ValueError("every node must aggregate at least one bottom series")
    if np.linalg.matrix_rank(S) < S.shape[1]:
        raise ValueError("the columns of S must be independent: each bottom series needs its own column")
    return S


def _check_base(base: Any, n: int) -> np.ndarray:
    Y = np.asarray(base, dtype=float)
    if Y.ndim == 1:
        Y = Y[None, :]
    if Y.ndim != 2 or Y.shape[1] != n:
        raise ValueError(f"base forecasts must have one column per node: {n} nodes expected")
    if not np.isfinite(Y).all():
        raise ValueError("base forecasts must be finite")
    return Y


def _check_residuals(residuals: Any, n: int) -> np.ndarray | None:
    if residuals is None:
        return None
    E = np.asarray(residuals, dtype=float)
    if E.size == 0:
        return E.reshape(0, n)
    if E.ndim != 2 or E.shape[1] != n:
        raise ValueError(f"residuals must have one column per node: {n} nodes expected")
    if not np.isfinite(E).all():
        raise ValueError("residuals must be finite")
    return E


def shrinkage_covariance(residuals: Any) -> tuple[np.ndarray, float]:
    """Shrunk covariance of residuals (T rows, one column per node) and the shrinkage intensity used.

    Correlations are shrunk toward the identity with the Schafer and Strimmer intensity: the sum of the estimated
    variances of the sample correlations divided by the sum of their squares, clipped to [0, 1]. The variances of the
    sample correlations use the standardised cross-products, Var(r_ij) = T / (T - 1)^3 * sum_t (w_tij - mean w_ij)^2,
    where w_tij = z_ti z_tj. The covariance is the standard deviations times the shrunk correlations, so the shrinkage
    is toward the diagonal of the sample covariance. Zero-variance nodes keep a zero row and column.
    """
    E = np.asarray(residuals, dtype=float)
    if E.ndim != 2 or E.shape[0] < MIN_RESIDUAL_ROWS:
        raise ValueError(f"a shrinkage covariance needs at least {MIN_RESIDUAL_ROWS} residual rows")
    T, n = E.shape
    X = E - E.mean(axis=0)
    sd = X.std(axis=0, ddof=1)
    Z = np.divide(X, sd, out=np.zeros_like(X), where=sd > 0)
    R = Z.T @ Z / (T - 1)                                  # sample correlation; diagonal is 1 for non-constant nodes
    # sum over t of (w_tij - mean w_ij)^2 = sum_t w_tij^2 - (sum_t w_tij)^2 / T, with sum_t w_tij = (T - 1) R_ij
    ss = (Z ** 2).T @ (Z ** 2) - ((T - 1) ** 2) * R ** 2 / T
    var_r = T / (T - 1) ** 3 * np.clip(ss, 0.0, None)
    off = ~np.eye(n, dtype=bool)
    den = float((R[off] ** 2).sum())
    lam = 1.0 if den <= 0 else float(np.clip(var_r[off].sum() / den, 0.0, 1.0))
    Rs = (1.0 - lam) * R
    np.fill_diagonal(Rs, np.diag(R))
    cov = sd[:, None] * Rs * sd[None, :]
    return cov, lam


def _structural_whitener(S: np.ndarray) -> np.ndarray:
    """Q with Q'Q = W^-1 for W = diag(S 1): a node's error variance is the number of bottom series it sums."""
    return np.diag(1.0 / np.sqrt(S.sum(axis=1)))


def _spd_whitener(W: np.ndarray) -> tuple[np.ndarray | None, list[str]]:
    """Q = L^-1 with W + ridge I = L L', so Q'Q = (W + ridge I)^-1. Returns (None, warnings) if it still will not factor."""
    warns: list[str] = []
    n = W.shape[0]
    scale = float(np.mean(np.diag(W)))
    if not scale > 0:
        return None, ["the residual covariance has no positive variance, so it cannot be used"]
    nugget = RIDGE * scale
    if float(np.linalg.eigvalsh(W)[0]) <= nugget:
        warns.append("the residual covariance is singular or near-singular (collinear or zero-variance residuals); "
                     f"a ridge of {RIDGE:g} times the mean variance is added")
    try:
        L = np.linalg.cholesky(W + nugget * np.eye(n))
    except np.linalg.LinAlgError:
        return None, warns + ["the ridged residual covariance would not factor"]
    return np.linalg.solve(L, np.eye(n)), warns


def _projection(S: np.ndarray, Q: np.ndarray) -> np.ndarray:
    """P (m by n) with P = (S' W^-1 S)^-1 S' W^-1 when Q'Q = W^-1. Solved as least squares on (Q S) P = Q, never inverted."""
    return np.linalg.lstsq(Q @ S, Q, rcond=None)[0]


def reconcile(S: Any, base: Any, residuals: Any = None, method: str = DEFAULT_METHOD) -> dict[str, Any]:
    """Reconcile base forecasts (h by n, one column per node in S's row order) onto the hierarchy S (n by m).

    ``residuals`` are in-sample base-forecast residuals (T by n) and are used only by "mint_shrink". Returns
    {"reconciled": (h, n) coherent forecasts, "bottom": (h, m) reconciled bottom series, "method_used", "shrinkage"
    (the shrinkage intensity, or None when it was not estimated), "warnings": [str]}. Warnings record any fallback or
    ridge; they are never raised.
    """
    if method not in METHODS:
        raise ValueError(f"method must be one of {', '.join(METHODS)}")
    S = _check_summing(S)
    n = S.shape[0]
    Y = _check_base(base, n)
    warnings: list[str] = []
    shrink: float | None = None
    used = method

    if method == "ols":
        Q = np.eye(n)
    elif method == "wls_structural":
        Q = _structural_whitener(S)
    else:
        E = _check_residuals(residuals, n)
        Q = None
        if E is None:
            warnings.append("no residuals were given, so MinT uses WLS-structural weights instead")
        elif E.shape[0] < MIN_RESIDUAL_ROWS:
            warnings.append(f"only {E.shape[0]} residual row(s); MinT needs at least {MIN_RESIDUAL_ROWS}, "
                            "so WLS-structural weights are used instead")
        else:
            zero = np.flatnonzero(E.std(axis=0, ddof=1) == 0)
            if zero.size:
                warnings.append(f"node(s) {zero.tolist()} have zero residual variance")
            W, shrink = shrinkage_covariance(E)
            Q, spd_warnings = _spd_whitener(W)
            warnings.extend(spd_warnings)
        if Q is None:
            used = "wls_structural"
            shrink = None
            Q = _structural_whitener(S)

    P = _projection(S, Q)
    bottom = Y @ P.T
    reconciled = bottom @ S.T
    return {"reconciled": reconciled, "bottom": bottom, "method_used": used, "shrinkage": shrink,
            "warnings": warnings}


def bottom_up(S: Any, base: Any) -> dict[str, Any]:
    """Bottom-up baseline: keep the base forecasts of the bottom series and sum them up the hierarchy.

    A bottom series is a row of S holding a single 1. Every bottom column needs such a row.
    """
    S = _check_summing(S)
    n, m = S.shape
    Y = _check_base(base, n)
    rows = []
    for j in range(m):
        found = [r for r in range(n) if S[r, j] == 1 and S[r].sum() == 1]
        if not found:
            raise ValueError(f"bottom-up needs a row of S that is bottom series {j} alone")
        rows.append(found[0])
    bottom = Y[:, rows]
    return {"reconciled": bottom @ S.T, "bottom": bottom}
