"""reconcile_forecasts: make base forecasts of a hierarchy add up (MinT by default; WLS-structural, OLS or bottom-up).

Motivation vs Logic
Motivation: a planner forecasts the total, its regions and their products, and the separate numbers rarely add up. A budget
or a supply plan needs one set that is coherent, so each parent equals the sum of its children. The choice of method is a
backtest question, so the tool reports which method it used and says when a fallback was taken.
Logic: the hierarchy is given as aggregation groups over bottom series (or as a 0/1 summing matrix). The base forecasts and
any in-sample residuals come from a saved run, read through series_io exactly as the other tools read their series, one
named list per node. ``forecast.reconcile`` does the algebra. This module reads the inputs with clear reasons, reports each
node's reconciled forecast, and checks coherence on the output values themselves, not on the algebra.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from prophet_agent.forecast import reconcile as rc
from prophet_agent.runtime.runner import Tool
from prophet_agent.tools.series_io import resolve_series

ALLOWED = {"source", "hierarchy", "method", "residuals"}
HIERARCHY_KEYS = {"bottom", "groups", "nodes", "matrix"}
METHODS = rc.METHODS + ("bottom_up",)
OBSERVE_NODES = 20      # nodes the planner sees; the ledger keeps all of them


def _names(value: Any, what: str) -> list[str]:
    if not isinstance(value, list) or not value or not all(isinstance(v, str) and v for v in value):
        raise ValueError(f"{what} must be a non-empty list of names")
    if len(set(value)) != len(value):
        raise ValueError(f"{what} must not repeat a name")
    return list(value)


def _hierarchy(h: Any) -> tuple[list[str], list[str], np.ndarray]:
    """(node names in row order, bottom series names in column order, summing matrix S with nodes x bottom)."""
    if not isinstance(h, dict):
        raise ValueError("hierarchy must be an object: {bottom: [...], groups: {...}} or {nodes, bottom, matrix}")
    extra = sorted(set(h) - HIERARCHY_KEYS)
    if extra:
        raise ValueError(f"hierarchy has unknown keys {extra}; allowed: bottom, groups, nodes, matrix")
    if ("groups" in h) == ("matrix" in h):
        raise ValueError("hierarchy needs groups or a summing matrix: give one of them, not both and not neither")
    bottom = _names(h.get("bottom"), "hierarchy bottom")
    if "groups" in h:
        groups = h["groups"]
        if not isinstance(groups, dict) or not groups:
            raise ValueError("groups must be an object mapping each aggregate name to the bottom series it sums")
        for g, members in groups.items():
            if g in bottom:
                raise ValueError(f"group {g!r} has the same name as a bottom series; give the aggregate another name")
            members = _names(members, f"group {g!r}")
            unknown = [m for m in members if m not in bottom]
            if unknown:
                raise ValueError(f"group {g!r} names unknown bottom series {unknown}; bottom has {bottom}")
        nodes = list(groups) + bottom
        S = np.zeros((len(nodes), len(bottom)))
        for i, g in enumerate(groups):
            for m in groups[g]:
                S[i, bottom.index(m)] = 1.0
        for j, b in enumerate(bottom):
            S[len(groups) + j, j] = 1.0
        return nodes, bottom, S
    nodes = _names(h.get("nodes"), "hierarchy nodes")
    matrix = h.get("matrix")
    if not isinstance(matrix, list) or len(matrix) != len(nodes) or not all(
            isinstance(r, list) and len(r) == len(bottom) for r in matrix):
        raise ValueError(f"matrix must have one row per node ({len(nodes)}) and one column per bottom series ({len(bottom)})")
    S = np.asarray(matrix, dtype=float)
    if not np.isin(S, (0.0, 1.0)).all():
        raise ValueError("matrix must hold only 0 and 1 entries")
    missing = [b for b in bottom if b not in nodes]
    if missing:
        raise ValueError(f"bottom series {missing} must also be rows of the matrix (nodes)")
    return nodes, bottom, S


def _run(args: dict, _key: str) -> dict[str, Any]:
    if args.get("source") is None or "values" in args:
        raise ValueError("reconcile_forecasts needs source: {run_id, file} pointing at a series.json with one named "
                         "forecast per node (write them with save_series)")
    extra = sorted(set(args) - ALLOWED)
    if extra:
        raise ValueError(f"unknown arguments {extra}; allowed: source, hierarchy, method, residuals")
    if "hierarchy" not in args:
        raise ValueError("reconcile_forecasts needs hierarchy: the bottom series and the aggregation groups")
    method = args.get("method", rc.DEFAULT_METHOD)
    if method not in METHODS:
        raise ValueError(f"method must be one of {', '.join(METHODS)}")
    nodes, bottom, S = _hierarchy(args["hierarchy"])
    rc._check_summing(S)                                             # the same checks as the forecast module

    base_series = dict(resolve_series({"source": args["source"]}))
    absent = [n for n in nodes if n not in base_series]
    if absent:
        raise ValueError(f"no base forecast for node(s) {absent} in the file; it holds {sorted(base_series)[:20]}")
    horizons = {len(base_series[n]) for n in nodes}
    if len(horizons) != 1:
        raise ValueError("all base forecasts must have the same horizon (number of steps)")
    Y = np.column_stack([base_series[n] for n in nodes])             # (horizon, nodes)
    warnings: list[str] = []
    ignored = sorted(set(base_series) - set(nodes))
    if ignored:
        warnings.append(f"series in the file but not in the hierarchy were ignored: {ignored[:10]}")

    residuals = None
    if args.get("residuals") is not None:
        res_series = dict(resolve_series({"source": args["residuals"]}))
        miss = [n for n in nodes if n not in res_series]
        if miss:
            raise ValueError(f"residuals have no series for node(s) {miss}")
        lengths = {len(res_series[n]) for n in nodes}
        if len(lengths) != 1:
            raise ValueError("all residual series must have the same length")
        residuals = np.column_stack([res_series[n] for n in nodes])
        if method != "mint_shrink":
            warnings.append(f"residuals are used only by mint_shrink; {method} ignores them")

    bottom_rows = [nodes.index(b) for b in bottom]
    base_gap = float(np.max(np.abs(Y - Y[:, bottom_rows] @ S.T)))

    if method == "bottom_up":
        out = rc.bottom_up(S, Y)
        reconciled, used, shrink = out["reconciled"], "bottom_up", None
    else:
        out = rc.reconcile(S, Y, residuals=residuals, method=method)
        reconciled, used, shrink = out["reconciled"], out["method_used"], out["shrinkage"]
        warnings.extend(out["warnings"])

    gap = float(np.max(np.abs(reconciled - reconciled[:, bottom_rows] @ S.T)))
    scale = max(1.0, float(np.max(np.abs(reconciled))))
    tol = 1e-6 * scale
    negative = int((reconciled < 0).sum())
    if negative and (Y >= 0).all():
        warnings.append(f"{negative} reconciled value(s) fall below zero although every base forecast is zero or more; "
                        "the adjustment created them, so check the hierarchy and method before using them")
    return {
        "method_used": used,
        "shrinkage": None if shrink is None else round(float(shrink), 4),
        "warnings": warnings,
        "horizon": int(Y.shape[0]),
        "nodes": {n: [round(float(v), 6) for v in reconciled[:, i]] for i, n in enumerate(nodes)},
        "hierarchy": {"nodes": len(nodes), "bottom_series": len(bottom)},
        "coherence": {"coherent_after": gap <= tol, "max_gap_after": round(gap, 9),
                      "base_coherent": base_gap <= 1e-6 * max(1.0, float(np.max(np.abs(Y)))),
                      "base_gap": round(base_gap, 6), "tolerance": tol},
        "basis": "inferred: arithmetic on the saved base forecasts; reconciled numbers are not an independent forecast",
    }


def observe(result: dict) -> dict:
    keep = ("method_used", "shrinkage", "warnings", "horizon", "hierarchy", "coherence", "basis")
    out = {k: result[k] for k in keep if k in result}
    rows = list(result["nodes"].items())
    out["nodes"] = dict(rows[:OBSERVE_NODES])
    if len(rows) > OBSERVE_NODES:
        out["nodes_note"] = f"first {OBSERVE_NODES} of {len(rows)} nodes shown; the coherence check covers all of them"
    return out


def reconcile_forecasts_tool() -> Tool:
    return Tool("reconcile_forecasts", lambda a, k: _run(a, k), idempotent=True, observe=observe)


RECONCILE_FORECASTS_SPEC = {
    "name": "reconcile_forecasts",
    "title": "Reconciled forecasts",
    "activity": "Making a set of forecasts add up across a hierarchy",
    "description": "Make forecasts of a hierarchy add up, so each parent equals the sum of its children. Give the bottom series "
                   "and the aggregation groups (or a summing matrix), and the base forecast of every node from a saved run "
                   "(one named series per node, written with save_series). Returns the reconciled forecast of each node, the "
                   "method used (MinT with residuals, otherwise WLS-structural), any warnings and a coherence check. Use it when "
                   "separate forecasts of a total, its parts and its products disagree.",
    "args": {"source": "object {run_id: id of the analysis run that wrote the file, file: 'out/forecasts.json'} holding one "
                       "named forecast per node (required)",
             "hierarchy": "object {bottom: [bottom series names], groups: {node name: [bottom names it sums]}}, or {nodes: "
                          "[names], bottom: [names], matrix: 0/1 rows by bottom columns}",
             "method": "optional, one of mint_shrink (default), wls_structural, ols, bottom_up",
             "residuals": "optional object {run_id, file} holding one named list of in-sample residuals per node, all the same "
                          "length, used by mint_shrink"},
}
