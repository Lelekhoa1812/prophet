"""Overlap between two accepted sources. The observed column is copied unchanged.

Motivation vs Logic
Motivation: an adjustment is useful only when a reader can still see the figure that was fetched.
Logic: the median relative gap on shared times is the report. A calibrated column is new and marked derived.
"""
from __future__ import annotations


def calibrate_overlap(left: list[dict], right: list[dict], *, time_column: str, measure_column: str,
                      source_left: str, source_right: str) -> dict:
    """``left`` is the observed series. ``right`` is the comparison. Both are lists of row dicts."""
    right_by_time = {str(row.get(time_column, "")).strip(): row.get(measure_column) for row in right}
    gaps = []
    rows = []
    shared = 0
    for row in left:
        key = str(row.get(time_column, "")).strip()
        observed = row.get(measure_column)
        other = right_by_time.get(key)
        item = {"time": key, measure_column: observed, "origin": "observed"}
        if other is not None and key:
            shared += 1
            gap = _relative(observed, other)
            if gap is not None:
                gaps.append(gap)
                item["overlap_gap"] = gap
        rows.append(item)
    median = _median(gaps)
    if median is not None:
        for item in rows:
            observed = item.get(measure_column)
            try:
                base = float(observed)
            except (TypeError, ValueError):
                continue
            item[f"calibrated_{measure_column}"] = base * (1 + median)
            item["calibrated_origin"] = "derived"
    return {
        "rows": rows,
        "shared_times": shared,
        "median_relative_difference": median,
        "source_left": source_left,
        "source_right": source_right,
        "formula": f"calibrated_{measure_column} = {source_left} * (1 + median relative gap versus {source_right})",
        "observed_column": measure_column,
    }


def _relative(left, right) -> float | None:
    try:
        a, b = float(left), float(right)
    except (TypeError, ValueError):
        return None
    scale = max(abs(a), 1e-12)
    return (b - a) / scale


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2
