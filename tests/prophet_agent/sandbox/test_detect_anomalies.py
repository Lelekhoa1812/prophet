"""detect_anomalies: one return schema on every path, with candidates, values and scores (prophet_lib, and the tool).

Motivation vs Logic
Motivation: the spec promised `candidates`, the code returned indices and scores, and the short-series path returned
neither values nor scores. Models read anomalies["candidates"] and series["values"] and got KeyError.
Logic: every path returns the same keys. candidates is a list of {index, date, value, score, rule}; date is filled when
dates are given, and is None otherwise.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from prophet_agent.sandbox import prophet_lib as pl
from prophet_agent.tools.drivers_tool import _observe_detect

REQUIRED = {"candidates", "values", "scores", "indices", "ignore_ranges", "method", "threshold", "points"}


def _series(n: int = 120, spike_at: int = 60) -> np.ndarray:
    rng = np.random.default_rng(4)
    x = 10 + np.sin(np.arange(n) * 2 * np.pi / 7) + rng.normal(0, 0.1, n)
    x[spike_at] = 60.0
    return x


def test_the_short_path_has_the_same_keys_as_the_full_path():
    short = pl.detect_anomalies([1.0, 2.0, 3.0])
    full = pl.detect_anomalies(_series())
    assert REQUIRED <= set(short) and REQUIRED <= set(full)
    assert short["candidates"] == [] and short["values"] == [] and short["scores"] == []


def test_a_spike_is_a_candidate_with_index_value_score_and_rule():
    res = pl.detect_anomalies(_series(), season=7)
    assert 60 in [c["index"] for c in res["candidates"]]
    cand = next(c for c in res["candidates"] if c["index"] == 60)
    assert cand["value"] == 60.0 and cand["score"] > 6
    assert cand["rule"] == "robust_z_cycle" and cand["date"] is None
    assert set(cand) == {"index", "date", "value", "score", "rule"}


def test_candidates_run_alongside_indices_values_and_scores():
    res = pl.detect_anomalies(_series())
    assert [c["index"] for c in res["candidates"]] == res["indices"]
    assert [c["value"] for c in res["candidates"]] == res["values"]
    assert [c["score"] for c in res["candidates"]] == res["scores"]


def test_dates_fill_the_date_field_when_given_and_match_the_length():
    x = _series()
    dates = pd.date_range("2024-01-01", periods=len(x), freq="D")
    res = pl.detect_anomalies(x, dates=dates)
    cand = next(c for c in res["candidates"] if c["index"] == 60)
    assert cand["date"] == str(dates[60])
    wrong_length = pl.detect_anomalies(x, dates=dates[:10])
    assert all(c["date"] is None for c in wrong_length["candidates"])


def test_the_tool_observes_the_candidates_list_on_every_path():
    short = pl.detect_anomalies([1.0] * 5)
    out = _observe_detect({"run_id": "r", "series": "series", "points": 5, **short, "dated": False})
    assert out["candidates"] == [] and out["candidate_count"] == 0
