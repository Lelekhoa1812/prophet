"""Argument aliases, explained errors and the ledger's entries (prophet_pair, prophet_stats, prophet_lib, prophet_scan).

Motivation vs Logic
Motivation: the planner guessed keyword names (save_series(name=), hampel(n_sigma=), changepoints(min_segment=), scan(time=))
and read keys that do not exist ('missing_steps', 'blockers', Ledger.entries, flags.get). Each guess cost a failed cell.
Logic: documented aliases map the guessed name to the real one. Any other unknown keyword raises a TypeError with the real
signature. A missing report key raises KeyError with the real keys. Dict-style reads on an array explain what it is.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from prophet_agent.sandbox import prophet_lib as pl
from prophet_agent.sandbox import prophet_scan as sc
from prophet_agent.sandbox import prophet_stats as ps
from prophet_agent.sandbox.prophet_pair import ExplainedArray, ReportDict


def _spiky(n: int = 120) -> np.ndarray:
    x = np.sin(np.arange(n) / 5.0)
    x[40] = 40.0
    return x


def _level_shift(n: int = 200) -> np.ndarray:
    rng = np.random.default_rng(3)
    return np.r_[rng.normal(0, 1, n // 2), rng.normal(5, 1, n // 2)]


# -- keyword aliases --------------------------------------------------------------------------------------------------

def test_hampel_accepts_n_sigma_and_threshold_as_n_sigmas():
    x = _spiky()
    base_flags, base_rep = ps.hampel(x, n_sigmas=3.0)
    for alias in ("n_sigma", "threshold"):
        flags, rep = ps.hampel(x, **{alias: 3.0})
        assert np.array_equal(flags, base_flags) and rep["n_sigmas"] == 3.0


def test_changepoints_maps_min_segment_and_max_cps_and_ignores_method():
    x = _level_shift()
    real, real_rep = ps.changepoints(x, min_size=30, max_changes=5)
    aliased, alias_rep = ps.changepoints(x, min_segment=30, max_cps=5, method="binary")
    assert [c["index"] for c in real] == [c["index"] for c in aliased]
    assert alias_rep["min_size"] == 30 and alias_rep["max_changes"] == 5


def test_stl_outliers_accepts_robust_as_a_no_op():
    x = _spiky(160)
    flags, rep = ps.stl_outliers(x, period=7, robust=True)
    assert rep["period"] == 7 and rep["causal"] is False


def test_scan_accepts_time_and_entity_col_as_the_real_names():
    df = pd.DataFrame({"d": pd.date_range("2024-01-01", periods=60), "v": range(60), "sku": ["a"] * 60})
    r = sc.scan(df, time="d", value="v", entity_col="sku")
    assert r.summary["entity_cols"] == ["sku"]


def test_save_series_name_labels_the_series_and_load_series_returns_it(tmp_path):
    path = str(tmp_path / "series.json")
    s = pd.Series([1.0, 2.0, 3.5], index=pd.date_range("2024-01-01", periods=3), name="x")
    pl.save_series(s, path, name="Beijing")
    back = pl.load_series(path)
    assert back.name == "Beijing" and list(back.values) == [1.0, 2.0, 3.5]
    assert list(pl.load_series(path, name="Beijing").values) == [1.0, 2.0, 3.5]


def test_save_series_name_with_a_dict_is_refused(tmp_path):
    with pytest.raises(ValueError, match="dict keys are the names"):
        pl.save_series({"a": [1.0, 2.0]}, str(tmp_path / "x.json"), name="b")


def test_an_unknown_keyword_raises_a_type_error_with_the_real_signature():
    with pytest.raises(TypeError, match=r"unexpected keyword argument.*bogus.*hampel\(x, window=7, n_sigmas=3.0, causal=True\)"):
        ps.hampel(_spiky(), bogus=1)


def test_giving_an_alias_and_its_real_name_together_is_refused():
    with pytest.raises(TypeError, match="both 'threshold' and 'n_sigmas'"):
        ps.hampel(_spiky(), threshold=3.0, n_sigmas=3.0)


# -- explained arrays ---------------------------------------------------------------------------------------------------

def test_get_on_a_returned_array_raises_the_same_explanation_as_a_text_key():
    flags, _ = ps.hampel(_spiky())
    assert isinstance(flags, ExplainedArray)
    with pytest.raises(TypeError, match=r"hampel\(\) returned a boolean array of flags.*report keys"):
        flags.get("outlier_indices")
    with pytest.raises(TypeError, match=r"hampel\(\) returned a boolean array of flags"):
        flags["is_outlier"]


# -- ledger and report keys ---------------------------------------------------------------------------------------------

def test_ledger_entries_is_a_read_only_list_of_dicts():
    led = ps.Ledger()
    assert led.entries == []
    led.add("impute", "two gaps", "linear")
    entries = led.entries
    assert entries == [{"index": 1, "stage": "impute", "finding": "two gaps", "decision": "linear", "evidence": None}]
    entries.append({"junk": True})                      # a copy: the ledger does not change
    assert len(led.entries) == 1
    with pytest.raises(AttributeError):
        led.entries = []


def test_a_missing_report_key_names_the_real_keys():
    s, rep = pl.prepare_series(pd.DataFrame({"d": pd.date_range("2024-01-01", periods=30).strftime("%Y-%m-%d"),
                                              "v": np.arange(30.0)}), value="v", time="d")
    assert isinstance(rep, ReportDict)
    with pytest.raises(KeyError, match="its keys are"):
        rep["no_such_key"]
    assert rep["missing_steps"] == rep["filled_periods"] == 0


def test_the_scan_summary_answers_blockers_and_names_its_keys_otherwise():
    df = pd.DataFrame({"d": pd.date_range("2024-01-01", periods=60), "v": range(60)})
    r = sc.scan(df, "d", "v")
    summary = r["summary"]
    assert summary["blockers"] == summary["blocked"] is False     # no blocker in this clean table
    with pytest.raises(KeyError, match="severity_counts"):
        summary["no_such_key"]
