"""Round 3 load_series (A1-07): run_id and carry were accepted and ignored, so a call that said "this run's copy" could
silently read the mounted copy.

Motivation vs Logic: 29 of 33 load_series calls in the six live runs passed run_id= (measured-live). The mount is the
only copy this run can read, so the call now refuses and says where the earlier output comes from.
"""
from __future__ import annotations

import pandas as pd
import pytest

from prophet_agent.sandbox import prophet_lib as pl


def _mounted(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "in" / "carried").mkdir(parents=True)
    pl.save_series(pd.Series([1.0, 2.0, 3.0], index=pd.date_range("2024-01-01", periods=3), name="x"),
                   "in/carried/series.json")


def test_load_series_refuses_run_id_and_teaches_carry(tmp_path, monkeypatch):
    _mounted(tmp_path, monkeypatch)
    with pytest.raises(TypeError, match="carry=") as err:
        pl.load_series("in/carried/series.json", run_id="e712f00a23b94b319c966cca95322d0b")
    assert "in/carried" in str(err.value)


def test_load_series_refuses_carry_and_teaches_the_sandbox_argument(tmp_path, monkeypatch):
    _mounted(tmp_path, monkeypatch)
    with pytest.raises(TypeError, match="sandbox call"):
        pl.load_series("in/carried/series.json", carry=[{"run_id": "abc", "file": "out/series.json"}])


def test_load_series_without_run_id_still_reads_the_mounted_copy(tmp_path, monkeypatch):
    _mounted(tmp_path, monkeypatch)
    assert list(pl.load_series("in/carried/series.json").values) == [1.0, 2.0, 3.0]
