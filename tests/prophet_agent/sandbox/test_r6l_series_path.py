"""Round 6l, audit item 6: every reader of a ``source`` file stays inside that run's out/ folder.

Probe (before the fix): source_covariates() and run_dates() opened an absolute file name given as source.file, so a JSON file
elsewhere on disk supplied covariates and dates. resolve_series() already refused it. Test files live in pytest's temp folder.
"""
from __future__ import annotations

import json
import uuid

import pytest

from prophet_agent.tools import series_io


@pytest.fixture
def planted(tmp_path):
    path = tmp_path / "planted.json"
    path.write_text(json.dumps({"covariates": {"leak": [1, 2, 3]}, "dates": {"series": ["2020-01-01"]},
                                "series": [1, 2, 3]}))
    return path


@pytest.fixture
def run_with_file(tmp_path, monkeypatch):
    """A run folder under a temp sandbox root, holding out/series.json with covariates and dates."""
    monkeypatch.setattr(series_io, "SANDBOX_ROOT", tmp_path / "sandbox")
    run_id = uuid.uuid4().hex
    out = tmp_path / "sandbox" / run_id / "out"
    out.mkdir(parents=True)
    (out / "series.json").write_text(json.dumps({"covariates": {"promo": [0, 1]}, "dates": {"series": ["2021-01-01"]},
                                                 "series": [5, 6]}))
    return run_id


def _args(run_id, file):
    return {"source": {"run_id": run_id, "file": file}}


def test_an_absolute_path_outside_the_run_is_not_read_for_covariates_or_dates(planted, run_with_file):
    args = _args(run_with_file, str(planted))
    assert series_io.source_covariates(args) == {}
    assert series_io.run_dates(args, "series") is None


def test_a_parent_folder_path_outside_the_run_is_not_read(planted, run_with_file):
    rel = "../../../../../../../../" + str(planted).lstrip("/")
    args = _args(run_with_file, rel)
    assert series_io.source_covariates(args) == {}
    assert series_io.run_dates(args, "series") is None
    with pytest.raises(ValueError):
        series_io.resolve_series(args)


def test_a_file_inside_out_is_still_read_by_every_reader(run_with_file):
    args = _args(run_with_file, "out/series.json")
    assert series_io.source_covariates(args) == {"promo": [0, 1]}
    assert series_io.run_dates(args, "series") == ["2021-01-01"]
    assert [name for name, _ in series_io.resolve_series(args)] == ["series"]
