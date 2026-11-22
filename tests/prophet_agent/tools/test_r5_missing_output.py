"""Round 5: a missing carried or source file says whether the run exists and, when it does, what it wrote.

Live electricity rep4 (measured, 2026-11-23): three calls named files their runs never wrote, one with a run id that no tool
had returned. The old message, "no file ... in the outputs of run ...", did not say which mistake it was.
"""
from __future__ import annotations

import pytest

from prophet_agent.tools import sandbox_tool, series_io


@pytest.fixture()
def root(tmp_path, monkeypatch):
    monkeypatch.setattr(series_io, "SANDBOX_ROOT", tmp_path)
    monkeypatch.setattr(sandbox_tool, "SANDBOX_ROOT", tmp_path)
    return tmp_path


def test_an_unknown_run_id_is_named_as_one(root):
    with pytest.raises(ValueError, match="no run a5c1e8b7f3d24e1a9c8d7e6f5a4b3c2d exists"):
        series_io.resolve_series({"source": {"run_id": "a5c1e8b7f3d24e1a9c8d7e6f5a4b3c2d", "file": "out/series.json"}})


def test_a_real_run_lists_the_files_it_wrote(root):
    rid = "0" * 32
    (root / rid / "out").mkdir(parents=True)
    (root / rid / "out" / "panel.json").write_text("{}")
    with pytest.raises(ValueError, match=r"that run wrote out/panel\.json"):
        series_io.source_path({"source": {"run_id": rid, "file": "out/series.json"}})
    with pytest.raises(ValueError, match=r"that run wrote out/panel\.json"):
        sandbox_tool.carried_inputs([{"run_id": rid, "file": "out/scan.json"}])


def test_a_run_with_no_outputs_says_so(root):
    rid = "1" * 32
    (root / rid).mkdir()
    with pytest.raises(ValueError, match="that run wrote no files"):
        series_io.source_path({"source": {"run_id": rid}})
