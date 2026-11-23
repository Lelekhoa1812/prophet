"""Round 6m: malformed inputs are refused with a message the model can act on, not a bare TypeError or a silent value.

Probes (before the fixes): a list or dict as the file name of inspect_data raised "unhashable type: 'list'"; a dict as
source.name raised the same; a series with nested lists was accepted as a two-dimensional array; a driver window given as
the integer 2026 was accepted as the date 1970-01-01.
"""
from __future__ import annotations

import pytest

from prophet_agent.tools import data_tools


def test_inspect_data_refuses_a_file_name_that_is_not_text(tmp_path):
    path = tmp_path / "a.csv"
    path.write_text("date,value\n2026-01-01,1\n")
    run = data_tools.make_inspect({"a.csv": path})
    for bad in (["a.csv"], {"name": "a.csv"}, 5):
        with pytest.raises(ValueError, match="file must be one of"):
            run({"file": bad}, "k")


@pytest.fixture()
def sandbox(tmp_path, monkeypatch):
    from prophet_agent.tools import series_io
    monkeypatch.setattr(series_io, "SANDBOX_ROOT", tmp_path)
    return tmp_path


def _panel_file(root, run_id, doc):
    import json
    (root / run_id / "out").mkdir(parents=True)
    (root / run_id / "out" / "series.json").write_text(json.dumps(doc))


def test_values_must_be_one_flat_list_of_numbers():
    from prophet_agent.tools import series_io
    with pytest.raises(ValueError, match="one flat list of numbers"):
        series_io._clean("series", [[1, 2], [3, 4]])
    with pytest.raises(ValueError, match="one flat list of numbers"):
        series_io._clean("series", [{"v": 1}, 2])
    with pytest.raises(ValueError, match="not numbers"):
        series_io._clean("series", ["1", "2"])
    with pytest.raises(ValueError, match="not numbers"):
        series_io._clean("series", [1, True, 3])
    assert series_io._clean("series", [1, 2.5, None][:2]).tolist() == [1.0, 2.5]
    with pytest.raises(ValueError, match="missing or non-finite"):
        series_io._clean("series", [1, None, 3])


def test_a_series_name_that_is_not_text_is_refused_with_the_fix(sandbox):
    from prophet_agent.tools import series_io
    rid = "2" * 32
    _panel_file(sandbox, rid, {"series": {"a": [1, 2, 3], "b": [4, 5, 6]}})
    for bad in ({"a": 1}, 5):
        with pytest.raises(ValueError, match="source.name is one series name"):
            series_io.resolve_series({"source": {"run_id": rid, "file": "out/series.json", "name": bad}})
    named = series_io.resolve_series({"source": {"run_id": rid, "file": "out/series.json", "name": "a"}})
    assert [n for n, _ in named] == ["a"]
