"""Round 6m: panel_backtest picks its entities from a large file.

Probe (before the fix): a file of 150 named series could not be backtested on 3 of them. The file's size limit (100 series)
was checked before the entities were picked, and the refusal said "Pick some with source.name", which this tool does not
take. The tool also accepted up to 200 entities, which the file reader would never pass.
"""
from __future__ import annotations

import json

import pytest

from prophet_agent.tools import panel_tool, series_io


class _Reached(Exception):
    """The call got past validation and reached the panel model."""


@pytest.fixture()
def root(tmp_path, monkeypatch):
    monkeypatch.setattr(series_io, "SANDBOX_ROOT", tmp_path)
    return tmp_path


def _panel_file(root, run_id, n):
    series = {f"s{i:03d}": [float(i + t) for t in range(30)] for i in range(n)}
    (root / run_id / "out").mkdir(parents=True)
    (root / run_id / "out" / "series.json").write_text(json.dumps({"series": series}))


def test_entities_are_picked_from_a_file_larger_than_the_limit(root, monkeypatch):
    rid = "3" * 32
    _panel_file(root, rid, 150)
    seen = {}

    def fake_panel_backtest(panel, horizon, **kwargs):
        seen["names"] = list(panel)
        raise _Reached()

    monkeypatch.setattr(panel_tool.global_gbm, "panel_backtest", fake_panel_backtest)
    args = {"source": {"run_id": rid, "file": "out/series.json"}, "horizon": 3,
            "entities": ["s007", "s002", "s100"]}
    with pytest.raises(_Reached):
        panel_tool._run(args, "k", engine_factory=lambda: object())
    assert seen["names"] == ["s007", "s002", "s100"]


def test_entities_cannot_exceed_the_per_call_limit(root):
    rid = "4" * 32
    _panel_file(root, rid, 150)
    args = {"source": {"run_id": rid, "file": "out/series.json"}, "horizon": 3,
            "entities": [f"s{i:03d}" for i in range(101)]}
    with pytest.raises(ValueError, match="entities must be a list of 1 to 100"):
        panel_tool._run(args, "k", engine_factory=lambda: object())


def test_an_unknown_entity_is_named_with_what_the_file_holds(root):
    rid = "5" * 32
    _panel_file(root, rid, 5)
    args = {"source": {"run_id": rid, "file": "out/series.json"}, "horizon": 3, "entities": ["s001", "nope"]}
    with pytest.raises(ValueError, match="nope"):
        panel_tool._run(args, "k", engine_factory=lambda: object())
