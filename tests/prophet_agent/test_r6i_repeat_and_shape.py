"""Round 6i (2026-11-23): an identical sandbox cell is answered with its earlier result, and a written JSON file names its keys.

Live Beijing e14af4bd (build 7ab9fbe): twelve sandbox cells in a row ran the same code again after another cell had run,
because the repeat rule re-ran any call once a sandbox cell came after it. The planner also guessed a scan's keys (per_entity)
three times, because nothing said what the file held. Every series here is synthetic.
"""
from __future__ import annotations

import json

from prophet_agent.runtime import goal as G
from prophet_agent.tools import sandbox_tool as S


def _tool(name, args):
    return {"kind": "tool", "tool": name, "args": args, "result": {"ok": True}}


def test_an_identical_sandbox_cell_after_another_cell_is_answered_with_a_pointer():
    a = {"code": "print(1)", "carry": []}
    history = [_tool("sandbox_python", a), _tool("sandbox_python", {"code": "print(2)", "carry": []})]
    assert G._repeat_of("sandbox_python", a, history) == 1


def test_a_read_tool_after_a_sandbox_cell_is_still_run_again():
    # A forecast reads files a sandbox cell may have written, so the earlier result is not reused.
    args = {"source": {"run_id": "r", "file": "out/series.json"}}
    history = [_tool("forecast_report", args), _tool("sandbox_python", {"code": "print(2)", "carry": []})]
    assert G._repeat_of("forecast_report", args, history) is None


def test_a_repeat_with_no_cell_between_is_pointed_to_for_any_tool():
    args = {"source": {"run_id": "r", "file": "out/series.json"}}
    history = [_tool("forecast_report", args)]
    assert G._repeat_of("forecast_report", args, history) == 1


def test_json_shape_names_the_top_level_keys_and_their_kinds(tmp_path):
    p = tmp_path / "scan.json"
    p.write_text(json.dumps({"findings": [{"id": "x", "severity": "major", "scope": "Dongsi"}],
                             "summary": {"rows_in": 17467, "entities_total": 12}}))
    shape = S.json_shape(p)
    assert shape == ("top-level keys: findings: list of 1 dicts, each with keys id, severity, scope; "
                     "summary: dict of 2 keys")


def test_json_shape_of_a_top_level_list_names_its_length(tmp_path):
    p = tmp_path / "rows.json"
    p.write_text(json.dumps([1, 2, 3]))
    assert S.json_shape(p) == "list of 3"


def test_json_shape_is_not_given_for_a_file_too_large_to_read(tmp_path, monkeypatch):
    p = tmp_path / "big.json"
    p.write_text(json.dumps({"a": 1}))
    monkeypatch.setattr(S, "SHAPE_MAX_BYTES", 3)
    assert S.json_shape(p) is None


def test_json_shape_is_not_given_for_text_that_is_not_json(tmp_path):
    p = tmp_path / "notes.json"
    p.write_text("not json")
    assert S.json_shape(p) is None


def test_observe_names_the_shape_of_each_written_json_file_and_nothing_else(tmp_path, monkeypatch):
    monkeypatch.setattr(S, "SANDBOX_ROOT", tmp_path)
    run = "a" * 32
    (tmp_path / run / "out").mkdir(parents=True)
    (tmp_path / run / "out" / "scan.json").write_text(json.dumps({"findings": [], "summary": {}}))
    (tmp_path / run / "out" / "table.csv").write_text("a,b\n1,2\n")
    result = {"ok": True, "run_id": run, "stdout": "done", "stderr": "", "timed_out": False, "returncode": 0,
              "outputs": [{"path": "out/scan.json", "bytes": 30}, {"path": "out/table.csv", "bytes": 8}]}
    out = S.observe(result)
    assert out["shapes"] == {"out/scan.json": "top-level keys: findings: list of 0; summary: dict of 0 keys"}
