"""Round 6m: the tools declared idempotent give the same answer when called twice, and a finished call is not run again.

Each test runs one declared-idempotent tool twice with the same arguments and key, then checks the second answer. Nothing
here uses the network or the TimesFM model: the sandbox runs local Python only, and the forecast engine is replaced.
"""
from __future__ import annotations

import importlib
import json

import pytest

from prophet_agent.tools.notes_tool import notes_tool

sandbox_tool = importlib.import_module("prophet_agent.tools.sandbox_tool")
runner = importlib.import_module("prophet_agent.sandbox.runner")
data_tools = importlib.import_module("prophet_agent.tools.data_tools")
notebook_tools = importlib.import_module("prophet_agent.tools.notebook_tools")


class _Store:
    def __init__(self):
        self.notes: dict[tuple[str, str], str] = {}

    def note_get(self, mission_id, scope, name):
        return self.notes.get((scope, name))

    def note_put(self, mission_id, scope, name, text, append=False):
        body = (self.notes[(scope, name)] + "\n" + text) if append and (scope, name) in self.notes else text
        self.notes[(scope, name)] = body
        return len(body)

    def note_list(self, mission_id, scope=None):
        return [{"scope": s, "name": n} for (s, n) in self.notes if scope in (None, s)]


def test_notes_write_twice_leaves_one_copy_and_says_so_the_second_time():
    store = _Store()
    tool = notes_tool(store, "m1", "main")
    first = tool.fn({"action": "write", "name": "plan", "text": "keep the 80 percent band"}, "k1")
    second = tool.fn({"action": "write", "name": "plan", "text": "keep the 80 percent band"}, "k1")
    assert store.notes == {("main", "plan"): "keep the 80 percent band"}
    assert first["chars"] == second["chars"]
    assert second.get("unchanged") is True


def test_notes_append_retried_after_a_crash_is_not_written_twice():
    store = _Store()
    tool = notes_tool(store, "m1", "main")
    tool.fn({"action": "append", "name": "log", "text": "finding A"}, "k1")
    retry = tool.fn({"action": "append", "name": "log", "text": "finding A"}, "k1")
    assert store.notes[("main", "log")] == "finding A"
    assert retry.get("unchanged") is True


@pytest.fixture()
def sandbox_root(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "SANDBOX_ROOT", tmp_path)
    return tmp_path


def test_sandbox_python_with_the_same_key_returns_the_finished_run_without_running_it_again(sandbox_root):
    key = "9" * 32
    code = ("import pathlib\n"
            "pathlib.Path('out').mkdir(exist_ok=True)\n"
            "with open('out/log.txt', 'a') as f:\n"
            "    f.write('x')\n"
            "print('ran once')\n")
    run = sandbox_tool.make_run(None)
    first = run({"code": code}, key)
    second = run({"code": code}, key)
    assert first["run_id"] == second["run_id"] == key
    assert second["stdout"] == first["stdout"]
    assert (sandbox_root / key / "out" / "log.txt").read_text() == "x"


def test_inspect_data_twice_gives_the_same_profile(tmp_path, sandbox_root):
    csv = tmp_path / "sales.csv"
    csv.write_text("date,value\n2026-01-01,1\n2026-01-02,2\n2026-01-03,3\n")
    inspect = data_tools.make_inspect({"sales.csv": csv})
    first = inspect({"file": "sales.csv"}, "k1")
    second = inspect({"file": "sales.csv"}, "k2")
    assert first["profile"] == second["profile"]
    assert first["profile"]["rows"] == second["profile"]["rows"] == 3


def test_save_table_twice_is_the_same_promotion(monkeypatch):
    calls = []

    class _Notebooks:
        def promote(self, nid, run_id, file, name, description="", turn_id=None):
            calls.append((nid, run_id, file, name, description))
            return {"name": name, "version": 1, "rows": 3, "columns": ["a", "b"], "unchanged": len(calls) > 1}

    binding = notebook_tools.NotebookBinding(notebooks=_Notebooks(), notebook_id="n1", turn_id="t1")
    save = notebook_tools.make_save(binding)
    args = {"run_id": "e" * 32, "file": "out/t.csv", "name": "t", "description": "the table"}
    first = notebook_tools.observe(save(args, "k1"))
    second = notebook_tools.observe(save(args, "k1"))
    assert calls[0] == calls[1]
    assert first["version"] == second["version"] and second.get("unchanged") is True


def test_observers_are_pure_over_the_same_result():
    result = {"horizon": 3, "points": 100, "period": 7, "fold_count": 2, "calls_spent": 4, "recommended": "default",
              "baseline_beats_default": False, "table": [], "skipped": [], "failed": [], "caveats": ["one"]}
    lab = importlib.import_module("prophet_agent.tools.lab_tool")
    assert json.dumps(lab.observe(result), sort_keys=True) == json.dumps(lab.observe(result), sort_keys=True)
    assert result["caveats"] == ["one"], "an observer must not change the stored result"
