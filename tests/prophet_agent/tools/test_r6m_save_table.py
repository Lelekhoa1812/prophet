"""Round 6m: save_table refuses a description longer than the notebook keeps.

Probe (before the fix): promote cut a 400-character description to 300 characters with no word to the caller, so the saved
table carried a description that ended mid-sentence and the planner was not told.
"""
from __future__ import annotations

import importlib

import pytest

nt = importlib.import_module("prophet_agent.tools.notebook_tools")


class _Notebooks:
    def __init__(self):
        self.calls = []

    def promote(self, nid, run_id, file, name, description="", turn_id=None):
        self.calls.append(description)
        return {"name": name, "version": 1, "rows": 3, "columns": ["a"]}


def _save(description):
    nb = _Notebooks()
    binding = nt.NotebookBinding(notebooks=nb, notebook_id="n1", turn_id="t1")
    run = nt.make_save(binding)
    args = {"run_id": "d" * 32, "file": "out/t.csv", "name": "t"}
    if description is not None:
        args["description"] = description
    return run(args, "k"), nb


def test_a_description_over_the_limit_is_refused_not_cut():
    with pytest.raises(ValueError, match="at most 300 characters"):
        _save("x" * 301)


def test_a_description_at_the_limit_is_passed_through_whole():
    text = "y" * 300
    _, nb = _save(text)
    assert nb.calls == [text]


def test_no_description_is_passed_as_empty():
    _, nb = _save(None)
    assert nb.calls == [""]
