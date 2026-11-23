"""A repeated notes append is dropped, and the result says so.

Defect (docs/prophet/GOAL.md): an append that repeats the note's last text was a no-op, and the result only carried
``unchanged: True``. Nothing in the result told the planner that its text was not added, so a drop looked like a save.
"""
from __future__ import annotations

from prophet_agent.tools.notes_tool import notes_tool


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


def _tool():
    store = _Store()
    return store, notes_tool(store, "m1", "main")


def test_a_repeated_append_says_in_the_result_that_it_was_a_repeat():
    store, tool = _tool()
    args = {"action": "append", "name": "log", "text": "step 1"}
    tool.fn(args, "k1")
    result = tool.fn(args, "k2")
    assert result["unchanged"] is True
    assert result.get("repeat") is True
    assert "repeat" in result["note"].lower()
    assert "not added" in result["note"].lower()
    assert store.notes[("main", "log")] == "step 1"


def test_a_repeated_write_says_in_the_result_that_it_was_a_repeat():
    store, tool = _tool()
    args = {"action": "write", "name": "plan", "text": "use the 80 percent band"}
    tool.fn(args, "k1")
    result = tool.fn(args, "k2")
    assert result["unchanged"] is True
    assert "repeat" in result["note"].lower()


def test_a_new_append_carries_no_repeat_note():
    _, tool = _tool()
    tool.fn({"action": "append", "name": "log", "text": "step 1"}, "k1")
    fresh = tool.fn({"action": "append", "name": "log", "text": "step 2"}, "k2")
    assert "unchanged" not in fresh
    assert "repeat" not in fresh and "note" not in fresh
