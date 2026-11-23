"""Round 6m: the notes tool keeps its own size and argument rules.

Probe (before the fix): ten appends of about 1,000 characters each left one note of 10,019 characters. The tool's own
contract says a note is at most 6,000 characters, but only the length of each single text was checked, not the note's total.
"""
from __future__ import annotations

import pytest

from prophet_agent.tools.notes_tool import MAX_NOTE, notes_tool


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


def test_appends_cannot_grow_a_note_past_the_limit():
    store, tool = _tool()
    refused = None
    for i in range(10):
        try:
            tool.fn({"action": "append", "name": "log", "text": "x" * 1000 + str(i)}, f"k{i}")
        except ValueError as exc:
            refused = str(exc)
            break
    assert refused is not None, "the tenth append should have been refused"
    assert "6000" in refused and "start a new note" in refused
    assert len(store.notes[("main", "log")]) <= MAX_NOTE


def test_a_refused_append_leaves_the_note_as_it_was():
    store, tool = _tool()
    for i in range(5):
        tool.fn({"action": "append", "name": "log", "text": "x" * 1000 + str(i)}, f"k{i}")
    before = store.notes[("main", "log")]
    with pytest.raises(ValueError, match="at most 6000"):
        tool.fn({"action": "append", "name": "log", "text": "y" * 1000}, "k-big")
    assert store.notes[("main", "log")] == before


def test_a_retry_of_the_last_append_is_still_a_no_op_at_the_limit():
    store, tool = _tool()
    for i in range(5):
        tool.fn({"action": "append", "name": "log", "text": "x" * 1000 + str(i)}, f"k{i}")
    last = {"action": "append", "name": "log", "text": "x" * 1000 + "4"}
    result = tool.fn(last, "k4-retry")
    assert result["unchanged"] is True
    assert len(store.notes[("main", "log")]) <= MAX_NOTE


def test_a_write_may_still_replace_a_long_note_up_to_the_limit():
    store, tool = _tool()
    tool.fn({"action": "write", "name": "log", "text": "a" * MAX_NOTE}, "k1")
    assert len(store.notes[("main", "log")]) == MAX_NOTE


def test_a_misspelt_action_is_named_as_the_problem():
    _, tool = _tool()
    with pytest.raises(ValueError, match="action must be one of"):
        tool.fn({"action": "appendd", "name": "log", "text": "x"}, "k")
    with pytest.raises(ValueError, match="action must be one of"):
        tool.fn({"action": "appendd", "text": "x"}, "k")


def test_a_name_that_is_not_text_is_refused_not_stored_as_its_text():
    store, tool = _tool()
    with pytest.raises(ValueError, match="name is required"):
        tool.fn({"action": "write", "name": None, "text": "t"}, "k")
    with pytest.raises(ValueError, match="name is required"):
        tool.fn({"action": "write", "name": 5, "text": "t"}, "k")
    assert store.notes == {}


def test_a_scope_that_is_not_text_is_refused_for_every_action():
    _, tool = _tool()
    for action in ("list", "read", "write"):
        with pytest.raises(ValueError, match="scope must be one of"):
            tool.fn({"action": action, "name": "log", "text": "t", "scope": ["main"]}, "k")
