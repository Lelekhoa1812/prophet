"""Round 6l, audit item 7: a retried notes call does not write its text twice.

Probe (before the fix): a write repeated with the same text stayed one copy (an upsert), but an append repeated with the same
text wrote it twice. The notes tool is declared idempotent, so the runtime re-runs a call left "started" by a crash with the
same arguments; the repeated append was the duplicate. The fix makes an append that repeats the note's last text a no-op.
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


def test_a_repeated_write_leaves_one_copy():
    store, tool = _tool()
    tool.fn({"action": "write", "name": "decision", "text": "use A"}, "k1")
    tool.fn({"action": "write", "name": "decision", "text": "use A"}, "k2")
    assert store.notes[("main", "decision")] == "use A"


def test_a_retried_append_is_not_written_twice():
    store, tool = _tool()
    args = {"action": "append", "name": "log", "text": "step 1"}
    tool.fn(args, "k1")
    result = tool.fn(args, "k1-retry")
    assert store.notes[("main", "log")] == "step 1"
    assert result["saved"] is True and result["unchanged"] is True


def test_different_appends_are_all_kept_in_order():
    store, tool = _tool()
    for text in ("step 1", "step 2", "step 1"):
        tool.fn({"action": "append", "name": "log", "text": text}, "k")
    assert store.notes[("main", "log")] == "step 1\nstep 2\nstep 1"


def test_the_tool_is_still_declared_idempotent():
    _, tool = _tool()
    assert tool.idempotent is True
