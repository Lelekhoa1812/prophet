"""Notes: an agent's own working memory, kept outside its context window.

Motivation vs Logic
Motivation: compaction folds old steps into a short memory, and a sub-agent starts with no history at all. A decision, a
run id or a list of open questions the agent wants to keep exactly must live somewhere compaction cannot touch. Manus
calls this the todo or notes file; Claude Code's agent memory is the same idea.
Logic: one note store per mission, one scope per agent (``main`` for the goal, ``a1``, ``a2`` for sub-agents). An agent writes
and appends only to its own scope and may read any scope, so the parent can read what a sub-agent left and a sub-agent can
read the parent's brief notes. A note is plain text, at most 6,000 characters. The tool is idempotent: a write is an upsert.
"""
from __future__ import annotations

import re

from prophet_agent.runtime.runner import Tool

MAX_NOTE = 6_000
MAX_NOTES = 40
NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 _.\-]{0,59}$")
ALLOWED = {"action", "name", "text", "scope"}


def notes_tool(store, mission_id: str, scope: str) -> Tool:
    def _run(args: dict, _key: str) -> dict:
        extra = sorted(set(args) - ALLOWED)
        if extra:
            raise ValueError(f"unknown arguments {extra}; allowed: action, name, text, scope")
        action = args.get("action", "list")
        if action == "list":
            return {"scope": scope, "notes": store.note_list(mission_id, args.get("scope"))}
        name = str(args.get("name", "")).strip()
        if not NAME.match(name):
            raise ValueError("name is 1 to 60 characters: letters, digits, space, _ . -")
        if action == "read":
            text = store.note_get(mission_id, str(args.get("scope") or scope), name)
            if text is None:
                raise ValueError(f"no note {name!r} in scope {args.get('scope') or scope!r}; action list shows the notes")
            return {"scope": str(args.get("scope") or scope), "name": name, "text": text}
        if action in ("write", "append"):
            if args.get("scope") not in (None, scope):
                raise ValueError(f"you may write only to your own scope {scope!r}")
            text = args.get("text")
            if not isinstance(text, str) or not text.strip():
                raise ValueError("text must be a non-empty string")
            if len(text) > MAX_NOTE:
                raise ValueError(f"a note is at most {MAX_NOTE} characters")
            if store.note_get(mission_id, scope, name) is None and len(store.note_list(mission_id, scope)) >= MAX_NOTES:
                raise ValueError(f"at most {MAX_NOTES} notes per agent; overwrite or reuse one")
            chars = store.note_put(mission_id, scope, name, text, append=(action == "append"))
            return {"scope": scope, "name": name, "chars": min(chars, MAX_NOTE * 4), "saved": True}
        raise ValueError("action is one of: list, read, write, append")
    return Tool("notes", _run, idempotent=True)


NOTES_SPEC = {
    "name": "notes",
    "title": "Notes",
    "activity": "Updating working notes",
    "description": "Your own working memory, kept outside your context window and never folded away. Write a decision, a "
                   "run id and file, a list of open questions or findings you must keep exactly; read them back after a "
                   "long stretch of work. You may read any agent's notes (scope) and write only your own.",
    "args": {"action": "list (default), read, write (replace) or append",
             "name": "note name, 1 to 60 characters (not needed for list)",
             "text": "the note text, at most 6000 characters (write and append)",
             "scope": "read or list another agent's notes: main, a1, a2, ... (default: your own)"},
}
