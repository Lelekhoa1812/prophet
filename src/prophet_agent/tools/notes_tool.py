"""Notes: an agent's own working memory, kept outside its context window.

Motivation vs Logic
Motivation: compaction folds old steps into a short memory, and a sub-agent starts with no history at all. A decision, a
run id or a list of open questions the agent wants to keep exactly must live somewhere compaction cannot touch. Manus
calls this the todo or notes file; Claude Code's agent memory is the same idea.
Logic: one note store per mission, one scope per agent (``main`` for the goal, ``a1``, ``a2`` for sub-agents). An agent writes
and appends only to its own scope and may read any scope, so the parent can read what a sub-agent left and a sub-agent can
read the parent's brief notes. A note is plain text, at most 6,000 characters. The tool is idempotent: a write is an upsert,
and an append that repeats the note's last text is a no-op, so a retried call does not duplicate it.
"""
from __future__ import annotations

import re

from prophet_agent.runtime.runner import Tool

MAX_NOTE = 6_000
MAX_NOTES = 40
REPEAT_NOTE = ("this text repeats the note's last text, so it was not added again; the note is unchanged. "
               "Read the note with action read if you need to see it.")
NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 _.\-]{0,59}$")
ALLOWED = {"action", "name", "text", "scope"}


def notes_tool(store, mission_id: str, scope: str) -> Tool:
    def _run(args: dict, _key: str) -> dict:
        extra = sorted(set(args) - ALLOWED)
        if extra:
            raise ValueError(f"unknown arguments {extra}; allowed: action, name, text, scope")
        # Round 6m: the action and the scope are checked before the name, and each is checked for its type. A name of None
        # used to be stored as the note "None", and a list scope reached the store unchecked.
        action = args.get("action") or "list"
        if action not in ("list", "read", "write", "append"):
            raise ValueError("action must be one of: list, read, write, append")
        scope_arg = args.get("scope")
        if scope_arg is not None and (not isinstance(scope_arg, str) or not scope_arg.strip()):
            raise ValueError("scope must be one of the agent scopes, for example main, a1 or a2, or left out")
        if action == "list":
            return {"scope": scope, "notes": store.note_list(mission_id, scope_arg)}
        raw_name = args.get("name")
        if raw_name is None or raw_name == "":
            raise ValueError("name is required: 1 to 60 characters, letters, digits, space, _ . -")
        if not isinstance(raw_name, str):
            raise ValueError("name is required as text: 1 to 60 characters, letters, digits, space, _ . -")
        name = raw_name.strip()
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
            current = store.note_get(mission_id, scope, name)
            if action == "append" and current is not None and (current == text or current.endswith("\n" + text)):
                # The tool is re-run after a crash while "started", with the same arguments. An append that repeats the
                # note's last text is therefore a no-op; otherwise a retry would write the same line twice (round 6l, item 7).
                # The result says so in words: a silent drop looked like a save to the planner (GOAL.md defect 1).
                return {"scope": scope, "name": name, "chars": len(current), "saved": True, "unchanged": True,
                        "repeat": True, "note": REPEAT_NOTE}
            if action == "write" and current == text:
                # Round 6m: a write that repeats the note's text changes nothing; say so, as an identical append already does,
                # so the planner does not repeat it to see whether it took.
                return {"scope": scope, "name": name, "chars": len(current), "saved": True, "unchanged": True,
                        "repeat": True, "note": REPEAT_NOTE}
            if action == "append" and current is not None and len(current) + 1 + len(text) > MAX_NOTE:
                # The limit is on the note, not on one text: an append that would take the note past it is refused, and the
                # note is left as it was (round 6m: ten 1,000-character appends had made a 10,019-character note).
                total = len(current) + 1 + len(text)
                raise ValueError(f"this append would make note {name!r} {total} characters; a note is at most {MAX_NOTE}. "
                                 "Write a shorter text, replace the note with action write, or start a new note name")
            if current is None and len(store.note_list(mission_id, scope)) >= MAX_NOTES:
                raise ValueError(f"at most {MAX_NOTES} notes per agent; overwrite or reuse one")
            chars = store.note_put(mission_id, scope, name, text, append=(action == "append"))
            return {"scope": scope, "name": name, "chars": chars, "saved": True}
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
