"""Notebook tool: keep a result an analysis run wrote as a named table that later turns can use.

Motivation vs Logic
Motivation: cleaning and reshaping a spreadsheet is work the person will build on. Redoing it in every question wastes
time and invites drift, so a result worth keeping is saved once under a name.
Logic: the tool asks the notebook to promote one CSV from one run's out/ folder. Saves are immutable versions with
lineage (run, source names, parent version). Saving identical content again changes nothing. The tool only exists for
a turn that belongs to a notebook, so a one-off mission never offers it.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from prophet_agent.runtime.runner import Tool

SEEN_COLUMNS = 30
MAX_DESCRIPTION = 300      # the notebook store keeps this many characters of a table's description


@dataclass(frozen=True)
class NotebookBinding:
    """What a notebook turn needs from the notebook layer. ``notebooks`` is None only for listing the catalogue."""
    notebooks: Any
    notebook_id: str
    turn_id: str


SAVE_TABLE_SPEC = {
    "name": "save_table",
    "title": "Keep a table",
    "activity": "Keeping a table for later questions",
    "description": "Keep a CSV that an analysis run wrote to out/ as a named table in this notebook. Later turns see the "
                   "latest version as in/derived/<name>.csv. Each save makes a new immutable version; saving the same "
                   "content again does nothing. Keep cleaned or reshaped data you will reuse, not every intermediate.",
    "args": {"run_id": "run_id returned by sandbox_python",
             "file": "the file the run wrote, for example out/clean.csv",
             "name": "table name: lowercase letters, digits and underscores, starting with a letter, at most 40",
             "description": "one line saying what the table is (optional)"},
}

_ARGS = {"run_id", "file", "name", "description"}


def make_save(binding: NotebookBinding | None):
    def _save(args: dict, _key: str) -> dict:
        if binding is None or binding.notebooks is None:
            raise ValueError("tables can only be kept inside a notebook")
        extra = sorted(set(args) - _ARGS)
        if extra:
            raise ValueError(f"unknown arguments {extra}; allowed: {sorted(_ARGS)}")
        description = str(args.get("description") or "")
        if len(description) > MAX_DESCRIPTION:
            # Round 6m: promote cut the text to 300 characters without a word, so the saved description ended mid-sentence.
            raise ValueError(f"description must be at most {MAX_DESCRIPTION} characters, given {len(description)}; "
                             "shorten it to one line saying what the table is")
        return binding.notebooks.promote(binding.notebook_id, str(args.get("run_id", "")), str(args.get("file", "")),
                                         str(args.get("name", "")), description, turn_id=binding.turn_id)
    return _save


def observe(result: dict) -> dict:
    cols = result.get("columns", [])
    out = {"name": result.get("name"), "version": result.get("version"), "rows": result.get("rows"),
           "columns": cols[:SEEN_COLUMNS]}
    if len(cols) > SEEN_COLUMNS:
        out["columns_omitted"] = len(cols) - SEEN_COLUMNS
    if result.get("unchanged"):
        out["unchanged"] = True
    return out


def save_table_tool(binding: NotebookBinding | None = None) -> Tool:
    return Tool("save_table", make_save(binding), idempotent=True, observe=observe)
