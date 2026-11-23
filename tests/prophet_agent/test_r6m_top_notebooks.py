"""Round 6m: notebook helpers that the unit tests did not reach. Offline: no sandbox run and no network."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from prophet_agent.datasets import DatasetStore
from prophet_agent.notebooks import CARD_CHARS, Notebooks, make_card
from prophet_agent.runtime.store import Store

SRC = Path(__file__).resolve().parents[2] / "src"


def card_in_a_child_process(profile: dict) -> dict:
    """make_card in a child process with a timeout, so a loop that never ends fails the test instead of hanging it."""
    code = "import json, sys\nfrom prophet_agent.notebooks import make_card\nprint(json.dumps(make_card(json.loads(sys.stdin.read()))))"
    proc = subprocess.run([sys.executable, "-c", code], input=json.dumps(profile), capture_output=True, text=True,
                          timeout=30, env={**os.environ, "PYTHONPATH": str(SRC)})
    assert proc.returncode == 0, proc.stderr[-500:]
    return json.loads(proc.stdout)


def one_column_sheet(issue: str) -> dict:
    return {"name": "Sheet1", "rows": 3, "columns": [{"name": "units", "kind": "number"}], "issues": [issue],
            "layout_hints": []}


def test_a_sheet_with_one_column_and_a_long_issue_gives_a_bounded_card_and_returns():
    card = card_in_a_child_process({"sheets": [one_column_sheet("unreadable cell " * 400)]})
    sheet = card["sheets"][0]
    assert sheet["columns"] == ["units: number"]
    assert len(json.dumps(card)) <= CARD_CHARS


def test_a_card_with_many_one_column_sheets_over_budget_still_returns():
    sheets = [one_column_sheet(f"sheet {i}: " + "x" * 600) for i in range(12)]
    card = card_in_a_child_process({"sheets": sheets})
    assert len(card["sheets"]) == 12
    assert all(len(s["columns"]) == 1 for s in card["sheets"])


def test_the_card_still_halves_wide_sheets_until_it_fits():
    columns = [{"name": f"c{i}", "kind": "text"} for i in range(40)]
    sheet = {"name": "Wide", "rows": 9, "columns": columns, "issues": [], "layout_hints": []}
    card = make_card({"sheets": [sheet]})
    shown = card["sheets"][0]
    assert len(shown["columns"]) + shown["columns_omitted"] == 40
    assert len(json.dumps(card)) <= CARD_CHARS


def test_a_notebook_created_with_a_blank_title_is_named_like_an_untitled_one(tmp_path):
    store = Store(tmp_path / "p.sqlite")
    notebooks = Notebooks(store, DatasetStore(tmp_path / "datasets"), tmp_path / "notebooks")
    try:
        assert notebooks.create("   ")["title"] == "New notebook"
        assert notebooks.create(None)["title"] == "New notebook"
    finally:
        store.close()


def test_table_path_version_zero_is_not_the_latest_version(tmp_path):
    """Versions start at 1. Version 0 names no version, so it must not quietly return the latest file."""
    store = Store(tmp_path / "p.sqlite")
    notebooks = Notebooks(store, DatasetStore(tmp_path / "datasets"), tmp_path / "notebooks")
    try:
        nid = notebooks.create("Sales")["id"]
        tid = "t" * 32
        with store._tx() as c:
            c.execute("INSERT INTO derived_tables (id, notebook_id, name, description, created) VALUES (?,?,?,?,?)",
                      (tid, nid, "forecast", "", 1.0))
            for version in (1, 2):
                c.execute("INSERT INTO derived_versions (id, table_id, version, rows, columns_json, sha256, file_path, "
                          "lineage_json, turn_id, created) VALUES (?,?,?,?,?,?,?,?,?,?)",
                          (f"v{version}".ljust(32, "0"), tid, version, 1, "[]", "0" * 64,
                           str(tmp_path / f"v{version}.csv"), "{}", None, 1.0))
        assert notebooks.table_path(nid, "forecast").name == "v2.csv"
        assert notebooks.table_path(nid, "forecast", 1).name == "v1.csv"
        with pytest.raises(KeyError):
            notebooks.table_path(nid, "forecast", 0)
    finally:
        store.close()
