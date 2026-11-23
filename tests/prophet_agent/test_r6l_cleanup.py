"""cleanup.py: which run folders and dataset folders a deleted item owns, and removing only those.

The ledger queries run against a small in-memory SQLite database that has just the columns cleanup reads. The file
helpers are pointed at a temporary sandbox and upload root, so nothing under data/ is touched.
"""
from __future__ import annotations

import sqlite3

import pytest

from prophet_agent import cleanup

RUN_A = "a" * 32
RUN_B = "b" * 32
RUN_C = "c" * 32
DATA_X = "1" * 32
DATA_Y = "2" * 32
DATA_Z = "3" * 32


@pytest.fixture
def ledger():
    c = sqlite3.connect(":memory:")
    c.executescript("""
        CREATE TABLE tool_calls (mission_id TEXT, result_json TEXT);
        CREATE TABLE steps (mission_id TEXT, output_json TEXT);
        CREATE TABLE events (mission_id TEXT, payload_json TEXT);
        CREATE TABLE sources (dataset_id TEXT);
        CREATE TABLE missions (id TEXT, budget_json TEXT);
    """)
    yield c
    c.close()


def test_run_ids_are_found_in_compact_and_spaced_json_and_the_union_is_returned():
    compact = '{"run_id":"%s","ok":true}' % RUN_A
    spaced = '{"run_id": "%s"}' % RUN_B
    assert cleanup.run_ids_in(compact, spaced, None, "") == {RUN_A, RUN_B}


def test_values_that_are_not_run_ids_are_ignored():
    wrong_length = '{"run_id":"%s"}' % ("d" * 33)
    upper_hex = '{"run_id":"%s"}' % ("E" * 32)
    other_key = '{"dataset_id":"%s"}' % RUN_C
    assert cleanup.run_ids_in(wrong_length, upper_hex, other_key) == set()


def test_mission_run_ids_read_all_three_ledger_tables_for_that_mission_only(ledger):
    ledger.execute("INSERT INTO tool_calls VALUES ('m1', ?)", ('{"run_id":"%s"}' % RUN_A,))
    ledger.execute("INSERT INTO steps VALUES ('m1', ?)", ('{"run_id": "%s"}' % RUN_B,))
    ledger.execute("INSERT INTO events VALUES ('m1', ?)", ('{"run_id":"%s"}' % RUN_C,))
    ledger.execute("INSERT INTO tool_calls VALUES ('m2', ?)", ('{"run_id":"%s"}' % DATA_X,))

    assert cleanup.mission_run_ids(ledger, ["m1"]) == {RUN_A, RUN_B, RUN_C}
    assert cleanup.mission_run_ids(ledger, []) == set()


def test_a_dataset_named_by_a_source_or_a_mission_budget_is_kept_and_an_orphan_is_returned(ledger):
    ledger.execute("INSERT INTO sources VALUES (?)", (DATA_X,))
    ledger.execute("INSERT INTO missions VALUES ('m1', ?)", ('{"datasets": ["%s"]}' % DATA_Y,))

    orphans = cleanup.unreferenced_datasets(ledger, [DATA_X, DATA_Y, DATA_Z])
    assert orphans == [DATA_Z]


def test_unreferenced_datasets_skips_malformed_ids_and_reports_each_orphan_once(ledger):
    assert cleanup.unreferenced_datasets(ledger, [DATA_Z, "not-an-id", DATA_Z, "../escape"]) == [DATA_Z]


def test_remove_runs_deletes_only_valid_run_folders(monkeypatch, tmp_path):
    sandbox = tmp_path / "sandbox"
    monkeypatch.setattr(cleanup, "SANDBOX_ROOT", sandbox)
    (sandbox / RUN_A).mkdir(parents=True)
    (sandbox / RUN_A / "out.txt").write_text("result")
    (sandbox / RUN_B).mkdir()
    keep_upper = sandbox / ("E" * 32)
    keep_upper.mkdir()
    escape = tmp_path / "escape"
    escape.mkdir()

    cleanup.remove_runs([RUN_A, RUN_C, "E" * 32, "../escape"])   # RUN_C has no folder: that must not raise

    assert not (sandbox / RUN_A).exists()
    assert (sandbox / RUN_B).exists()
    assert keep_upper.exists()
    assert escape.exists()


def test_remove_datasets_deletes_only_valid_dataset_folders_under_the_root(monkeypatch, tmp_path):
    root = tmp_path / "uploads"
    (root / DATA_X).mkdir(parents=True)
    (root / DATA_X / "sales.csv").write_text("a,b\n1,2\n")
    (root / DATA_Y).mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()

    cleanup.remove_datasets(root, [DATA_X, "../outside", "not-an-id"])

    assert not (root / DATA_X).exists()
    assert (root / DATA_Y).exists()
    assert outside.exists()
