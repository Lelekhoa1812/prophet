"""Round 6n, defect 3: a scrubbed input keeps the hash of the bytes the person uploaded.

The bundle removes key-shaped text from text files before it writes them. Hashing the scrubbed copy alone meant a person who
compared hashes.json with their original file found a mismatch. A changed file now records sha256_original (the bytes as
read from the sandbox, the bytes the person supplied) and sha256_scrubbed (the copy in the bundle), and only when they differ.

The key below is synthetic: "nvapi-" followed by twenty made-up letters. No real key is used anywhere in these tests.
"""
from __future__ import annotations

import hashlib
import io
import json
import time
import uuid
import zipfile

import pytest

from prophet_agent.api import bundle as bundle_mod
from prophet_agent.api import titles
from prophet_agent.api.bundle import build_bundle
from prophet_agent.api.service import MissionService
from prophet_agent.tools import default_tools

SYNTHETIC_KEY = "nvapi-" + "abcdefghijklmnopqrst"


@pytest.fixture(autouse=True)
def no_title_model(monkeypatch):
    monkeypatch.setattr(titles, "name_in_background", lambda *a, **k: None)


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _read(data: bytes) -> dict[str, bytes]:
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        return {name: zf.read(name) for name in zf.namelist()}


def _mission_with_run(service, sandbox_root, files: dict[str, bytes]):
    """A mission whose ledger names one run folder holding ``files``."""
    mid = service.create("forecast the regional demand")
    rid = uuid.uuid4().hex
    for rel, data in files.items():
        path = sandbox_root / rid / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    with service.store._tx() as c:
        c.execute("INSERT INTO tool_calls (idem_key, mission_id, step_idx, tool, args_json, status, result_json, error, updated) "
                  "VALUES (?,?,?,?,?,?,?,?,?)",
                  (f"{mid}-1", mid, 1, "forecast_report", "{}", "done", json.dumps({"run_id": rid}), None, time.time()))
    return mid, rid


def test_a_scrubbed_input_records_the_hash_of_the_original_bytes_and_of_the_scrubbed_copy(tmp_path):
    service = MissionService(tmp_path / "api.sqlite", lambda: None, default_tools())
    sandbox = tmp_path / "sandbox"
    original = f"date,api_key\n2026-01-01,{SYNTHETIC_KEY}\n".encode()
    mid, rid = _mission_with_run(service, sandbox, {"program.py": b"pass\n", "in/data.csv": original})

    entries = _read(build_bundle(service.store, mid, sandbox))
    stored = entries[f"runs/{rid}/in/data.csv"]
    assert SYNTHETIC_KEY.encode() not in stored, "the copy in the bundle is still scrubbed"

    record = json.loads(entries["hashes.json"])[f"runs/{rid}/in/data.csv"]
    assert record["sha256_original"] == _sha(original), "the original hash must match the bytes the person supplied"
    assert record["sha256_scrubbed"] == _sha(stored), "the scrubbed hash must match the copy in the bundle"
    assert record["sha256_scrubbed"] != record["sha256_original"]


def test_a_file_that_needed_no_scrubbing_has_one_hash_that_matches_its_bytes(tmp_path):
    service = MissionService(tmp_path / "api.sqlite", lambda: None, default_tools())
    sandbox = tmp_path / "sandbox"
    data = b"date,value\n2026-01-01,4\n"
    mid, rid = _mission_with_run(service, sandbox, {"program.py": b"pass\n", "in/data.csv": data})
    hashes = json.loads(_read(build_bundle(service.store, mid, sandbox))["hashes.json"])
    assert hashes[f"runs/{rid}/in/data.csv"] == _sha(data)
    assert "sha256_scrubbed" not in json.dumps(hashes), "a file that did not change records no second hash"


def test_the_rerun_note_tells_the_reader_which_hash_to_compare(tmp_path):
    note = bundle_mod.RERUN_NOTE
    assert "sha256_original" in note and "sha256_scrubbed" in note
    service = MissionService(tmp_path / "api.sqlite", lambda: None, default_tools())
    sandbox = tmp_path / "sandbox"
    mid, _ = _mission_with_run(service, sandbox, {"program.py": b"pass\n"})
    readme = _read(build_bundle(service.store, mid, sandbox))["README.txt"].decode()
    assert "sha256_original" in readme
