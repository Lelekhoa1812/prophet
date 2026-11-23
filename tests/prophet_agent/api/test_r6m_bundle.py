"""Behaviour of the reproducibility bundle (api/bundle.py) and its route (round 6m).

A bundle is built from the store and from one sandbox folder per run. These tests build a real mission with a run folder
in tmp_path, then read the zip back and check what it holds, what it leaves out and what it scrubs.
"""
from __future__ import annotations

import hashlib
import io
import json
import time
import uuid
import zipfile

import pytest
from fastapi.testclient import TestClient

from prophet_agent.api import app as app_module
from prophet_agent.api import bundle as bundle_mod
from prophet_agent.api import titles
from prophet_agent.api.app import create_app
from prophet_agent.api.bundle import build_bundle
from prophet_agent.api.service import MissionService
from prophet_agent.tools import default_tools

FAKE_KEY = "nvapi-FAKETESTKEY0123456789"       # a fake, key-shaped string: no real key is used anywhere in these tests
ENTRIES = {"README.txt", "answer.md", "configuration.json", "events.json", "ledger.json", "mission.json", "hashes.json"}


@pytest.fixture(autouse=True)
def no_title_model(monkeypatch):
    monkeypatch.setattr(titles, "name_in_background", lambda *a, **k: None)


def _service(tmp_path):
    return MissionService(tmp_path / "api.sqlite", lambda: None, default_tools())


def _read(data: bytes) -> dict[str, bytes]:
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        return {name: zf.read(name) for name in zf.namelist()}


def _run(service, sandbox_root, goal, files: dict[str, bytes], *, budget=None):
    """A mission whose ledger holds one finished tool call that names a run folder, and that folder with ``files``."""
    mid = service.create(goal, budget=budget)
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


def _standard_run():
    return {"program.py": b"print('forecast')\n", "manifest.json": b"{}", "lib/helper.py": b"X = 1\n",
            "in/data.csv": b"date,value\n2026-01-01,4\n", "out/result.json": b'{"point": [1.0]}',
            "out/chart.png": b"\x89PNG-not-really", "notes.txt": b"not part of a rerun"}


def test_the_bundle_holds_the_mission_files_and_a_sha256_for_each_of_them(tmp_path):
    service = _service(tmp_path)
    sandbox = tmp_path / "sandbox"
    mid, rid = _run(service, sandbox, "forecast the regional demand", _standard_run())
    entries = _read(build_bundle(service.store, mid, sandbox))
    assert ENTRIES <= set(entries)
    hashes = json.loads(entries["hashes.json"])
    assert set(hashes) == set(entries) - {"hashes.json"}
    for name, digest in hashes.items():
        assert hashlib.sha256(entries[name]).hexdigest() == digest, name
    mission = json.loads(entries["mission.json"])
    assert mission["id"] == mid and mission["goal"] == "forecast the regional demand"


def test_a_run_folder_brings_its_rerun_files_and_nothing_else(tmp_path):
    service = _service(tmp_path)
    sandbox = tmp_path / "sandbox"
    mid, rid = _run(service, sandbox, "forecast the regional demand", _standard_run())
    names = set(_read(build_bundle(service.store, mid, sandbox)))
    base = f"runs/{rid}/"
    assert {base + "program.py", base + "manifest.json", base + "lib/helper.py", base + "in/data.csv",
            base + "out/result.json"} <= names
    assert base + "out/chart.png" not in names          # outputs other than numbers are not copied
    assert base + "notes.txt" not in names              # files at the run root other than the listed ones are not copied
    config = json.loads(_read(build_bundle(service.store, mid, sandbox))["configuration.json"])
    assert config["runs"] == [rid] and config["missing_runs"] == []


def test_key_shaped_text_is_removed_from_every_file_the_bundle_holds(tmp_path):
    service = _service(tmp_path)
    sandbox = tmp_path / "sandbox"
    files = _standard_run()
    files["program.py"] = f"KEY = '{FAKE_KEY}'\n".encode()
    mid, _ = _run(service, sandbox, f"forecast with {FAKE_KEY} as the key", files)
    entries = _read(build_bundle(service.store, mid, sandbox))
    for name, data in entries.items():
        assert FAKE_KEY.encode() not in data, name
    assert b"[key removed]" in entries["mission.json"]


def test_an_input_over_the_copy_limit_is_listed_by_hash_and_not_copied(tmp_path, monkeypatch):
    monkeypatch.setattr(bundle_mod, "MAX_INPUT_BYTES", 10)
    service = _service(tmp_path)
    sandbox = tmp_path / "sandbox"
    files = {"in/big.csv": b"a,b\n" * 40, "program.py": b"pass\n"}
    mid, rid = _run(service, sandbox, "forecast the regional demand", files)
    entries = _read(build_bundle(service.store, mid, sandbox))
    assert f"runs/{rid}/in/big.csv" not in entries
    config = json.loads(entries["configuration.json"])
    assert config["inputs_not_copied"] == ["in/big.csv"]
    hashes = json.loads(entries["hashes.json"])
    assert hashes[f"runs/{rid}/in/big.csv"] == hashlib.sha256(files["in/big.csv"]).hexdigest()


def test_a_run_folder_that_is_gone_is_listed_as_missing(tmp_path):
    service = _service(tmp_path)
    sandbox = tmp_path / "sandbox"
    sandbox.mkdir()
    mid, rid = _run(service, sandbox, "forecast the regional demand", {"program.py": b"pass\n"})
    for path in (sandbox / rid).rglob("*"):
        if path.is_file():
            path.unlink()
    (sandbox / rid).rmdir()
    config = json.loads(_read(build_bundle(service.store, mid, sandbox))["configuration.json"])
    assert config["missing_runs"] == [rid] and config["runs"] == [rid]


def test_budget_entries_that_look_like_secrets_are_left_out(tmp_path):
    service = _service(tmp_path)
    sandbox = tmp_path / "sandbox"
    mid, _ = _run(service, sandbox, "forecast the regional demand", {"program.py": b"pass\n"},
                  budget={"max_tool_calls": 5, "api_key": "x", "password_hint": "y", "secret_word": "z"})
    budget = json.loads(_read(build_bundle(service.store, mid, sandbox))["mission.json"])["budget"]
    assert budget == {"max_tool_calls": 5}


def test_an_unknown_mission_is_a_key_error(tmp_path):
    service = _service(tmp_path)
    with pytest.raises(KeyError):
        build_bundle(service.store, "0" * 32, tmp_path / "sandbox")


def test_a_symlink_in_a_run_folder_cannot_pull_in_a_file_outside_it(tmp_path):
    """A run folder is written by analysis code, so an entry in out/ or in/ may be a symlink. The bundle must read only
    files inside the run folder; a symlink to a file outside it is left out."""
    service = _service(tmp_path)
    sandbox = tmp_path / "sandbox"
    outside = tmp_path / "outside.json"
    outside.write_text('{"private": "OUTSIDE-ONLY-TEXT"}')
    files = {"program.py": b"pass\n"}
    mid, rid = _run(service, sandbox, "forecast the regional demand", files)
    link_dir = sandbox / rid / "out"
    link_dir.mkdir(parents=True, exist_ok=True)
    (link_dir / "leak.json").symlink_to(outside)
    entries = _read(build_bundle(service.store, mid, sandbox))
    assert f"runs/{rid}/out/leak.json" not in entries
    assert not any(b"OUTSIDE-ONLY-TEXT" in data for data in entries.values())


def _client(tmp_path, monkeypatch):
    service = _service(tmp_path)
    monkeypatch.setattr(app_module, "SANDBOX_ROOT", tmp_path / "sandbox")
    return TestClient(create_app(service), base_url="http://127.0.0.1:8000"), service


def test_the_repro_route_returns_the_zip_as_a_download(tmp_path, monkeypatch):
    client, service = _client(tmp_path, monkeypatch)
    mid, _ = _run(service, tmp_path / "sandbox", "forecast the regional demand", _standard_run())
    r = client.get(f"/api/missions/{mid}/repro")
    assert r.status_code == 200
    assert r.headers["content-type"] == "application/zip"
    assert r.headers["content-disposition"] == f'attachment; filename="prophet-{mid[:12]}-repro.zip"'
    assert ENTRIES <= set(_read(r.content))


def test_the_repro_route_is_404_for_an_unknown_mission(tmp_path, monkeypatch):
    client, _ = _client(tmp_path, monkeypatch)
    assert client.get("/api/missions/" + "f" * 32 + "/repro").status_code == 404
