"""Round 3, B6: a mission's reproducibility bundle (A7-11).

The bundle is a read-only zip: the code cells that ran, their inputs and outputs (with sha256), the ledger, the events, the
configuration (prompt versions, package versions, tool arguments) and the final answer. It never holds .env or a key.
"""
from __future__ import annotations

import hashlib
import io
import json
import zipfile

from fastapi.testclient import TestClient

import prophet_agent.api.app as app_mod
from prophet_agent.api.app import create_app
from prophet_agent.api.service import MissionService
from prophet_agent.tools import default_tools

RUN_A = "a" * 32
KEY = "nvapi-PLANTEDTESTKEY0123456789"


def _client(tmp_path):
    service = MissionService(tmp_path / "api.sqlite", lambda: None, default_tools())
    return TestClient(create_app(service), base_url="http://127.0.0.1:8000"), service


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _make_run(root, run_id: str, *, leak: bool = False) -> dict[str, bytes]:
    wd = root / run_id
    files = {
        "program.py": b"import prophet_lib as pl\nprint(pl.load('in/series.csv'))\n" + (f"# {KEY}\n".encode() if leak else b""),
        "manifest.json": b'{"returncode": 0, "outputs": []}',
        "lib/prophet_lib.py": b"def load():\n    return 1\n",
        "in/series.csv": b"date,units\n2024-01-01,10\n2024-01-08,12\n",
        "out/forecast.json": b'{"point": [11.0, 12.0]}',
        "out/chart.png": b"\x89PNG-not-really",
    }
    for name, data in files.items():
        (wd / name).parent.mkdir(parents=True, exist_ok=True)
        (wd / name).write_bytes(data)
    return files


def _mission_with_run(service, root, *, leak: bool = False) -> tuple[str, dict[str, bytes]]:
    mid = service.store.create_mission("forecast units", {"max_tool_calls": 20, "model": "nim-default"})
    service.store.tool_start(mid, 0, "sandbox_python", {"code": "print(1)"}, "k-sandbox")
    service.store.tool_set("k-sandbox", "done", {"run_id": RUN_A, "outputs": []})
    service.store.tool_start(mid, 1, "timesfm_forecast", {"values": [1, 2, 3], "horizon": 2}, "k-tfm")
    service.store.tool_set("k-tfm", "done", {"checkpoint": "stub", "point": [1.0]})
    service.store.set_answer(mid, {"status": "met", "answer": "Units rise to about 12.", "stop_reason": "done_verified"})
    service.store.set_status(mid, "completed")
    return mid, _make_run(root, RUN_A, leak=leak)


def _zip(client, mid) -> zipfile.ZipFile:
    r = client.get(f"/api/missions/{mid}/repro")
    assert r.status_code == 200, r.text
    assert r.headers["content-type"].startswith("application/zip")
    assert "attachment" in r.headers["content-disposition"]
    return zipfile.ZipFile(io.BytesIO(r.content))


def test_the_bundle_holds_code_inputs_outputs_ledger_configuration_and_answer(tmp_path, monkeypatch):
    root = tmp_path / "sandbox"
    monkeypatch.setattr(app_mod, "SANDBOX_ROOT", root)
    client, service = _client(tmp_path)
    mid, files = _mission_with_run(service, root)
    names = set(_zip(client, mid).namelist())
    for expected in ("README.txt", "mission.json", "answer.md", "ledger.json", "events.json", "configuration.json",
                     "hashes.json", f"runs/{RUN_A}/program.py", f"runs/{RUN_A}/manifest.json",
                     f"runs/{RUN_A}/lib/prophet_lib.py", f"runs/{RUN_A}/in/series.csv", f"runs/{RUN_A}/out/forecast.json"):
        assert expected in names, expected
    assert f"runs/{RUN_A}/out/chart.png" not in names          # charts are described by their hash, not bundled


def test_the_bundled_inputs_match_their_recorded_sha256(tmp_path, monkeypatch):
    root = tmp_path / "sandbox"
    monkeypatch.setattr(app_mod, "SANDBOX_ROOT", root)
    client, service = _client(tmp_path)
    mid, files = _mission_with_run(service, root)
    zf = _zip(client, mid)
    hashes = json.loads(zf.read("hashes.json"))
    assert hashes[f"runs/{RUN_A}/in/series.csv"] == _sha(files["in/series.csv"])
    assert zf.read(f"runs/{RUN_A}/in/series.csv") == files["in/series.csv"]


def test_the_ledger_and_configuration_record_the_tool_arguments_and_prompt_versions(tmp_path, monkeypatch):
    root = tmp_path / "sandbox"
    monkeypatch.setattr(app_mod, "SANDBOX_ROOT", root)
    client, service = _client(tmp_path)
    mid, _ = _mission_with_run(service, root)
    zf = _zip(client, mid)
    ledger = json.loads(zf.read("ledger.json"))
    tfm = next(row for row in ledger if row["tool"] == "timesfm_forecast")
    assert tfm["args"] == {"values": [1, 2, 3], "horizon": 2}
    cfg = json.loads(zf.read("configuration.json"))
    from prophet_agent.prompts import CURRENT
    assert cfg["prompts"]["planner"]["version"] == CURRENT["planner"]
    assert len(cfg["prompts"]["planner"]["sha256"]) == 12
    assert cfg["budget"]["model"] == "nim-default"
    assert "packages" in cfg


def test_the_bundle_never_holds_env_files_or_key_shaped_text(tmp_path, monkeypatch):
    root = tmp_path / "sandbox"
    monkeypatch.setattr(app_mod, "SANDBOX_ROOT", root)
    client, service = _client(tmp_path)
    mid, _ = _mission_with_run(service, root, leak=True)
    (tmp_path / ".env").write_text(f"NVIDIA_NIM_API_KEY_1={KEY}\n", encoding="utf-8")
    zf = _zip(client, mid)
    names = zf.namelist()
    assert not any(n.endswith(".env") or n == ".env" for n in names)
    for name in names:
        data = zf.read(name)
        assert b"nvapi-" not in data, name
        assert KEY.encode() not in data, name
    assert "nvapi-" not in json.dumps(json.loads(zf.read("mission.json")))


def test_the_bundle_says_what_a_rerun_should_match(tmp_path, monkeypatch):
    root = tmp_path / "sandbox"
    monkeypatch.setattr(app_mod, "SANDBOX_ROOT", root)
    client, service = _client(tmp_path)
    mid, _ = _mission_with_run(service, root)
    readme = _zip(client, mid).read("README.txt").decode("utf-8")
    assert "Rerunning the same code on the same input should give the same figures" in readme


def test_a_missing_run_folder_is_reported_not_an_error(tmp_path, monkeypatch):
    root = tmp_path / "sandbox"
    monkeypatch.setattr(app_mod, "SANDBOX_ROOT", root)
    client, service = _client(tmp_path)
    mid, _ = _mission_with_run(service, root)
    import shutil
    shutil.rmtree(root / RUN_A)
    zf = _zip(client, mid)
    assert not any(n.startswith(f"runs/{RUN_A}/") for n in zf.namelist())
    assert RUN_A in json.loads(zf.read("configuration.json"))["missing_runs"]


def test_an_unknown_mission_is_404_and_the_route_is_read_only(tmp_path, monkeypatch):
    monkeypatch.setattr(app_mod, "SANDBOX_ROOT", tmp_path / "sandbox")
    client, _ = _client(tmp_path)
    assert client.get("/api/missions/nope/repro").status_code == 404
    assert client.post("/api/missions/nope/repro").status_code == 405
