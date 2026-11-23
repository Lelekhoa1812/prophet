"""Route audit for the HTTP API (round 6m): write guards, path parameters, error leaks, input validation.

Each test is a behaviour the API should have. The routes checked here already hold it; the defects found in this round
have their own test files (test_r6m_body_limit, test_r6m_missions, test_r6m_tables, test_r6m_settings, test_r6m_bundle).
Nothing here uses the network or a real key: fake key-shaped strings only, and the environment is restored after each test.
"""
from __future__ import annotations

import os
import uuid

import pytest
from fastapi.testclient import TestClient

from prophet_agent.api import app as app_module
from prophet_agent.api import titles
from prophet_agent.api.app import create_app
from prophet_agent.api.service import MissionService
from prophet_agent.tools import default_tools

FAKE_KEY = "nvapi-FAKETESTKEY0123456789"
MISSING = "0" * 32


@pytest.fixture(autouse=True)
def no_title_model(monkeypatch):
    monkeypatch.setattr(titles, "name_in_background", lambda *a, **k: None)


@pytest.fixture()
def restore_env():
    """Settings.set_keys writes the process environment; put it back exactly as it was."""
    before = dict(os.environ)
    yield
    os.environ.clear()
    os.environ.update(before)


def _client(tmp_path, *, raise_server_exceptions=True):
    service = MissionService(tmp_path / "api.sqlite", lambda: None, default_tools())
    client = TestClient(create_app(service), base_url="http://127.0.0.1:8000",
                        raise_server_exceptions=raise_server_exceptions)
    return client, service


# -- write guards --------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("origin", ["http://evil.example", "https://127.0.0.1.evil.example", "null"])
def test_a_write_from_another_origin_is_refused_and_changes_nothing(tmp_path, origin):
    client, service = _client(tmp_path)
    before = len(service.list(limit=500))
    r = client.post("/api/missions", json={"goal": "forecast the regional demand"}, headers={"origin": origin})
    assert r.status_code == 403
    assert len(service.list(limit=500)) == before


def test_a_write_from_a_loopback_origin_on_any_port_passes_the_origin_guard(tmp_path):
    client, _ = _client(tmp_path)
    r = client.post("/api/notebooks", json={"title": "from the dev server"}, headers={"origin": "http://localhost:5173"})
    assert r.status_code == 201


def test_a_request_for_an_unknown_host_is_refused(tmp_path):
    client, _ = _client(tmp_path)
    assert client.get("/api/health", headers={"host": "evil.example"}).status_code == 400


# -- path parameters -----------------------------------------------------------------------------------------------

def test_a_dataset_id_that_is_not_a_hex_id_is_not_found(tmp_path):
    client, _ = _client(tmp_path)
    assert client.get("/api/datasets/..%2F..%2Fetc%2Fpasswd").status_code == 404


def test_a_skill_name_outside_the_playbook_list_is_not_found(tmp_path):
    client, _ = _client(tmp_path)
    assert client.get("/api/skills/..%2F..%2Fsettings").status_code == 404


def test_an_artifact_name_cannot_climb_out_of_the_run_folder(tmp_path, monkeypatch):
    client, _ = _client(tmp_path)
    root = tmp_path / "sandbox"
    rid = uuid.uuid4().hex
    (root / rid / "out").mkdir(parents=True)
    (root / rid / "out" / "chart.png").write_bytes(b"png")
    (root / "secret.txt").write_text("outside the sandbox")
    monkeypatch.setattr(app_module, "SANDBOX_ROOT", root)
    assert client.get(f"/api/artifacts/{rid}/chart.png").status_code == 200       # the control: a file in out/
    assert client.get(f"/api/artifacts/{rid}/..%2F..%2Fsecret.txt").status_code == 404
    assert client.get(f"/api/artifacts/{rid}/%2e%2e%2f%2e%2e%2fsecret.txt").status_code == 404
    assert client.get("/api/artifacts/not-a-run-id/chart.png").status_code == 404


def test_an_artifact_that_is_a_symlink_to_a_file_outside_out_is_not_served(tmp_path, monkeypatch):
    client, _ = _client(tmp_path)
    root = tmp_path / "sandbox"
    rid = uuid.uuid4().hex
    (root / rid / "out").mkdir(parents=True)
    outside = tmp_path / "outside.csv"
    outside.write_text("a,b\n1,2\n")
    (root / rid / "out" / "leak.csv").symlink_to(outside)
    monkeypatch.setattr(app_module, "SANDBOX_ROOT", root)
    assert client.get(f"/api/artifacts/{rid}/leak.csv").status_code == 404


# -- input validation ----------------------------------------------------------------------------------------------

def test_a_test_keys_request_for_an_unknown_provider_is_refused(tmp_path):
    client, _ = _client(tmp_path)
    assert client.post("/api/settings/test-keys?provider=no_such_provider").status_code == 422


def test_an_answer_with_an_overlong_question_header_is_refused(tmp_path):
    client, _ = _client(tmp_path)
    assert client.post(f"/api/questions/{MISSING}/answer", json={"answers": {"x" * 13: "yes"}}).status_code == 422


def test_a_blank_notebook_title_gets_the_default_name(tmp_path):
    client, _ = _client(tmp_path)
    assert client.post("/api/notebooks", json={"title": "   "}).json()["title"] == "New notebook"


def test_a_notebook_title_longer_than_the_limit_is_refused(tmp_path):
    client, _ = _client(tmp_path)
    assert client.post("/api/notebooks", json={"title": "t" * 121}).status_code == 422


# -- keys and secrets ----------------------------------------------------------------------------------------------

def test_a_refused_key_is_not_echoed_back_in_the_error(tmp_path, restore_env):
    client, _ = _client(tmp_path)
    r = client.put("/api/settings", json={"nim": {"keys": ["nvapi-SECRETVALUE-with-a-bang!"]}})
    assert r.status_code == 422
    assert "SECRETVALUE" not in r.text


def test_saved_keys_are_masked_in_settings_and_absent_from_the_export(tmp_path, restore_env):
    client, service = _client(tmp_path)
    service.settings.set_keys([FAKE_KEY])
    settings_text = client.get("/api/settings").text
    export_text = client.get("/api/settings/export").text
    assert FAKE_KEY not in settings_text and "0123456789" not in settings_text
    assert FAKE_KEY not in export_text and "0123456789" not in export_text


# -- error leaks ---------------------------------------------------------------------------------------------------

def test_an_unexpected_error_returns_a_bare_500_without_its_message_or_a_traceback(tmp_path):
    client, service = _client(tmp_path, raise_server_exceptions=False)

    def boom(*_a, **_k):
        raise RuntimeError("internal detail /private/path/.env")

    service.list = boom
    r = client.get("/api/missions")
    assert r.status_code == 500
    assert "/private/path" not in r.text and "Traceback" not in r.text
