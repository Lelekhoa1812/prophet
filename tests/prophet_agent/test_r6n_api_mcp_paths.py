"""Round 6n, defect 1: the MCP bridge refuses to upload secret-looking local files.

prophet_add_file and prophet_quick_forecast read a local path and send its bytes to the API, from where the planner can
reach them. The deny-list is checked on the RESOLVED path, so a symlink into a protected folder is refused as well.

Every test points HOME at a fake home under tmp_path and stubs the API call, so nothing touches the real home or the network.
"""
from __future__ import annotations

import json

import pytest

from prophet_agent import mcp_server

PRIVATE_TEXT = "PRIVATE-KEY-MATERIAL-DO-NOT-ECHO"


@pytest.fixture()
def fake_home(tmp_path, monkeypatch):
    home = tmp_path / "home"
    (home / ".ssh").mkdir(parents=True)
    (home / ".ssh" / "id_rsa").write_text(PRIVATE_TEXT)
    (home / ".aws").mkdir()
    (home / ".aws" / "credentials").write_text(PRIVATE_TEXT)
    (home / ".config" / "gcloud").mkdir(parents=True)
    (home / ".config" / "gcloud" / "token.json").write_text(PRIVATE_TEXT)
    (home / "Library" / "Keychains").mkdir(parents=True)
    (home / "Library" / "Keychains" / "login.keychain-db").write_text(PRIVATE_TEXT)
    monkeypatch.setenv("HOME", str(home))
    return home


@pytest.fixture()
def api(monkeypatch):
    """Records every API call. Any call means the file was read and sent, which a refused path must never cause."""
    calls: list[tuple[str, str, bytes | None]] = []

    def fake_call(method, path, body=None, raw=None, headers=None, timeout=60.0):
        calls.append((method, path, raw))
        if path.endswith("/artifacts"):
            return []
        return {"id": "src1", "status": "ready", "mission_id": "m1", "run_id": "r1"}

    monkeypatch.setattr(mcp_server, "_call", fake_call)
    return calls


def _call_tool(name: str, args: dict) -> dict:
    reply = mcp_server.handle({"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": name, "arguments": args}})
    return reply["result"]


def _add_file(path, notebook="nb1"):
    return _call_tool("prophet_add_file", {"notebook_id": notebook, "path": str(path), "wait_seconds": 0})


def _quick(path):
    return _call_tool("prophet_quick_forecast", {"goal": "forecast demand", "path": str(path), "wait_seconds": 0})


@pytest.mark.parametrize("relative", [
    ".aws/credentials",                       # a dot-prefixed folder under home
    ".ssh/id_rsa",                            # a dot-prefixed folder under home, and an id_rsa file
    ".config/gcloud/token.json",              # a nested dot-prefixed config folder
    "Library/Keychains/login.keychain-db",    # a keychain folder
])
def test_a_protected_file_under_home_is_refused_before_anything_is_read_or_sent(fake_home, api, relative):
    result = _add_file(fake_home / relative)
    assert result["isError"] is True
    assert api == [], "a refused file must never reach the API"
    assert PRIVATE_TEXT not in result["content"][0]["text"], "the refusal must not echo file contents"


@pytest.mark.parametrize("name", [".env", ".env.production", "id_rsa_backup", "server.pem", "tls.key", "cert.p12", "cert.pfx"])
def test_a_secret_named_file_is_refused_anywhere(tmp_path, fake_home, api, name):
    path = tmp_path / "ops" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("a,b\n1,2\n")
    result = _add_file(path)
    assert result["isError"] is True and api == []


def test_the_refusal_names_the_rule_and_not_the_full_path(fake_home, api):
    result = _add_file(fake_home / ".aws" / "credentials")
    text = result["content"][0]["text"]
    assert "dot-prefixed" in text, "the refusal names the rule that fired"
    assert str(fake_home) not in text, "the refusal does not repeat the full path"


def test_a_directory_inside_a_protected_folder_is_refused(fake_home, api):
    result = _quick(fake_home / ".ssh")
    assert result["isError"] is True and api == []


def test_a_symlink_from_tmp_path_into_ssh_is_refused(tmp_path, fake_home, api):
    link_dir = tmp_path / "linked"
    link_dir.symlink_to(fake_home / ".ssh", target_is_directory=True)
    result = _add_file(link_dir / "id_rsa")
    assert result["isError"] is True and api == []


def test_a_symlink_named_like_a_csv_that_points_at_a_key_is_refused(tmp_path, fake_home, api):
    link = tmp_path / "sales.csv"
    link.symlink_to(fake_home / ".ssh" / "id_rsa")
    result = _quick(link)
    assert result["isError"] is True and api == []


def test_a_normal_csv_under_tmp_path_is_accepted_and_sent(tmp_path, fake_home, api):
    data = b"date,value\n2026-01-01,4\n"
    path = tmp_path / "sales.csv"
    path.write_bytes(data)
    result = _add_file(path)
    assert result["isError"] is False
    posts = [(m, p, raw) for m, p, raw in api if m == "POST"]
    assert posts and posts[0][2] == data
    assert json.loads(result["content"][0]["text"])["id"] == "src1"
