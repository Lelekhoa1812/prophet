"""Round 6n, defect 2: notebook, turn, question and run IDs reach the API URL encoded.

An ID is interpolated into a path. One that holds "/", "?" or "#" must not change the route it reaches, so every
interpolated ID is quoted with urllib.parse.quote(value, safe=''). The API call is stubbed; nothing goes to the network.
"""
from __future__ import annotations

import json
import urllib.parse

import pytest

from prophet_agent import mcp_server

HOSTILE = "a/b?c#d"
ENCODED = urllib.parse.quote(HOSTILE, safe="")   # a%2Fb%3Fc%23d
assert ENCODED == "a%2Fb%3Fc%23d"


@pytest.fixture()
def api(monkeypatch):
    seen: list[str] = []

    def fake_call(method, path, body=None, raw=None, headers=None, timeout=60.0):
        seen.append(f"{method} {path}")
        if path.endswith("/artifacts"):
            return []
        return {"id": "fresh1", "mission_id": "fresh1", "status": "completed"}

    monkeypatch.setattr(mcp_server, "_call", fake_call)
    return seen


def _call_tool(name: str, args: dict) -> dict:
    reply = mcp_server.handle({"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": name, "arguments": args}})
    return reply["result"]


@pytest.mark.parametrize("name, args", [
    ("prophet_notebook", {"notebook_id": HOSTILE}),
    ("prophet_ask", {"notebook_id": HOSTILE, "question": "total sales?", "wait_seconds": 0}),
    ("prophet_status", {"turn_id": HOSTILE, "wait_seconds": 0}),
    ("prophet_cancel", {"turn_id": HOSTILE}),
    ("prophet_continue", {"turn_id": HOSTILE, "wait_seconds": 0}),
    ("prophet_answer_now", {"turn_id": HOSTILE, "wait_seconds": 0}),
    ("prophet_answer", {"question_id": HOSTILE, "answers": {"Horizon": "4 weeks"}, "wait_seconds": 0}),
    ("prophet_files", {"turn_id": HOSTILE}),
])
def test_a_hostile_id_reaches_the_url_encoded(api, name, args):
    result = _call_tool(name, args)
    assert result["isError"] is False, result["content"][0]["text"]
    assert api, "the tool made an API call"
    first = api[0]
    assert ENCODED in first, first
    assert HOSTILE not in first, "the raw ID must not appear in the path"


def test_a_hostile_run_id_and_file_name_are_encoded_in_the_download_url(api, monkeypatch):
    def fake_call(method, path, body=None, raw=None, headers=None, timeout=60.0):
        return [{"run_id": "r/1?2#3", "files": [{"path": "out/a b#1.csv", "bytes": 3, "sha256": "x"}]}]

    monkeypatch.setattr(mcp_server, "_call", fake_call)
    files = _call_tool("prophet_files", {"turn_id": "t1"})
    url = json.loads(files["content"][0]["text"])[0]["files"][0]["url"]
    assert url == f"{mcp_server.BASE}/api/artifacts/r%2F1%3F2%233/out/a%20b%231.csv"
