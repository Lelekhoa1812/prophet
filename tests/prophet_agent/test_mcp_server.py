"""The MCP bridge: protocol shape, tool listing, and the ask loop through a stubbed HTTP layer."""
from prophet_agent import mcp_server as m


def rpc(method, params=None, mid=1):
    return m.handle({"jsonrpc": "2.0", "id": mid, "method": method, "params": params or {}})


def test_initialize_and_list():
    assert rpc("initialize")["result"]["serverInfo"]["name"] == "prophet"
    tools = rpc("tools/list")["result"]["tools"]
    names = {t["name"] for t in tools}
    assert {"prophet_ask", "prophet_add_file", "prophet_answer", "prophet_cancel", "prophet_files"} <= names
    assert all(t["inputSchema"]["type"] == "object" for t in tools)
    assert m.handle({"jsonrpc": "2.0", "method": "notifications/initialized"}) is None


def test_unknown_tool_and_method():
    assert rpc("tools/call", {"name": "nope"})["error"]["code"] == -32602
    assert rpc("nope")["error"]["code"] == -32601


def test_ask_returns_the_answer(monkeypatch):
    calls = []

    def fake(method, path, body=None, **kw):
        calls.append((method, path))
        if path.endswith("/turns"):
            return {"id": "t1", "seq": 1}
        return {"id": "t1", "status": "met", "answer": {"status": "met", "answer": "North 25,436.75 over 20 rows"}}

    monkeypatch.setattr(m, "_call", fake)
    out = rpc("tools/call", {"name": "prophet_ask", "arguments": {"notebook_id": "n", "question": "totals?"}})["result"]
    assert out["isError"] is False
    assert "25,436.75" in out["content"][0]["text"]


def test_waiting_user_surfaces_questions(monkeypatch):
    q = {"id": "q1", "payload": {"questions": [{"question": "Which date?", "header": "Date", "multiSelect": False,
                                                 "options": [{"label": "a", "description": ""}, {"label": "b", "description": ""}]}]}}
    monkeypatch.setattr(m, "_call", lambda method, path, body=None, **kw: {"id": "t1", "seq": 1} if path.endswith("/turns") else
                        {"id": "t1", "status": "waiting_user", "answer": None, "questions": [q]})
    text = rpc("tools/call", {"name": "prophet_ask", "arguments": {"notebook_id": "n", "question": "x y z"}})["result"]["content"][0]["text"]
    assert "q1" in text and "prophet_answer" in text


def test_errors_are_reported_not_raised(monkeypatch):
    def boom(*a, **k):
        raise m.ProphetError("Prophet is not running")
    monkeypatch.setattr(m, "_call", boom)
    r = rpc("tools/call", {"name": "prophet_list_notebooks", "arguments": {}})["result"]
    assert r["isError"] is True and "not running" in r["content"][0]["text"]
    assert rpc("tools/call", {"name": "prophet_ask", "arguments": {}})["result"]["isError"] is True
