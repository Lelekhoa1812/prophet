"""Round 6m: the MCP bridge answers every request, even when a tool or the API misbehaves. Offline: no server is started."""
from __future__ import annotations

import io
import json
import sys

from prophet_agent import mcp_server as m


def call(name: str, args: dict, mid: int = 1) -> dict:
    return m.handle({"jsonrpc": "2.0", "id": mid, "method": "tools/call", "params": {"name": name, "arguments": args}})


def run_loop(monkeypatch, lines: list[str], capsys) -> list[dict]:
    monkeypatch.setattr(sys, "stdin", io.StringIO("\n".join(lines) + "\n"))
    m.main()
    return [json.loads(line) for line in capsys.readouterr().out.splitlines()]


def test_an_unexpected_error_inside_a_tool_is_an_error_result_for_that_request(monkeypatch):
    # A non-numeric wait_seconds raises ValueError inside the handler. It used to escape handle() and the request got no answer.
    reply = call("prophet_status", {"turn_id": "t", "wait_seconds": "soon"}, mid=42)
    assert reply["id"] == 42
    assert reply["result"]["isError"] is True
    assert "prophet_status" in reply["result"]["content"][0]["text"]


def test_the_server_loop_survives_a_tool_error_and_answers_the_next_request(monkeypatch, capsys):
    bad = {"jsonrpc": "2.0", "id": 1, "method": "tools/call",
           "params": {"name": "prophet_status", "arguments": {"turn_id": "t", "wait_seconds": "soon"}}}
    ping = {"jsonrpc": "2.0", "id": 2, "method": "ping"}
    replies = run_loop(monkeypatch, [json.dumps(bad), json.dumps(ping)], capsys)
    assert [r["id"] for r in replies] == [1, 2]
    assert replies[0]["result"]["isError"] is True


def test_a_crash_outside_any_tool_still_answers_with_the_request_id(monkeypatch, capsys):
    def boom(_msg):
        raise RuntimeError("unexpected")

    monkeypatch.setattr(m, "handle", boom)
    replies = run_loop(monkeypatch, [json.dumps({"jsonrpc": "2.0", "id": 5, "method": "ping"})], capsys)
    assert replies == [{"jsonrpc": "2.0", "id": 5, "error": {"code": -32603, "message": "internal error"}}]


def test_a_crash_on_a_notification_writes_nothing(monkeypatch, capsys):
    def boom(_msg):
        raise RuntimeError("unexpected")

    monkeypatch.setattr(m, "handle", boom)
    assert run_loop(monkeypatch, [json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"})], capsys) == []
