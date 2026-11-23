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


def test_a_read_timeout_from_the_local_api_is_reported_as_no_answer(monkeypatch):
    def slow(*_args, **_kwargs):
        raise TimeoutError("timed out")

    monkeypatch.setattr(m.urllib.request, "urlopen", slow)
    reply = call("prophet_notebook", {"notebook_id": "n1"})
    text = reply["result"]["content"][0]["text"]
    assert reply["result"]["isError"] is True
    assert "did not answer" in text and "TimeoutError" not in text


def test_a_response_with_an_unexpected_shape_is_not_reported_as_a_missing_argument(monkeypatch):
    # The API answered without a 'payload' key. The caller sent every argument, so "missing argument" would mislead.
    monkeypatch.setattr(m, "_call", lambda *a, **k: {"id": "t", "status": "waiting_user", "questions": [{"id": "q"}]})
    reply = call("prophet_status", {"turn_id": "t"})
    text = reply["result"]["content"][0]["text"]
    assert reply["result"]["isError"] is True
    assert "missing argument" not in text


def test_a_call_without_a_required_argument_names_it(monkeypatch):
    reply = call("prophet_ask", {"question": "what sold?"})
    assert reply["result"]["isError"] is True
    assert "missing argument 'notebook_id'" in reply["result"]["content"][0]["text"]


def test_a_line_that_is_not_json_gets_a_parse_error(monkeypatch, capsys):
    replies = run_loop(monkeypatch, ["{not json"], capsys)
    assert replies == [{"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": "parse error: the line is not JSON"}}]


def test_every_argument_a_handler_reads_is_declared_in_its_schema():
    """The schema is what clients see; a handler reading an undeclared name would never receive it from a client."""
    import ast
    import inspect

    tree = ast.parse(inspect.getsource(m))
    read: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name) and node.value.id == "a" \
                and isinstance(node.slice, ast.Constant) and isinstance(node.slice.value, str):
            read.add(node.slice.value)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "get" \
                and isinstance(node.func.value, ast.Name) and node.func.value.id == "a" and node.args \
                and isinstance(node.args[0], ast.Constant):
            read.add(node.args[0].value)
    declared: set[str] = set()
    for name, (_desc, schema, _fn) in m.TOOLS.items():
        assert set(schema["required"]) <= set(schema["properties"]), name
        declared |= set(schema["properties"])
    assert read <= declared, f"handlers read undeclared arguments: {sorted(read - declared)}"
