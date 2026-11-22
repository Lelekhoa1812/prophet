"""Round 3, B6: the MCP bridge speaks the same plain reasons as the console (A7-08) and can continue a checkpointed mission
the way the HTTP API does (A7-06). The HTTP layer is stubbed; the reasons come from the real outcome_for."""
from __future__ import annotations

import json

from prophet_agent import mcp_server as m
from prophet_agent.api.outcome import outcome_for

WTI_REASON = ("This answer did not pass every required check: the backtest found mean_last_season more accurate; record each "
              "one with ps.Ledger().add(stage, finding, decision) [scan_decisions_recorded]")


def rpc(name, args, mid=1):
    return m.handle({"jsonrpc": "2.0", "id": mid, "method": "tools/call", "params": {"name": name, "arguments": args}})


def _mission(status, answer=None, events=(), budget=None):
    return {"id": "t1", "status": status, "answer": answer, "questions": [],
            "outcome": outcome_for(status, answer, budget or {}, list(events))}


def test_a_partial_answer_gives_plain_reasons_and_no_developer_text(monkeypatch):
    answer = {"status": "partial", "answer": "The forecast is partial.", "reason": WTI_REASON,
              "contract_failures": ["scan_decisions_recorded"], "stop_reason": "partial_unverified"}
    monkeypatch.setattr(m, "_call", lambda *a, **k: _mission("partial", answer))
    text = rpc("prophet_status", {"turn_id": "t1", "wait_seconds": 1})["result"]["content"][0]["text"]
    assert "ps.Ledger" not in text
    assert "scan_decisions_recorded" not in text
    out = json.loads(text)
    assert out["partial"] and "required checks" in out["partial"]
    assert out["reasons"][0]["code"] == "blocked_by_check"
    assert out["reasons"][0]["next"]


def test_a_checkpoint_tells_the_agent_how_to_continue_or_answer_now(monkeypatch):
    events = [{"kind": "mission.checkpoint", "payload": {"which": "max_tool_calls"}}]
    monkeypatch.setattr(m, "_call", lambda *a, **k: _mission("checkpoint", None, events))
    out = json.loads(rpc("prophet_status", {"turn_id": "t1", "wait_seconds": 1})["result"]["content"][0]["text"])
    assert out["status"] == "checkpoint"
    assert "prophet_continue" in out["next"] and "prophet_answer_now" in out["next"]


def test_continue_posts_to_the_continue_route_as_the_http_api_does(monkeypatch):
    calls = []

    def fake(method, path, body=None, **kw):
        calls.append((method, path))
        return {"id": "t1", "status": "running"}

    monkeypatch.setattr(m, "_call", fake)
    r = rpc("prophet_continue", {"turn_id": "t1", "wait_seconds": 0})["result"]
    assert r["isError"] is False
    assert ("POST", "/api/missions/t1/continue") in calls


def test_answer_now_posts_to_the_answer_now_route(monkeypatch):
    calls = []

    def fake(method, path, body=None, **kw):
        calls.append((method, path))
        return {"id": "t1", "status": "running"}

    monkeypatch.setattr(m, "_call", fake)
    r = rpc("prophet_answer_now", {"turn_id": "t1", "wait_seconds": 0})["result"]
    assert r["isError"] is False
    assert ("POST", "/api/missions/t1/answer-now") in calls


def test_continue_and_answer_now_are_listed_as_tools():
    names = set(m.TOOLS)
    assert {"prophet_continue", "prophet_answer_now"} <= names


def test_a_stopped_mission_without_an_answer_sends_reasons_not_the_raw_answer(monkeypatch):
    answer = {"status": "impossible", "reason": "no column named sales; ps.Ledger() missing"}
    monkeypatch.setattr(m, "_call", lambda *a, **k: _mission("impossible", answer))
    text = rpc("prophet_status", {"turn_id": "t1", "wait_seconds": 1})["result"]["content"][0]["text"]
    assert "ps.Ledger" not in text and "no column named sales" not in text
    assert json.loads(text)["reasons"][0]["code"] == "impossible"
