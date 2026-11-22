"""Round 3, B6: why a mission stopped short (A7-04) and what the person is told about it (A7-08).

The reasons are read from the stored answer, the events and the budget. Each reason has one code, one plain sentence and one
next step. The planner's own text is kept under ``developer`` and never shown as the reason.
"""
from __future__ import annotations

import re

import pytest
from fastapi.testclient import TestClient

from prophet_agent.api.app import create_app
from prophet_agent.api.service import MissionService
from prophet_agent.tools import default_tools

# The stored text of live missions (data/live_gates_wti_rep3.json and data/live_panel_beijing_rep1.json), copied here so the
# tests do not depend on the git-ignored data folder.
WTI_REP3_REASON = ("This answer did not pass every required check: the forecast keeps TimesFM's centre line, but the backtest "
                   "found mean_last_season more accurate (MASE 6.663 against 7.278); switch the centre to that rule "
                   "[baseline_compared]. Found blocked: the scan has blocker or major findings with no ledger decision: "
                   "time.dst_shift; record each one with ps.Ledger().add(stage, finding, decision) and put the finding id "
                   "in the text [scan_decisions_recorded]")
BEIJING_REP1_REASON = ("The forecast file lists Wanliu and Wanshouxigong as missing; the sandbox read shows they are NOT FOUND, "
                       "so calibrate_intervals runs on the other ten series.")
REPEAT_REASON = "The analysis kept repeating the same step, so this answer rests on what was found before that."

# Code-shaped text that must never reach a user-facing reason: Python calls, dotted names, tool names, shouting.
CODE_TOKEN = re.compile(r"\bps\.|\(\)|\w+\(\w*\)|\bcalibrate_\w+|\btimesfm_\w+|\bsandbox\b|\bNOT FOUND\b|\[\w+\]|\w+\.\w+\(")


def _client(tmp_path):
    service = MissionService(tmp_path / "api.sqlite", lambda: None, default_tools())
    return TestClient(create_app(service), base_url="http://127.0.0.1:8000"), service


def _mission(service, *, status: str, answer: dict | None, budget: dict | None = None, events=()):
    mid = service.store.create_mission("forecast the next 12 weeks", budget or {"max_tool_calls": 20})
    for kind, payload in events:
        service.store.append_event(mid, kind, payload)
    if answer is not None:
        service.store.set_answer(mid, answer)
    service.store.set_status(mid, status)
    return mid


def _reasons(client, mid) -> list[dict]:
    return client.get(f"/api/missions/{mid}").json()["outcome"]["reasons"]


def _codes(reasons) -> list[str]:
    return [r["code"] for r in reasons]


# -- A7-04: named partial reasons --------------------------------------------------------------------------------------

def test_a_partial_names_each_failed_check(tmp_path):
    client, service = _client(tmp_path)
    mid = _mission(service, status="partial", answer={
        "status": "partial", "reason": WTI_REP3_REASON, "stop_reason": "partial_unverified", "answer": "text",
        "contract_failures": ["baseline_compared", "scan_decisions_recorded"]})
    reasons = _reasons(client, mid)
    blocked = [r for r in reasons if r["code"] == "blocked_by_check"]
    assert len(blocked) == 1
    assert [c["name"] for c in blocked[0]["checks"]] == ["baseline_compared", "scan_decisions_recorded"]
    assert all(c["plain"] for c in blocked[0]["checks"])


def test_unverified_figures_are_their_own_reason(tmp_path):
    client, service = _client(tmp_path)
    mid = _mission(service, status="partial", answer={
        "status": "partial", "reason": "Some figures in this answer could not be matched to the analysis output.",
        "stop_reason": "partial_unverified", "answer": "text", "unverified": ["1,098", "28.1"]})
    reasons = _reasons(client, mid)
    assert _codes(reasons) == ["unverified_figures"]
    assert reasons[0]["figures"] == ["1,098", "28.1"]


def test_mixed_causes_are_both_named(tmp_path):
    client, service = _client(tmp_path)
    mid = _mission(service, status="partial", answer={
        "status": "partial", "reason": WTI_REP3_REASON, "stop_reason": "partial_unverified", "answer": "text",
        "contract_failures": ["baseline_compared"], "unverified": ["82"]})
    assert _codes(_reasons(client, mid)) == ["blocked_by_check", "unverified_figures"]


def test_a_repeated_step_is_named_stalled_not_a_check_failure(tmp_path):
    client, service = _client(tmp_path)
    mid = _mission(service, status="partial", answer={
        "status": "partial", "reason": REPEAT_REASON, "stop_reason": "partial_unverified", "answer": "text"})
    assert _codes(_reasons(client, mid)) == ["stalled"]


def test_an_answer_written_at_the_person_s_request_is_named(tmp_path):
    client, service = _client(tmp_path)
    mid = _mission(service, status="partial", budget={"max_tool_calls": 20, "finish_now": True}, answer={
        "status": "partial", "reason": "Two checks were not run.", "stop_reason": "partial_unverified", "answer": "text"})
    assert "answered_now" in _codes(_reasons(client, mid))


def test_a_checkpoint_names_the_allowance_that_ran_out(tmp_path):
    client, service = _client(tmp_path)
    mid = _mission(service, status="checkpoint", answer=None,
                   events=[("mission.checkpoint", {"which": "max_seconds"})])
    reasons = _reasons(client, mid)
    assert _codes(reasons) == ["budget_spent"]
    assert reasons[0]["which"] == "max_seconds"


def test_a_completed_mission_has_no_reasons(tmp_path):
    client, service = _client(tmp_path)
    mid = _mission(service, status="completed", answer={"status": "met", "answer": "all good", "stop_reason": "done_verified"})
    assert _reasons(client, mid) == []


def test_each_partial_has_a_reason_even_with_no_known_cause(tmp_path):
    client, service = _client(tmp_path)
    mid = _mission(service, status="partial", answer={"status": "partial", "reason": "Unclear.", "answer": "text"})
    assert _codes(_reasons(client, mid)) == ["not_settled"]


# -- A7-08: developer text stays out of the user-facing reason ---------------------------------------------------------

def test_the_user_reason_has_no_developer_instructions(tmp_path):
    client, service = _client(tmp_path)
    mid = _mission(service, status="partial", answer={
        "status": "partial", "reason": WTI_REP3_REASON, "stop_reason": "partial_unverified", "answer": "text",
        "contract_failures": ["baseline_compared", "scan_decisions_recorded"]})
    body = client.get(f"/api/missions/{mid}").json()
    for r in body["outcome"]["reasons"]:
        assert not CODE_TOKEN.search(r["user_reason"]), r["user_reason"]
        assert not CODE_TOKEN.search(r["next_step"]), r["next_step"]
        for c in r.get("checks", []):
            assert not CODE_TOKEN.search(c["plain"]), c["plain"]
    # The technical text is still there, for a developer, under its own key.
    assert "ps.Ledger" in body["outcome"]["developer"]["reason"]
    assert body["outcome"]["developer"]["contract_failures"] == ["baseline_compared", "scan_decisions_recorded"]


def test_the_beijing_reason_with_tool_names_is_replaced_by_plain_words(tmp_path):
    client, service = _client(tmp_path)
    mid = _mission(service, status="partial", answer={
        "status": "partial", "reason": BEIJING_REP1_REASON, "stop_reason": "partial_unverified", "answer": "text"})
    reasons = _reasons(client, mid)
    assert reasons and all(not CODE_TOKEN.search(r["user_reason"]) for r in reasons)
    assert all(r["user_reason"] for r in reasons)


def test_the_impossible_reason_is_plain_and_the_planner_text_is_kept_for_developers(tmp_path):
    client, service = _client(tmp_path)
    mid = _mission(service, status="impossible", answer={"status": "impossible", "reason": "no column named sales in the in/ folder"})
    body = client.get(f"/api/missions/{mid}").json()
    assert _codes(body["outcome"]["reasons"]) == ["impossible"]
    assert not CODE_TOKEN.search(body["outcome"]["reasons"][0]["user_reason"])
    assert body["outcome"]["developer"]["reason"] == "no column named sales in the in/ folder"


@pytest.mark.parametrize("code", ["blocked_by_check", "unverified_figures", "stalled", "answered_now", "not_settled"])
def test_every_reason_code_has_a_next_step(tmp_path, code):
    from prophet_agent.api.outcome import NEXT_STEP

    assert NEXT_STEP[code]
