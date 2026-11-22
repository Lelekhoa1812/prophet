"""Why a mission stopped short of a verified answer, in plain words for the person.

Motivation vs Logic
Motivation: a partial or stopped mission used to carry one label and the planner's note. A person could not tell a failed
check from an untraced figure or a spent allowance, and the planner's note named code (``ps.Ledger()``, tool names).
Logic: the reasons are read from the stored answer, the mission events and the budget. Each reason has one code, one
plain sentence and one next step. Check names map to plain phrases; the planner's own text is kept under ``developer``
and is never shown as a reason. Nothing here changes a status; it only explains the one the runtime recorded.
"""
from __future__ import annotations

from typing import Any

# What a failed check means, in words a person can act on. A check with no entry falls back to a generic phrase, so a
# new check name is never shown raw.
CHECK_PLAIN = {
    "analysis_ledger": "the analysis notes were not saved",
    "baseline_compared": "the forecast was not compared with a simple baseline",
    "beats_or_reports": "the answer does not say whether the forecast beats a simple baseline",
    "controls_tested": "the controls were not tested",
    "data_inspected": "the data was not inspected first",
    "data_scanned": "the data was not checked for problems",
    "deliverable_exists": "a promised file was not produced",
    "deliverable_hash": "a produced file could not be checksummed",
    "deliverable_kinds": "the files produced are not the kinds promised",
    "forecast_exists": "no forecast was produced",
    "forecast_finite": "the forecast has values that are not numbers",
    "forecast_length": "the forecast length does not match the request",
    "has_baselines": "no simple baseline was run for comparison",
    "health_repaired": "the forecast failed a shape check and was not fixed",
    "horizon_matches_request": "the forecast period differs from the one asked for",
    "horizon_present": "the forecast period is missing",
    "interval_order": "the likely range is not ordered from low to high",
    "metrics_finite": "an accuracy figure is not a number",
    "model_does_not_beat_baseline": "a simple baseline was more accurate than the forecast",
    "non_negative_when_history_is": "the forecast goes below zero where the history cannot",
    "plausible_scale": "the forecast is far from the scale of the history",
    "scan_decisions_recorded": "a data problem was found and no decision was recorded for it",
}
GENERIC_CHECK = "a required check did not pass"

# One next step per reason code. Every code the API can return has one (tested).
NEXT_STEP = {
    "blocked_by_check": "Ask again and name the piece that was missing, or continue the work so the failed checks can be fixed.",
    "unverified_figures": "Check the listed figures against the results. Ask again so they are worked out from the files.",
    "stalled": "Add detail to the request (the column to forecast, the time range, or what the numbers mean) and try again.",
    "answered_now": "Continue the work if the missing parts matter to you.",
    "budget_spent": "Continue to let Prophet keep going, or answer now with what it has found.",
    "not_settled": "Read the answer as partial. Ask again with more detail to close the gap.",
    "impossible": "Add more history or more detail, or change what you are asking for.",
    "failed": "Try again. If it keeps happening, rephrase the request or share your data in a different layout.",
    "outcome_unknown": "Check the result of that step yourself, then start a new request.",
    "cancelled": "Ask it again when you are ready, or change the question.",
}

USER_REASON = {
    "blocked_by_check": "Some required checks did not pass, so the answer is partial.",
    "unverified_figures": "Some figures in the answer could not be traced to the analysis results. They are listed below.",
    "stalled": "Prophet kept repeating the same step, so it stopped with what it had found.",
    "answered_now": "You asked for the answer now, so it was written from the evidence found so far.",
    "budget_spent": "This part of the work used up its allowance. Every step so far is kept.",
    "not_settled": "The checks did not all settle, so the answer says what was established and what was not.",
    "impossible": "The data or the request does not support a reliable forecast.",
    "failed": "Something went wrong while Prophet was working.",
    "outcome_unknown": "Prophet paused because a step may not have finished.",
    "cancelled": "You stopped this question. Nothing more was done after that.",
}

BUDGET_REASON = {
    "max_seconds": "This part of the work used its time allowance.",
    "max_tool_calls": "This part of the work used its tool-call allowance.",
}

STALL_MARK = "kept repeating the same step"


def _reason(code: str, **extra: Any) -> dict:
    return {"code": code, "user_reason": USER_REASON[code], "next_step": NEXT_STEP[code], **extra}


def _partial_reasons(answer: dict, budget: dict) -> list[dict]:
    out: list[dict] = []
    failed = [str(n) for n in answer.get("contract_failures") or [] if n]
    if failed:
        checks = [{"name": n, "plain": CHECK_PLAIN.get(n, GENERIC_CHECK)} for n in failed]
        out.append(_reason("blocked_by_check", checks=checks))
    figures = [str(f) for f in answer.get("unverified") or [] if f not in (None, "")]
    if figures:
        out.append(_reason("unverified_figures", figures=figures))
    reason_text = answer.get("reason") if isinstance(answer.get("reason"), str) else ""
    if STALL_MARK in reason_text:
        out.append(_reason("stalled"))
    if budget.get("finish_now"):
        out.append(_reason("answered_now"))
    if not out:
        out.append(_reason("not_settled"))
    return out


def outcome_for(status: str, answer: Any, budget: dict | None, events: list[dict]) -> dict:
    """The named reasons a mission stopped, its developer text, and nothing else.

    ``status`` is the mission status. ``answer`` is the stored answer (a dict, or None while running). ``events`` are the
    mission's events in order, so the last checkpoint names the allowance that ran out."""
    answer = answer if isinstance(answer, dict) else {}
    budget = budget or {}
    kind = str(answer.get("status") or status)
    reasons: list[dict] = []
    if status == "checkpoint":
        which = next((e["payload"].get("which") for e in reversed(events) if e.get("kind") == "mission.checkpoint"), None)
        reasons.append(_reason("budget_spent", which=which, user_reason=BUDGET_REASON.get(str(which), USER_REASON["budget_spent"])))
    elif kind == "partial":
        reasons.extend(_partial_reasons(answer, budget))
    elif kind == "stalled":
        reasons.append(_reason("stalled"))
    elif kind == "impossible":
        reasons.append(_reason("impossible"))
    elif kind == "failed":
        reasons.append(_reason("failed"))
    elif kind == "outcome_unknown":
        reasons.append(_reason("outcome_unknown"))
    elif kind == "cancelled":
        reasons.append(_reason("cancelled"))
    developer = {
        "status": kind,
        "stop_reason": answer.get("stop_reason"),
        "reason": answer.get("reason") if isinstance(answer.get("reason"), str) else None,
        "contract_failures": list(answer.get("contract_failures") or []),
    }
    return {"status": status, "reasons": reasons, "developer": developer}
