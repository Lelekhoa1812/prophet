"""Offline mission evals: recorded missions replayed through the real goal loop (docs/prophet/EVALS.md).

Each fixture in fixtures/ holds one recorded mission: its goal, the planner and eval decisions in the order they were asked,
the tool results keyed by tool and arguments, and the checks its outcome must pass. Every check is its own test, so a failure
names the fixture, the check and why it matters. No model, no network: the loop, runtime, store, stall rule and guards are real.

Motivation vs Logic
Motivation: harness changes (stall rule, guards, repairs) must be checked against what real missions did, in the offline gate
(scripts/check.sh), without NIM keys or an hour of live runs. The EPL re-parse loop and the Apple file saved as NVDA are the two
failures the Mission 9 batch is meant to fix.
Logic: each fixture is replayed once (cached) into a temporary store. A check reads only the outcome of that replay: the final
status and stop reason, the number of repeats before the stop, the named files and the refused downloads. A check that fails is
a finding about the code or the fixture, and is reported, not weakened.
"""
from __future__ import annotations

import json
import sqlite3
import sys
import tempfile
from functools import lru_cache
from pathlib import Path
from typing import Any

import pytest

from prophet_agent.runtime.store import Store
from tests.prophet_agent.evals.replay import FIXTURES, Outcome, load, replay

ROOT = Path(__file__).resolve().parents[3]
BUILDER = ROOT / "scripts" / "build_eval_fixture.py"

BUILT_IN_CHECKS = [{"check": "no_unrecorded_calls",
                    "why": "every call the replayed loop made had a recorded result; otherwise the replay has diverged"}]


def _fixture_ids() -> list[str]:
    return sorted(p.stem for p in FIXTURES.glob("*.json"))


@lru_cache(maxsize=None)
def _replayed(fixture_id: str) -> tuple[dict, Outcome]:
    fixture = load(FIXTURES / f"{fixture_id}.json")
    with tempfile.TemporaryDirectory() as tmp:
        store = Store(Path(tmp) / "replay.sqlite")
        try:
            outcome = replay(fixture, store)
        finally:
            store.close()
    return fixture, outcome


def _cases() -> list[tuple[str, dict]]:
    cases = []
    for fixture_id in _fixture_ids():
        fixture = load(FIXTURES / f"{fixture_id}.json")
        own = list(fixture.get("checks", []))
        named = {spec["check"] for spec in own}
        for spec in own + [b for b in BUILT_IN_CHECKS if b["check"] not in named]:    # a fixture may set its own
            cases.append((fixture_id, spec))
    return cases


# ---- checks: each returns None when it passes, or the reason it fails ----------------------------------------------------

CHECKS: dict[str, Any] = {}


def _check(name: str):
    def register(fn):
        CHECKS[name] = fn
        return fn
    return register


def _executed(o: Outcome, tool: str | None = None) -> list[dict]:
    return [e for e in o.tool_log if e["ok"] and (tool is None or e["tool"] == tool)]


def _stdout(entry: dict) -> str:
    return str((entry.get("result") or {}).get("stdout") or "")


@_check("no_unrecorded_calls")
def _(spec: dict, o: Outcome) -> str | None:
    allowed = int(spec.get("allowed", 0))
    if len(o.misses) <= allowed:
        return None
    first = o.misses[0]
    return f"{len(o.misses)} call(s) had no recorded result; first: {first['tool']} {json.dumps(first['args'])[:160]}"


@_check("final_status_in")
def _(spec: dict, o: Outcome) -> str | None:
    if o.result is None:
        return f"the mission produced no result (runtime status {o.runtime_status}, {o.raw.get('error', 'no error text')})"
    return None if o.status in spec["values"] else f"final status {o.status!r} is not one of {spec['values']}"


@_check("stop_reason_in")
def _(spec: dict, o: Outcome) -> str | None:
    if o.result is None:
        return f"the mission produced no result (runtime status {o.runtime_status})"
    got = o.result.get("stop_reason")
    return None if got in spec["values"] else f"stop reason {got!r} is not one of {spec['values']}"


@_check("runs_before_step")
def _(spec: dict, o: Outcome) -> str | None:
    if o.result is None:
        return f"the mission produced no result (runtime status {o.runtime_status})"
    steps = o.result.get("iterations")
    if steps is None:
        return "the result does not say how many steps ran"
    return None if steps < spec["step"] else f"the loop ran {steps} steps, not fewer than {spec['step']}"


@_check("stall_within_repeats")
def _(spec: dict, o: Outcome) -> str | None:
    matches = [e for e in _executed(o, spec["tool"]) if all(m in _stdout(e) for m in spec["contains"])]
    if not matches:
        return f"no executed {spec['tool']} call printed {spec['contains']}, so there was no re-parse to repeat"
    if o.exhausted:
        return (f"the loop asked for more planner decisions after the recording ran out ({o.plan_calls} asked, "
                f"{len(matches)} re-parses ran): the stall rule did not stop it")
    repeats = len(matches) - 1
    if repeats > spec["max_repeats"]:
        return f"{repeats} repeats of the re-parse ran before the stop (limit {spec['max_repeats']})"
    return None


def _lead_stdout(h: dict) -> str:
    res = h.get("full") if isinstance(h.get("full"), dict) else h.get("result")
    return str(res.get("stdout") or "") if isinstance(res, dict) else ""


@_check("first_stall_within_repeats")
def _(spec: dict, o: Outcome) -> str | None:
    """The harness's own part of M9-7: the stall rule fires (a rescue or a stop) within max_repeats repeats of the re-parse.
    What a rescue analyst does next comes from the recording, so it is checked separately (rescues_at_most)."""
    if o.result is None:
        return f"the mission produced no result (runtime status {o.runtime_status})"
    evidence = o.result.get("evidence") or []
    stop = next((i for i, h in enumerate(evidence) if h.get("kind") == "stall_rescue"), len(evidence))
    hits = [h for h in evidence[:stop] if h.get("kind") == "tool" and h.get("tool") == spec["tool"]
            and all(m in _lead_stdout(h) for m in spec["contains"])]
    if not hits:
        return f"no {spec['tool']} call before the first stall printed {spec['contains']}"
    if stop == len(evidence) and o.result.get("status") != "partial":
        return "the stall rule never fired"
    repeats = len(hits) - 1
    return None if repeats <= spec["max_repeats"] else (
        f"{repeats} repeats of the re-parse ran before the first stall (limit {spec['max_repeats']})")


@_check("rescues_at_most")
def _(spec: dict, o: Outcome) -> str | None:
    if o.result is None:
        return f"the mission produced no result (runtime status {o.runtime_status})"
    rescues = sum(1 for h in o.result.get("evidence") or [] if h.get("kind") == "stall_rescue")
    return None if rescues <= spec["count"] else f"{rescues} stall rescues ran (limit {spec['count']})"


@_check("no_unverified")
def _(spec: dict, o: Outcome) -> str | None:
    if o.result is None:
        return f"the mission produced no result (runtime status {o.runtime_status})"
    bad = o.result.get("unverified")
    return None if not bad else f"figures not found in the analysis output: {list(bad)[:5]}"


@_check("file_named")
def _(spec: dict, o: Outcome) -> str | None:
    name = spec["name"]
    present = any(str(out.get("path", "")).endswith(name)
                  for e in _executed(o) for out in ((e.get("result") or {}).get("outputs") or []))
    if spec.get("present", True):
        return None if present else f"no executed call wrote a sandbox output named {name}"
    return f"a sandbox output named {name} exists, but the check expects none" if present else None


@_check("download_refused")
def _(spec: dict, o: Outcome) -> str | None:
    calls = [e for e in o.tool_log if e["tool"] == spec["tool"] and (e["args"] or {}).get("name") == spec["name"]]
    if any(e["ok"] for e in calls):
        return f"a download named {spec['name']} was accepted"
    refused = [e for e in calls if not e["ok"] and all(s in str(e["error"]) for s in spec["reason_contains"])]
    if refused:
        return None
    seen = [str(e["error"])[:120] for e in calls]
    return f"no refusal of {spec['name']} mentions {spec['reason_contains']}; errors seen: {seen or 'none'}"


@_check("no_download_accepted")
def _(spec: dict, o: Outcome) -> str | None:
    accepted = [e for e in o.tool_log if e["tool"] == "download_web_data" and (e["args"] or {}).get("name") == spec["name"]
                and e["ok"]]
    return f"{len(accepted)} download(s) named {spec['name']} were accepted" if accepted else None


@_check("max_eval_calls")
def _(spec: dict, o: Outcome) -> str | None:
    return None if o.eval_calls <= spec["value"] else f"the evaluator was asked {o.eval_calls} times (limit {spec['value']})"


# ---- tests ---------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("fixture_id,spec", _cases(), ids=lambda v: v if isinstance(v, str) else v["check"])
def test_recorded_mission_check(fixture_id: str, spec: dict):
    assert spec["check"] in CHECKS, f"unknown check {spec['check']!r} in {fixture_id}"
    _, outcome = _replayed(fixture_id)
    problem = CHECKS[spec["check"]](spec, outcome)
    assert problem is None, (f"{fixture_id}: check {spec['check']} failed: {problem}\n"
                             f"  why it matters: {spec.get('why', '(no note)')}")


def test_every_fixture_has_checks_and_a_recorded_goal():
    for fixture_id in _fixture_ids():
        fixture = load(FIXTURES / f"{fixture_id}.json")
        assert fixture.get("checks"), f"{fixture_id} has no checks"
        assert fixture.get("goal"), f"{fixture_id} has no goal"
        assert fixture["decisions"]["plan"], f"{fixture_id} has no recorded planner decisions"


def test_builder_reads_a_database_read_only(tmp_path):
    """The builder extracts a fixture from a stored mission, and the database it reads cannot be written through its connection."""
    sys.path.insert(0, str(ROOT / "scripts"))
    try:
        import build_eval_fixture as builder
    finally:
        sys.path.pop(0)
    db = tmp_path / "stored.sqlite"
    con = sqlite3.connect(db)
    con.executescript("""
        create table missions (id text primary key, goal text, status text, budget_json text, answer_json text);
        create table steps (mission_id text, idx int, name text, status text, output_json text);
        create table tool_calls (mission_id text, tool text, status text, args_json text, result_json text, error text,
                                 updated real);
        insert into missions values ('m1', 'goal text', 'partial', '{}', '{"status": "partial", "iterations": 2}');
        insert into steps values ('m1', 0, 'plan:d0:0', 'done', '{"v": {"action": "call_tool", "out_of_calls": false}}');
        insert into steps values ('m1', 1, 'eval:d0:1', 'done', '{"v": {"verdict": "not_met"}}');
        insert into steps values ('m1', 2, 'plan:d1.a1:0', 'done', '{"v": {"action": "finish"}}');
        insert into tool_calls values ('m1', 'web_search', 'done', '{"q": 1}', '{"hits": []}', null, 1.0);
        insert into tool_calls values ('m1', 'ask_user', 'done', '{"questions": []}', '{"answers": {"A": "B"}}', null, 2.0);
        """)
    con.commit()
    con.close()
    fixture = builder.build(db, "m1", "tiny", [])
    assert fixture["decisions"]["plan"] == [{"action": "call_tool"}, {"action": "finish"}]
    assert fixture["decisions"]["eval"] == [{"verdict": "not_met"}]
    assert [r["tool"] for r in fixture["results"]] == ["web_search"]
    assert fixture["answers"] == [{"A": "B"}]
    ro = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    with pytest.raises(sqlite3.OperationalError):
        ro.execute("update missions set status = 'changed'")
    ro.close()


def test_the_rescue_analyst_is_told_what_the_lead_repeated():
    """EPL 56465803: after the first stall, the fresh analyst's brief names the repeated step and asks for another route."""
    _, outcome = _replayed("epl_56465803")
    evidence = (outcome.result or {}).get("evidence") or []
    rescue = next(h for h in evidence if h.get("kind") == "stall_rescue")
    assert "sandbox_python" in rescue.get("repeated", "")
    briefs = json.dumps([h for h in evidence if h.get("kind") == "subagent"])
    assert "The lead stalled repeating" in briefs and "different route" in briefs
