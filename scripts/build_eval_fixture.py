#!/usr/bin/env python3
"""Extract an offline mission-eval fixture from a stored mission (read-only).

Usage (from a worktree, point --db at the main checkout's database):

    PYTHONPATH=src .venv/bin/python scripts/build_eval_fixture.py \
        --db /Users/liamle/Downloads/prophet/data/prophet.sqlite \
        --mission 56465803efa547009cc7e054312ea7ab \
        --checks tests/prophet_agent/evals/checks/epl_56465803.json \
        --id epl_56465803 --out tests/prophet_agent/evals/fixtures/epl_56465803.json

Motivation vs Logic
Motivation: a harness change (stall rule, guards, repair) must be checked against what a real mission did, without a live
model or the network. The fixture keeps the main planner's decisions in order, the eval and compose verdicts, the answers
the person gave, and the recorded tool results, so the real goal loop can be replayed offline.
Logic: the database is opened read-only (mode=ro), so nothing is written to it. Only depth-0 steps are kept (sub-agent steps
have their own history and are not replayed). Decisions are taken from the stored step payloads, minus the out_of_calls
flag the loop adds itself. Tool results come from the lead lane of tool_calls, keyed by tool and arguments in the order
they ran, and are trimmed to what the checks need (long text cut, long lists shortened, list fields the guards read kept
whole). The live outcome is stored as ``recorded`` for comparison only; no check reads it.
"""
from __future__ import annotations

import argparse
import json
import re
import sqlite3
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
DEFAULT_DB = REPO / "data" / "prophet.sqlite"
DEFAULT_OUT_DIR = REPO / "tests" / "prophet_agent" / "evals" / "fixtures"

STDOUT_CHARS = 4000        # sandbox stdout kept up to this many characters (the stall rule reads the table rows)
TEXT_CHARS = 240           # other long strings are cut to this many characters (web text, passages)
LIST_ITEMS = 3             # other long lists keep their first items (hits, passages)
WHOLE_LISTS = {"columns", "outputs", "first_rows"}   # fields the guards and the stall rule read: never shortened
# Brain decisions by step name. Depth-0 steps and the sub-agent steps they start (d1.a1, d1.a2, ...) are kept: a sub-agent
# runs inside the lead's step numbering, so the stored step order is the order in which the brain was asked. Free-text
# composition (compose, composestall) is not kept: the replay does not call write().
DECISION = re.compile(r"^(plan|eval|evalstall|eval2|repair):(d0|d1\.a\d+):(\d+)$")
PLAN_KINDS = {"plan"}
EVAL_KINDS = {"eval", "evalstall", "eval2", "repair"}
TRIM_NOTE = "_trimmed"


def _trim(value: Any, key: str | None = None, text: int = TEXT_CHARS, items: int = LIST_ITEMS) -> Any:
    """A JSON value cut to what the checks need. Shape is kept: dicts stay dicts, lists stay lists of the same kind of item."""
    if isinstance(value, str):
        limit = STDOUT_CHARS if key == "stdout" else text
        if len(value) <= limit:
            return value
        return value[:limit] + f"...[{len(value) - limit} characters trimmed]"
    if isinstance(value, list):
        kept = [_trim(v, key, text, items) for v in value]
        if key in WHOLE_LISTS or len(kept) <= items:
            return kept
        return kept[:items]
    if isinstance(value, dict):
        out = {k: _trim(v, k, text, items) for k, v in value.items()}
        if any(out.get(k) != value.get(k) for k in value):
            out[TRIM_NOTE] = True
        return out
    return value


def _loads(text: str | None) -> Any:
    return None if text is None else json.loads(text)


def _open_readonly(path: Path) -> sqlite3.Connection:
    """The database is opened read-only; a write attempt fails with an error instead of changing the file."""
    if not path.exists():
        raise SystemExit(f"database not found: {path}")
    return sqlite3.connect(f"file:{path}?mode=ro", uri=True)


def build(db: Path, mission_id: str, fixture_id: str, checks: list[dict]) -> dict:
    con = _open_readonly(db)
    try:
        row = con.execute("select goal, status, budget_json, answer_json from missions where id = ?", (mission_id,)).fetchone()
        if row is None:
            raise SystemExit(f"no mission {mission_id} in {db}")
        goal, status, budget_json, answer_json = row
        answer = _loads(answer_json) or {}

        decisions: dict[str, list] = {"plan": [], "eval": []}
        steps = 0
        for name, step_status, output in con.execute(
                "select name, status, output_json from steps where mission_id = ? order by idx", (mission_id,)):
            steps += 1
            m = DECISION.match(name)
            if m is None or step_status != "done":
                continue
            kind, value = m.group(1), _loads(output)
            value = value.get("v") if isinstance(value, dict) else value
            if kind in PLAN_KINDS:
                value = {k: v for k, v in value.items() if k != "out_of_calls"}
                decisions["plan"].append(value)
            elif kind in EVAL_KINDS:
                decisions["eval"].append(value)

        answers: list[dict] = []
        results: list[dict] = []
        for tool, tool_status, args_json, result_json, error in con.execute(
                "select tool, status, args_json, result_json, error from tool_calls "
                "where mission_id = ? order by updated, rowid", (mission_id,)):
            args = _loads(args_json) or {}
            if tool == "ask_user":
                if tool_status == "done":
                    answers.append((_loads(result_json) or {}).get("answers", {}))
                continue
            if tool_status == "done":
                # sandbox cells keep their stdout (the stall rule reads it); other tools keep a short view of their result
                small = {} if tool == "sandbox_python" else {"text": TEXT_CHARS // 2, "items": 2}
                results.append({"tool": tool, "args": args, "ok": True,
                                "result": _trim(_loads(result_json), **small)})
            elif tool_status == "failed":
                results.append({"tool": tool, "args": args, "ok": False, "error": error or "failed"})
            # rejected and started calls are not replayed: a rejected call is not charged, a started one has no outcome

        return {
            "about": ("Offline mission-eval fixture (see docs/prophet/EVALS.md). Built by scripts/build_eval_fixture.py from a "
                      "stored mission, read-only. Decisions are the planner and eval outputs in the order the brain was asked "
                      "(depth 0 and sub-agents); results are the tool calls of every lane, in run order, trimmed. Replayed "
                      "numbers are recorded tool output, not new observations."),
            "id": fixture_id,
            "source": {"mission_id": mission_id, "stored_status": status, "stored_budget": _loads(budget_json)},
            "goal": goal,
            "config": {"auto_delegate": True, "stall_window": 3},
            "decisions": decisions,
            "answers": answers,
            "results": results,
            "recorded": {"status": answer.get("status"), "stop_reason": answer.get("stop_reason"),
                         "iterations": answer.get("iterations"), "stored_steps": steps,
                         "note": "what the live run did; for comparison only, no check reads it"},
            "checks": checks,
        }
    finally:
        con.close()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--db", type=Path, default=DEFAULT_DB, help="stored database (opened read-only)")
    ap.add_argument("--mission", required=True, help="mission id")
    ap.add_argument("--id", dest="fixture_id", required=True, help="short fixture name, e.g. epl_56465803")
    ap.add_argument("--checks", type=Path, required=True, help="JSON list of expected-outcome checks")
    ap.add_argument("--out", type=Path, help="output file (default: tests/prophet_agent/evals/fixtures/<id>.json)")
    args = ap.parse_args(argv)
    checks = json.loads(args.checks.read_text(encoding="utf-8"))
    fixture = build(args.db.expanduser(), args.mission, args.fixture_id, checks)
    out = args.out or (DEFAULT_OUT_DIR / f"{args.fixture_id}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(fixture, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {out} ({out.stat().st_size} bytes): {len(fixture['decisions']['plan'])} plan, "
          f"{len(fixture['decisions']['eval'])} eval, {len(fixture['results'])} tool results, "
          f"{len(fixture['answers'])} answers")
    return 0


if __name__ == "__main__":
    sys.exit(main())
