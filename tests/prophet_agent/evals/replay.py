"""Offline replay of a recorded mission through the real goal loop (see docs/prophet/EVALS.md).

The planner, eval and answer decisions come from a fixture in order; the tool results come from the same fixture, keyed by
tool and arguments. The goal loop, the runtime, the store, the stall rule and the tool guards are the real code. Nothing
calls a model or the network.

Motivation vs Logic
Motivation: a change to the harness (stall rule, guard, repair) has to be checked against a real mission, but the live panel
needs NIM keys, the network and an hour of wall time. A replay gives the same loop the same decisions and the same tool
results, so the outcome can be compared with the recorded one in seconds.
Logic: ReplayBrain pops the recorded decisions per kind and raises ScriptExhausted when the loop asks for more than the
recording holds, which means the loop did not stop where the live run did. ReplayTools serves each call the recorded result
for its tool and arguments; a repeated call gets the next recorded result for that key, and the last one after that. A call
with no recorded result is a miss, reported to the test, not guessed. The download tool re-runs the current entity guard
(name_entity_refusal) on the table the recorded result describes, so a guard change shows up in the replay. Write (free-text
composition and memory compaction) is not replayed: the answer is the verdict's draft, which the checks say so about.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from prophet_agent.runtime.goal import GoalConfig, goal_program
from prophet_agent.runtime.runner import Runtime, Tool
from prophet_agent.runtime.store import Store
from prophet_agent.tools.opendata_tools import name_entity_refusal

FIXTURES = Path(__file__).parent / "fixtures"


class ScriptExhausted(Exception):
    """The loop asked the brain for a decision the recording does not hold: the replay ran past the live stop."""


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _unordered(value: Any) -> Any:
    """The loop builds its carry list (files carried into a cell) from a set, so its order differs from run to run. A list
    of {"file": ...} entries is compared as a set; every other list keeps its order."""
    if isinstance(value, dict):
        return {k: _unordered(v) for k, v in value.items()}
    if isinstance(value, list):
        items = [_unordered(v) for v in value]
        if items and all(isinstance(i, dict) and "file" in i for i in items):
            return sorted(items, key=lambda i: json.dumps(i, sort_keys=True))
        return items
    return value


def _canon(value: Any) -> str:
    return json.dumps(_unordered(value), sort_keys=True, ensure_ascii=False, default=str)


class ReplayBrain:
    """Recorded decisions, in order, per kind. Same call signature as the live brains (decide(messages, schema, name))."""

    def __init__(self, decisions: dict):
        self._queues = {"plan": list(decisions.get("plan", [])), "eval": list(decisions.get("eval", []))}
        self.calls = {"plan": 0, "eval": 0}
        self.exhausted: str | None = None      # the kind the loop asked for after the recording ran out

    def decide(self, messages: list[dict], schema: dict, name: str) -> dict:
        self.calls[name] = self.calls.get(name, 0) + 1
        queue = self._queues[name]
        if not queue:
            # The runtime turns any exception into a failed mission, so the brain records the overrun itself.
            self.exhausted = self.exhausted or name
            raise ScriptExhausted(name)
        return queue.pop(0)


class ReplayTools:
    """Tool objects that answer from the recording. ``log`` has one entry per call the loop made; ``misses`` lists calls
    with no recorded result (a divergence from the live run, which the test reports)."""

    def __init__(self, results: list[dict]):
        self._by_key: dict[tuple[str, str], list[dict]] = {}
        self._by_code: dict[tuple[str, str], list[dict]] = {}
        for rec in results:
            self._by_key.setdefault((rec["tool"], _canon(rec["args"])), []).append(rec)
            if rec["tool"] == "sandbox_python":
                self._by_code.setdefault((rec["tool"], _canon(_without_carry(rec["args"]))), []).append(rec)
        self._served: dict[tuple[str, str], int] = {}
        self.names = sorted({rec["tool"] for rec in results})
        self.log: list[dict] = []
        self.misses: list[dict] = []

    def tools(self) -> dict[str, Tool]:
        return {name: Tool(name, self._serve_for(name), idempotent=True) for name in self.names}

    def _serve_for(self, name: str):
        def serve(args: dict, _key: str) -> Any:
            return self._serve(name, args)
        return serve

    def _serve(self, name: str, args: dict) -> Any:
        key = (name, _canon(args))
        options = self._by_key.get(key)
        if not options and name == "sandbox_python":
            # The carry list is built by the harness (_auto_carry) from the run ids before the call, so it changes when a
            # harness change moves a stop or a rescue. The planner's own decision is the code: match on that.
            key = (name, _canon(_without_carry(args)))
            options = self._by_code.get(key)
        if not options:
            self.misses.append({"tool": name, "args": args})
            raise RuntimeError(f"replay: no recorded result for {name} with these arguments")
        n = self._served.get(key, 0)
        self._served[key] = n + 1
        rec = options[min(n, len(options) - 1)]
        if not rec["ok"]:
            self.log.append({"tool": name, "args": args, "ok": False, "error": rec["error"], "result": None})
            raise RuntimeError(rec["error"])
        result = rec["result"]
        if name == "download_web_data":
            try:
                _guard_download(args, result)      # the current entity guard, on the table the recording describes
            except ValueError as exc:
                self.log.append({"tool": name, "args": args, "ok": False, "error": str(exc), "result": None})
                raise
        self.log.append({"tool": name, "args": args, "ok": True, "error": None, "result": result})
        return result


def _without_carry(args: Any) -> Any:
    return {k: v for k, v in args.items() if k != "carry"} if isinstance(args, dict) else args


def _guard_download(args: dict, result: Any) -> None:
    """Re-runs the name guard of download_web_data on the recorded table: its header and first rows, which carry the
    column tickers the guard reads. A caller name that names another ticker is refused with the same reason the tool gives."""
    name = args.get("name")
    columns = result.get("columns") if isinstance(result, dict) else None
    if not isinstance(name, str) or not name.strip() or not columns:
        return
    lines = [",".join(str(c) for c in columns)]
    lines += [",".join(str(v) for v in row) for row in result.get("first_rows", [])]
    reason = name_entity_refusal(name, "\n".join(lines).encode("utf-8"), str(args.get("url", "")))
    if reason:
        raise ValueError(reason)


@dataclass
class Outcome:
    fixture_id: str
    runtime_status: str | None = None
    result: dict | None = None
    exhausted: str | None = None           # the decision kind that ran out, if the loop went past the recording
    plan_calls: int = 0
    eval_calls: int = 0
    answers_given: int = 0
    tool_log: list[dict] = field(default_factory=list)
    misses: list[dict] = field(default_factory=list)
    mission_id: str = ""
    raw: dict = field(default_factory=dict)  # the runtime's own return value, for diagnosis

    @property
    def status(self) -> str | None:
        return None if self.result is None else self.result.get("status")


def replay(fixture: dict, store: Store) -> Outcome:
    """Runs the fixture's mission through goal_program with the recorded decisions and results, answering each question the
    loop opens with the next recorded answer, and returns what happened."""
    tools = ReplayTools(fixture["results"])
    brain = ReplayBrain(fixture["decisions"])
    cfg = GoalConfig(**fixture.get("config", {}))
    program = goal_program(fixture["goal"], tools.tools(), brain, cfg)
    mid = store.create_mission(f"replay {fixture['id']}", {})
    out = Outcome(fixture_id=fixture["id"], mission_id=mid)
    answers = list(fixture.get("answers", []))
    runtime_out: dict = {}
    try:
        runtime_out = Runtime(store, owner="replay").run(mid, program)
        while runtime_out.get("status") == "waiting_user" and answers:
            qid = store.open_questions(mid)[0]["id"]
            store.answer_question(qid, answers.pop(0))
            out.answers_given += 1
            runtime_out = Runtime(store, owner="replay").run(mid, program)
    except ScriptExhausted as exc:
        out.exhausted = str(exc)
    out.exhausted = brain.exhausted or out.exhausted
    out.raw = runtime_out
    out.runtime_status = runtime_out.get("status")
    out.result = runtime_out.get("result")
    out.plan_calls = brain.calls["plan"]
    out.eval_calls = brain.calls["eval"]
    out.tool_log = tools.log
    out.misses = tools.misses
    return out
