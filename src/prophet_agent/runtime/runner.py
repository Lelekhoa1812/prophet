"""Mission runner: replays a deterministic program against the ledger, so a crash resumes without repeating work.

Motivation vs Logic
Motivation: an agent program calls the model and tools many times. If the process dies, the next run must continue
from the last finished step, must not re-run a tool that already succeeded, and must not guess at a tool whose outcome
was never recorded.
Logic: the program is ordinary Python that calls ``ctx.step(name, fn)`` and ``ctx.call(tool, args)``. Steps are
numbered by call order, so a replay reaches the same numbers. A finished step returns its stored value without running
``fn``. A tool call is keyed by (mission, step, tool, args); a recorded result is returned as is. A call left in
"started" is re-run only if the tool is declared idempotent, and then it receives the same key so the external system
can deduplicate. Otherwise the mission stops with ``outcome_unknown`` for a human to decide. ``ctx.ask`` records an
open question and raises ``Paused``; answering it resolves the call, and the next run continues past it.

Crashes are simulated with a BaseException subclass so that the runner's cleanup does not run, which matches a real
kill: the lease and the status stay as they were, and a later runtime must wait for the lease to expire.
"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from typing import Any, Callable

from prophet_agent.nim.errors import redact

from .store import LeaseHeld, OutcomeUnknown, Store, idem_key


class Paused(Exception):
    """The program is waiting for the user to answer a question."""

    def __init__(self, question_id: str):
        super().__init__("waiting for user answer")
        self.question_id = question_id


class Cancelled(Exception):
    """The person asked this mission to stop."""


class BudgetExceeded(Exception):
    def __init__(self, which: str):
        super().__init__(f"budget exhausted: {which}")
        self.which = which


@dataclass(frozen=True)
class Tool:
    name: str
    fn: Callable[[dict, str], Any]     # fn(args, idempotency_key) -> JSON-serialisable result
    idempotent: bool = False
    doc: str = ""                      # one line for the planner: what the tool does and its exact argument names
    title: str = ""                    # short noun for people, e.g. "Forecast"; the UI never shows the tool's code name
    activity: str = ""                 # present-tense phrase for a running step, e.g. "Forecasting the series"
    observe: Callable[[Any], Any] | None = None   # compact view of a result for the planner; the ledger keeps the full one


class Ctx:
    def __init__(self, runtime: "Runtime", mission_id: str):
        self.rt = runtime
        self.store = runtime.store
        self.mission_id = mission_id
        self.step_idx = -1
        self.deadline = runtime.deadline

    def _guard_time(self) -> None:
        if self.deadline is not None and time.time() > self.deadline:
            raise BudgetExceeded("max_seconds")
        if self.store.is_cancel_requested(self.mission_id):
            raise Cancelled()
        # A long mission keeps its lease alive as it works, so another runtime cannot take it over mid-run.
        self.store.renew_lease(self.mission_id, self.rt.owner, self.rt.lease_s)

    def _guard_budget(self) -> None:
        limit = self.rt.budget.get("max_tool_calls")
        if limit is not None and self.store.count_tool_calls(self.mission_id) >= int(limit):
            raise BudgetExceeded("max_tool_calls")

    def step(self, name: str, fn: Callable[["Ctx"], Any]) -> Any:
        self.step_idx += 1
        idx = self.step_idx
        done = self.store.step_output(self.mission_id, idx)
        if done is not None:
            return done["v"]
        self._guard_time()
        out = fn(self)
        self.store.finish_step(self.mission_id, idx, name, {"v": out})
        self.store.append_event(self.mission_id, "step.done", {"idx": idx, "name": name})
        return out

    def call(self, tool: Tool, args: dict) -> Any:
        if self.step_idx < 0:
            raise RuntimeError("tool calls must happen inside ctx.step")
        key = idem_key(self.mission_id, self.step_idx, tool.name, args)
        rec = self.store.tool_get(key)
        if rec is not None and rec.status == "done":
            return rec.result
        if rec is not None and rec.status == "started" and not tool.idempotent:
            raise OutcomeUnknown(f"{tool.name} started in an earlier run and its outcome was not recorded")
        self._guard_budget()
        self._guard_time()
        self.store.tool_start(self.mission_id, self.step_idx, tool.name, args, key)
        self.store.append_event(self.mission_id, "tool.started", {"tool": tool.name, "key": key})
        try:
            result = tool.fn(args, key)
        except Exception as exc:
            msg = redact(str(exc))
            self.store.tool_set(key, "failed", error=msg)
            self.store.append_event(self.mission_id, "tool.failed", {"tool": tool.name, "error": msg})
            raise
        self.store.tool_set(key, "done", result)
        self.store.append_event(self.mission_id, "tool.done", {"tool": tool.name, "key": key})
        return result

    def ask(self, payload: dict) -> dict:
        """Ask the user. Returns the answers once they exist; otherwise records the question and pauses."""
        if self.step_idx < 0:
            raise RuntimeError("ctx.ask must happen inside ctx.step")
        key = idem_key(self.mission_id, self.step_idx, "ask_user", payload)
        rec = self.store.tool_get(key)
        if rec is not None and rec.status == "done":
            return rec.result["answers"]
        self.store.tool_start(self.mission_id, self.step_idx, "ask_user", payload, key)
        self.store.tool_set(key, "waiting")
        open_for_key = [q for q in self.store.open_questions(self.mission_id) if q["tool_key"] == key]
        if open_for_key:
            qid = open_for_key[0]["id"]
        else:
            qid = self.store.open_question(self.mission_id, key, payload)
            self.store.append_event(self.mission_id, "question.opened", {"question": qid})
        raise Paused(qid)


class Runtime:
    def __init__(self, store: Store, owner: str | None = None):
        self.store = store
        self.owner = owner or uuid.uuid4().hex
        self.deadline: float | None = None
        self.budget: dict = {}
        self.lease_s: float = 300.0

    def run(self, mission_id: str, program: Callable[[Ctx], Any], *, lease_s: float = 300.0) -> dict:
        mission = self.store.mission(mission_id)
        self.budget = dict(mission["budget"])
        max_s = self.budget.get("max_seconds")
        self.deadline = time.time() + float(max_s) if max_s else None
        self.lease_s = lease_s
        self.store.acquire_lease(mission_id, self.owner, lease_s)
        self.store.set_status(mission_id, "running")
        ctx = Ctx(self, mission_id)
        try:
            result = program(ctx)
        except Paused as p:
            self.store.set_status(mission_id, "waiting_user")
            self.store.release_lease(mission_id, self.owner)
            return {"status": "waiting_user", "question": p.question_id}
        except Cancelled:
            self.store.close_open_questions(mission_id)
            self.store.set_status(mission_id, "cancelled")
            self.store.release_lease(mission_id, self.owner)
            return {"status": "cancelled"}
        except OutcomeUnknown as exc:
            self.store.set_status(mission_id, "outcome_unknown")
            self.store.append_event(mission_id, "mission.outcome_unknown", {"reason": redact(str(exc))})
            self.store.release_lease(mission_id, self.owner)
            return {"status": "outcome_unknown", "reason": redact(str(exc))}
        except BudgetExceeded as exc:
            self.store.set_status(mission_id, "budget_exhausted")
            self.store.append_event(mission_id, "mission.budget_exhausted", {"which": exc.which})
            self.store.release_lease(mission_id, self.owner)
            return {"status": "budget_exhausted", "which": exc.which}
        except Exception as exc:
            msg = redact(f"{type(exc).__name__}: {exc}")
            self.store.set_status(mission_id, "failed")
            self.store.append_event(mission_id, "mission.failed", {"error": msg})
            self.store.release_lease(mission_id, self.owner)
            return {"status": "failed", "error": msg}
        self.store.set_answer(mission_id, result)
        # Report the goal's own outcome. A stalled or budget-capped run is not "completed" in the sense of "met".
        outcome = result.get("status") if isinstance(result, dict) else None
        mission_status = "completed" if outcome in (None, "met") else str(outcome)
        self.store.set_status(mission_id, mission_status)
        self.store.release_lease(mission_id, self.owner)
        return {"status": mission_status, "result": result}



__all__ = ["Runtime", "Ctx", "Tool", "Paused", "BudgetExceeded", "Cancelled", "LeaseHeld"]
