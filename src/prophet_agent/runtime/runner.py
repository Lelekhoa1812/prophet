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

import re
import time
import uuid
from dataclasses import dataclass
from typing import Any, Callable

from prophet_agent.nim.errors import redact

from .store import LeaseBound, LeaseHeld, LeaseLost, OutcomeUnknown, Store, idem_key


class Paused(Exception):
    """The program is waiting for the user to answer a question."""

    def __init__(self, question_id: str):
        super().__init__("waiting for user answer")
        self.question_id = question_id


# Calls that do no work on the data are not charged to a call budget: reads of playbooks and notes, and a read of a stored
# observation (round 3, A5-02). ``ask_user`` is not a tool call either: the question is the cost, the answer is free.
READ_ONLY = ("analyst_skill", "notes", "read_observation")
UNCHARGED = READ_ONLY + ("ask_user",)
# Mission-wide cap on calls made by delegated sub-agents (round 3, A5-01): a safety stop, not the lead's allowance. It is
# the largest type's step budget (14) times the mission's sub-agent limit (24), so it is reached only by runaway delegation.
DELEGATED_CALLS = 14 * 24


class Cancelled(Exception):
    """The person asked this mission to stop."""


class ArgumentRejected(Exception):
    """A tool refused its arguments before doing any work (round 3, A5-12). The call is recorded as "rejected": it is not
    charged to a budget, and the same refusal is returned on a replay without running the tool again."""


# The refusals the tools raise today, matched by their wording: "unknown arguments ...; allowed: ...", "x must be one of
# [...]". Tools should raise ArgumentRejected directly; until they do, the wording is the signal.
_ARG_REFUSAL = re.compile(r"unknown arguments|must be one of|allowed: |use these names|is required|missing required", re.I)


def is_argument_refusal(exc: BaseException) -> bool:
    return isinstance(exc, (ValueError, TypeError, ArgumentRejected)) and bool(_ARG_REFUSAL.search(str(exc)))


class BudgetExceeded(Exception):
    def __init__(self, which: str, detail: str | None = None):
        super().__init__(f"budget exhausted: {which}")
        self.which = which
        self.detail = detail      # what stopped the part, in words (a provider error, already redacted)


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
        # "lead" for the lead agent's calls, "delegated" while a sub-agent runs on this context (round 3, A5-01). The lane
        # is set by the program, so a replay charges the same calls to the same allowance.
        self.lane = "lead"

    def branch(self, base: int) -> "Ctx":
        """A context for work that runs beside this one (a parallel sub-agent). It shares the mission, store and runtime but
        numbers its steps from ``base``, a block of numbers no other branch or the parent uses, so concurrent work keeps a
        replay-stable order: the same program asks for the same step numbers on every run."""
        child = Ctx(self.rt, self.mission_id)
        child.step_idx = base - 1
        return child

    def _guard_time(self) -> None:
        if self.deadline is not None and time.time() > self.deadline:
            raise BudgetExceeded("max_seconds")
        if self.store.is_cancel_requested(self.mission_id):
            raise Cancelled()
        # A long mission keeps its lease alive as it works, so another runtime cannot take it over mid-run.
        self.store.renew_lease(self.mission_id, self.rt.owner, self.rt.lease_s)

    def seconds_left(self) -> float | None:
        """Seconds left in this segment's time allowance, or None when the mission has none (round 4, A5-06)."""
        return None if self.deadline is None else self.deadline - time.time()

    def _guard_budget(self) -> None:
        """The lead's calls are held to ``max_tool_calls``; a sub-agent's calls to ``max_delegated_calls``. Each lane is
        counted on its own, so a delegation cannot use up the lead's allowance (live Beijing: 44 delegated calls, lead
        forced to finish at 45 of 20)."""
        if self.lane == "delegated":
            cap = self.rt.budget.get("max_delegated_calls")
            if cap is not None and self.store.count_tool_calls(self.mission_id, exclude=UNCHARGED, lane="delegated") >= int(cap):
                raise BudgetExceeded("max_delegated_calls")
            return
        limit = self.rt.budget.get("max_tool_calls")
        if limit is not None and self.store.count_tool_calls(self.mission_id, exclude=UNCHARGED, lane="lead") >= int(limit):
            raise BudgetExceeded("max_tool_calls")

    def step(self, name: str, fn: Callable[["Ctx"], Any]) -> Any:
        self.step_idx += 1
        idx = self.step_idx
        rec = self.store.step_record(self.mission_id, idx)
        if rec is not None and rec["status"] == "done":
            # A replay must ask for the same step at the same position. A different name here means the program took
            # another path than the one stored, and returning the stored value would answer a different question.
            if rec["name"] != name:
                raise RuntimeError(f"replay diverged at step {idx}: the ledger has {rec['name']!r}, the program asked "
                                   f"for {name!r}")
            return rec["v"]
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
        if rec is not None and rec.status == "rejected":
            raise ArgumentRejected(rec.error or "the arguments were refused")
        if rec is not None and rec.status == "started" and not tool.idempotent:
            raise OutcomeUnknown(f"{tool.name} started in an earlier run and its outcome was not recorded")
        if rec is not None and rec.status == "failed" and not tool.idempotent:
            # The tool already ran and failed; it may have changed something before failing. Not re-run on resume.
            raise OutcomeUnknown(f"{tool.name} failed in an earlier run and is not idempotent, so it is not retried: "
                                 f"{rec.error or 'no detail'}")
        self._guard_budget()
        self._guard_time()
        self.store.tool_start(self.mission_id, self.step_idx, tool.name, args, key, lane=self.lane)
        self.store.append_event(self.mission_id, "tool.started", {"tool": tool.name, "key": key})
        try:
            result = tool.fn(args, key)
        except Exception as exc:
            msg = redact(str(exc))
            if is_argument_refusal(exc):
                self.store.tool_set(key, "rejected", error=msg)
                self.store.append_event(self.mission_id, "tool.rejected", {"tool": tool.name, "error": msg})
                raise ArgumentRejected(msg) from None
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
        self.owner = owner or uuid.uuid4().hex
        # Every write this runtime makes is made as this owner, so the lease fence can tell it from another runtime.
        base = store._store if isinstance(store, LeaseBound) else store
        self.store = LeaseBound(base, self.owner)
        self.deadline: float | None = None
        self.budget: dict = {}
        self.lease_s: float = 300.0

    def run(self, mission_id: str, program: Callable[[Ctx], Any], *, lease_s: float = 300.0) -> dict:
        # A takeover can be detected by any write, including the ones after the program (the answer, the final status) and
        # the handlers below. Each of those would escape run() unless it is caught here, so a lost lease always ends as
        # lease_lost: this runtime writes nothing more and the new owner decides (store._fence).
        try:
            return self._run(mission_id, program, lease_s=lease_s)
        except LeaseLost:
            return {"status": "lease_lost"}

    def _run(self, mission_id: str, program: Callable[[Ctx], Any], *, lease_s: float = 300.0) -> dict:
        mission = self.store.mission(mission_id)
        self.budget = dict(mission["budget"])
        self.budget.setdefault("max_delegated_calls", DELEGATED_CALLS)
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
        except LeaseLost:
            # Another runtime took the mission over. This one stops and writes nothing more: the new owner decides.
            return {"status": "lease_lost"}
        except OutcomeUnknown as exc:
            self.store.set_status(mission_id, "outcome_unknown")
            self.store.append_event(mission_id, "mission.outcome_unknown", {"reason": redact(str(exc))})
            self.store.release_lease(mission_id, self.owner)
            return {"status": "outcome_unknown", "reason": redact(str(exc))}
        except BudgetExceeded as exc:
            # The segment's allowance of time or tool calls is spent. That is a checkpoint, never a failure: the stored
            # steps stay, and the person continues with a fresh allowance (see MissionService.continue_mission).
            self.store.set_status(mission_id, "checkpoint")
            self.store.append_event(mission_id, "mission.checkpoint",
                                    {"which": exc.which, **({"detail": exc.detail} if exc.detail else {})})
            self.store.release_lease(mission_id, self.owner)
            return {"status": "checkpoint", "which": exc.which,
                    "reason": f"this part of the work reached its {exc.which} allowance"}
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



__all__ = ["Runtime", "Ctx", "Tool", "Paused", "BudgetExceeded", "Cancelled", "ArgumentRejected", "LeaseHeld"]
