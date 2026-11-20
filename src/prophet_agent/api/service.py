"""Mission service: starts and resumes missions on background threads and exposes their state to the HTTP layer.

Motivation vs Logic
Motivation: a mission can run for minutes. The HTTP request that starts it must return at once, and an answer to a
question must resume the same mission without the browser holding it open.
Logic: each start or resume runs ``Runtime.run`` on a daemon thread. The durable store is the source of truth; the
service keeps no mission state of its own. A second start on a mission that is already running hits the lease and
returns a conflict rather than running twice.
"""
from __future__ import annotations

import threading
from pathlib import Path
from typing import Any, Callable

from prophet_agent.runtime.goal import GoalConfig, goal_program
from prophet_agent.runtime.runner import Runtime, Tool
from prophet_agent.runtime.store import LeaseHeld, Store

BrainFactory = Callable[[], Any]


class MissionService:
    def __init__(self, db_path: str | Path, brain_factory: BrainFactory, tools: dict[str, Tool],
                 goal_cfg: GoalConfig | None = None):
        self.store = Store(db_path)
        self.brain_factory = brain_factory
        self.tools = tools
        self.goal_cfg = goal_cfg or GoalConfig()
        self._threads: dict[str, threading.Thread] = {}
        self._lock = threading.Lock()

    def create(self, goal: str, budget: dict | None = None) -> str:
        return self.store.create_mission(goal, budget or {"max_tool_calls": 12, "max_seconds": 600})

    def start(self, mission_id: str) -> None:
        """Run (or resume) the mission on a background thread. Raises LeaseHeld if it is already running."""
        mission = self.store.mission(mission_id)
        with self._lock:
            live = self._threads.get(mission_id)
            if live is not None and live.is_alive():
                raise LeaseHeld(f"mission {mission_id} is already running")
            program = goal_program(mission["goal"], self.tools, self.brain_factory(), self.goal_cfg)
            runtime = Runtime(self.store, owner=f"api-{mission_id[:8]}")
            t = threading.Thread(target=self._run, args=(runtime, mission_id, program), daemon=True)
            self._threads[mission_id] = t
            t.start()

    def _run(self, runtime: Runtime, mission_id: str, program: Callable) -> None:
        try:
            runtime.run(mission_id, program)
        except LeaseHeld:
            return
        except Exception as exc:  # the runtime records its own failures; this guards against bugs in the service
            self.store.append_event(mission_id, "service.error", {"error": type(exc).__name__})

    def answer(self, question_id: str, answers: dict) -> str:
        """Record the answer and resume the mission it belongs to. Returns the mission id."""
        info = self.store.answer_question(question_id, answers)
        self.start(info["mission_id"])
        return info["mission_id"]

    def wait(self, mission_id: str, timeout: float | None = None) -> None:
        t = self._threads.get(mission_id)
        if t is not None:
            t.join(timeout)

    def snapshot(self, mission_id: str, after_seq: int = 0) -> dict:
        mission = self.store.mission(mission_id)
        return {
            "id": mission_id,
            "goal": mission["goal"],
            "status": mission["status"],
            "budget": mission["budget"],
            "answer": mission["answer"],
            "events": self.store.events(mission_id, after_seq),
            "steps": self.store.steps(mission_id),
            "questions": self.store.open_questions(mission_id),
        }

    def list(self, limit: int = 20) -> list[dict]:
        with self.store.lock:
            rows = self.store.conn.execute(
                "SELECT id, goal, status, created, updated FROM missions ORDER BY created DESC LIMIT ?",
                (int(limit),)).fetchall()
        return [dict(r) for r in rows]
