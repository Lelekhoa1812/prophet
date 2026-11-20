"""Durable state for missions: SQLite in WAL mode with an event log, a step ledger, a tool-call ledger and questions.

Motivation vs Logic
Motivation: a mission can run for minutes, call paid models and tools, and be interrupted by a crash, a closed laptop
or a stopped process. Restarting must never repeat a side effect that already happened, and must never ask the model
to re-decide a step whose decision was recorded.
Logic: every decision and every tool call is written before the runtime moves on. Steps have a deterministic index,
so replay finds the recorded output by (mission, index). Tool calls are keyed by a hash of (mission, step, tool,
canonical arguments); a call already marked done returns its stored result and is never executed again. A call
left in "started" by a crash is re-run only if the tool declares itself idempotent; otherwise the runtime stops and
reports the outcome as unknown, rather than guessing. One runtime holds the mission lease at a time.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1

_DDL = """
CREATE TABLE IF NOT EXISTS missions (
  id TEXT PRIMARY KEY,
  goal TEXT NOT NULL,
  status TEXT NOT NULL,
  budget_json TEXT NOT NULL,
  created REAL NOT NULL,
  updated REAL NOT NULL,
  lease_owner TEXT,
  lease_until REAL NOT NULL DEFAULT 0,
  answer_json TEXT
);
CREATE TABLE IF NOT EXISTS events (
  seq INTEGER PRIMARY KEY AUTOINCREMENT,
  mission_id TEXT NOT NULL REFERENCES missions(id),
  kind TEXT NOT NULL,
  payload_json TEXT NOT NULL,
  ts REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS events_by_mission ON events(mission_id, seq);
CREATE TABLE IF NOT EXISTS steps (
  mission_id TEXT NOT NULL REFERENCES missions(id),
  idx INTEGER NOT NULL,
  name TEXT NOT NULL,
  status TEXT NOT NULL,
  output_json TEXT,
  attempts INTEGER NOT NULL DEFAULT 0,
  updated REAL NOT NULL,
  PRIMARY KEY (mission_id, idx)
);
CREATE TABLE IF NOT EXISTS tool_calls (
  idem_key TEXT PRIMARY KEY,
  mission_id TEXT NOT NULL REFERENCES missions(id),
  step_idx INTEGER NOT NULL,
  tool TEXT NOT NULL,
  args_json TEXT NOT NULL,
  status TEXT NOT NULL,
  result_json TEXT,
  error TEXT,
  updated REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS questions (
  id TEXT PRIMARY KEY,
  mission_id TEXT NOT NULL REFERENCES missions(id),
  tool_key TEXT NOT NULL REFERENCES tool_calls(idem_key),
  payload_json TEXT NOT NULL,
  status TEXT NOT NULL,
  answers_json TEXT,
  created REAL NOT NULL,
  answered REAL
);
"""


def canonical(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)


def idem_key(mission_id: str, step_idx: int, tool: str, args: Any) -> str:
    raw = f"{mission_id}:{step_idx}:{tool}:{canonical(args)}"
    return hashlib.sha256(raw.encode()).hexdigest()


class LeaseHeld(RuntimeError):
    """Another runtime holds the mission lease and it has not expired."""


class OutcomeUnknown(RuntimeError):
    """A non-idempotent tool started, the process died, and its outcome is not recorded."""


@dataclass(frozen=True)
class ToolRecord:
    idem_key: str
    status: str                 # started | waiting | done | failed
    result: Any
    error: str | None


class Store:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.path, isolation_level=None, check_same_thread=False, timeout=10.0)
        self.conn.row_factory = sqlite3.Row
        self.lock = threading.RLock()
        with self.lock:
            self.conn.execute("PRAGMA journal_mode=WAL")
            self.conn.execute("PRAGMA synchronous=NORMAL")
            self.conn.execute("PRAGMA foreign_keys=ON")
            self.conn.executescript(_DDL)
            self.conn.execute(f"PRAGMA user_version={SCHEMA_VERSION}")

    def close(self) -> None:
        with self.lock:
            self.conn.close()

    def _tx(self):
        store = self

        class _Tx:
            def __enter__(self):
                store.lock.acquire()
                try:
                    store.conn.execute("BEGIN IMMEDIATE")
                except BaseException:
                    store.lock.release()
                    raise
                return store.conn

            def __exit__(self, exc_type, exc, tb):
                try:
                    store.conn.execute("ROLLBACK" if exc_type else "COMMIT")
                finally:
                    store.lock.release()
                return False

        return _Tx()

    # -- missions and leases -----------------------------------------------------------------------------------

    def create_mission(self, goal: str, budget: dict) -> str:
        mission_id = uuid.uuid4().hex
        now = time.time()
        with self._tx() as c:
            c.execute("INSERT INTO missions (id, goal, status, budget_json, created, updated) VALUES (?,?,?,?,?,?)",
                      (mission_id, goal, "created", canonical(budget), now, now))
        self.append_event(mission_id, "mission.created", {"goal": goal, "budget": budget})
        return mission_id

    def mission(self, mission_id: str) -> dict:
        with self.lock:
            row = self.conn.execute("SELECT * FROM missions WHERE id=?", (mission_id,)).fetchone()
        if row is None:
            raise KeyError(f"no mission {mission_id}")
        out = dict(row)
        out["budget"] = json.loads(out.pop("budget_json"))
        out["answer"] = json.loads(out.pop("answer_json") or "null")
        return out

    def set_status(self, mission_id: str, status: str) -> None:
        with self._tx() as c:
            c.execute("UPDATE missions SET status=?, updated=? WHERE id=?", (status, time.time(), mission_id))
        self.append_event(mission_id, "mission.status", {"status": status})

    def set_answer(self, mission_id: str, answer: Any) -> None:
        with self._tx() as c:
            c.execute("UPDATE missions SET answer_json=?, updated=? WHERE id=?",
                      (canonical(answer), time.time(), mission_id))

    def acquire_lease(self, mission_id: str, owner: str, ttl_s: float) -> None:
        now = time.time()
        with self._tx() as c:
            cur = c.execute(
                "UPDATE missions SET lease_owner=?, lease_until=? WHERE id=? "
                "AND (lease_owner IS NULL OR lease_owner=? OR lease_until<?)",
                (owner, now + ttl_s, mission_id, owner, now))
            if cur.rowcount != 1:
                raise LeaseHeld(f"mission {mission_id} is leased by another runtime")

    def renew_lease(self, mission_id: str, owner: str, ttl_s: float) -> None:
        with self._tx() as c:
            c.execute("UPDATE missions SET lease_until=? WHERE id=? AND lease_owner=?",
                      (time.time() + ttl_s, mission_id, owner))

    def release_lease(self, mission_id: str, owner: str) -> None:
        with self._tx() as c:
            c.execute("UPDATE missions SET lease_owner=NULL, lease_until=0 WHERE id=? AND lease_owner=?",
                      (mission_id, owner))

    # -- event log ---------------------------------------------------------------------------------------------

    def append_event(self, mission_id: str, kind: str, payload: dict) -> int:
        with self._tx() as c:
            cur = c.execute("INSERT INTO events (mission_id, kind, payload_json, ts) VALUES (?,?,?,?)",
                            (mission_id, kind, canonical(payload), time.time()))
            return int(cur.lastrowid)

    def events(self, mission_id: str, after_seq: int = 0) -> list[dict]:
        with self.lock:
            rows = self.conn.execute("SELECT seq, kind, payload_json, ts FROM events "
                                     "WHERE mission_id=? AND seq>? ORDER BY seq", (mission_id, after_seq)).fetchall()
        return [{"seq": r["seq"], "kind": r["kind"], "payload": json.loads(r["payload_json"]), "ts": r["ts"]}
                for r in rows]

    # -- steps -------------------------------------------------------------------------------------------------

    def step_output(self, mission_id: str, idx: int) -> Any | None:
        """Return the stored output of a finished step, or None if it has not finished."""
        with self.lock:
            row = self.conn.execute("SELECT status, output_json FROM steps WHERE mission_id=? AND idx=?",
                                    (mission_id, idx)).fetchone()
        if row is None or row["status"] != "done":
            return None
        return json.loads(row["output_json"])

    def finish_step(self, mission_id: str, idx: int, name: str, output: Any) -> None:
        with self._tx() as c:
            c.execute("INSERT INTO steps (mission_id, idx, name, status, output_json, attempts, updated) "
                      "VALUES (?,?,?,?,?,?,?) ON CONFLICT(mission_id, idx) DO UPDATE SET "
                      "status='done', output_json=excluded.output_json, updated=excluded.updated",
                      (mission_id, idx, name, "done", canonical(output), 1, time.time()))

    # -- tool calls --------------------------------------------------------------------------------------------

    def tool_get(self, key: str) -> ToolRecord | None:
        with self.lock:
            row = self.conn.execute("SELECT idem_key, status, result_json, error FROM tool_calls WHERE idem_key=?",
                                    (key,)).fetchone()
        if row is None:
            return None
        return ToolRecord(row["idem_key"], row["status"], json.loads(row["result_json"] or "null"), row["error"])

    def tool_start(self, mission_id: str, step_idx: int, tool: str, args: Any, key: str) -> None:
        """Record that a call is starting. Never downgrades a call that already has a result (DO NOTHING on conflict)."""
        with self._tx() as c:
            c.execute("INSERT INTO tool_calls (idem_key, mission_id, step_idx, tool, args_json, status, updated) "
                      "VALUES (?,?,?,?,?,?,?) ON CONFLICT(idem_key) DO NOTHING",
                      (key, mission_id, step_idx, tool, canonical(args), "started", time.time()))

    def tool_set(self, key: str, status: str, result: Any = None, error: str | None = None) -> None:
        with self._tx() as c:
            c.execute("UPDATE tool_calls SET status=?, result_json=?, error=?, updated=? WHERE idem_key=?",
                      (status, canonical(result) if result is not None else None, error, time.time(), key))

    def count_tool_calls(self, mission_id: str) -> int:
        with self.lock:
            row = self.conn.execute("SELECT COUNT(*) AS n FROM tool_calls WHERE mission_id=?", (mission_id,)).fetchone()
        return int(row["n"])

    def tool_keys_waiting(self, mission_id: str) -> list[str]:
        with self.lock:
            rows = self.conn.execute("SELECT idem_key FROM tool_calls WHERE mission_id=? AND status='waiting'",
                                     (mission_id,)).fetchall()
        return [r["idem_key"] for r in rows]

    # -- questions ---------------------------------------------------------------------------------------------

    def open_question(self, mission_id: str, tool_key: str, payload: dict) -> str:
        qid = uuid.uuid4().hex
        with self._tx() as c:
            c.execute("INSERT INTO questions (id, mission_id, tool_key, payload_json, status, created) "
                      "VALUES (?,?,?,?,?,?)", (qid, mission_id, tool_key, canonical(payload), "open", time.time()))
        return qid

    def open_questions(self, mission_id: str) -> list[dict]:
        with self.lock:
            rows = self.conn.execute("SELECT id, tool_key, payload_json FROM questions "
                                     "WHERE mission_id=? AND status='open' ORDER BY created", (mission_id,)).fetchall()
        return [{"id": r["id"], "tool_key": r["tool_key"], "payload": json.loads(r["payload_json"])} for r in rows]

    def answer_question(self, question_id: str, answers: dict) -> dict:
        """Record the answer and resolve the waiting tool call, so the runtime can resume without asking again."""
        with self._tx() as c:
            row = c.execute("SELECT mission_id, tool_key, status FROM questions WHERE id=?",
                            (question_id,)).fetchone()
            if row is None:
                raise KeyError(f"no question {question_id}")
            if row["status"] != "open":
                raise ValueError("question already answered")
            c.execute("UPDATE questions SET status='answered', answers_json=?, answered=? WHERE id=?",
                      (canonical(answers), time.time(), question_id))
            c.execute("UPDATE tool_calls SET status='done', result_json=?, updated=? WHERE idem_key=?",
                      (canonical({"answers": answers}), time.time(), row["tool_key"]))
        self.append_event(row["mission_id"], "question.answered", {"question": question_id})
        return {"mission_id": row["mission_id"]}
