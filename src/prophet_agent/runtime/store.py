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

SCHEMA_VERSION = 2

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


# Schema v2 adds notebooks: persistent workspaces whose turns are missions. v1 databases are upgraded in place; their
# missions keep notebook_id NULL and run exactly as before.
_DDL_V2 = """
CREATE TABLE IF NOT EXISTS notebooks (
  id TEXT PRIMARY KEY,
  title TEXT NOT NULL,
  instructions TEXT NOT NULL DEFAULT '',
  archived INTEGER NOT NULL DEFAULT 0,
  created REAL NOT NULL,
  updated REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS sources (
  id TEXT PRIMARY KEY,
  notebook_id TEXT NOT NULL REFERENCES notebooks(id),
  dataset_id TEXT NOT NULL,
  name TEXT NOT NULL,
  sha256 TEXT NOT NULL,
  bytes INTEGER NOT NULL,
  status TEXT NOT NULL,
  card_json TEXT,
  error TEXT,
  included INTEGER NOT NULL DEFAULT 1,
  removed INTEGER NOT NULL DEFAULT 0,
  created REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS sources_by_notebook ON sources(notebook_id);
CREATE TABLE IF NOT EXISTS derived_tables (
  id TEXT PRIMARY KEY,
  notebook_id TEXT NOT NULL REFERENCES notebooks(id),
  name TEXT NOT NULL,
  description TEXT NOT NULL DEFAULT '',
  created REAL NOT NULL,
  UNIQUE(notebook_id, name)
);
CREATE TABLE IF NOT EXISTS derived_versions (
  id TEXT PRIMARY KEY,
  table_id TEXT NOT NULL REFERENCES derived_tables(id),
  version INTEGER NOT NULL,
  rows INTEGER NOT NULL,
  columns_json TEXT NOT NULL,
  sha256 TEXT NOT NULL,
  file_path TEXT NOT NULL,
  lineage_json TEXT NOT NULL,
  turn_id TEXT,
  created REAL NOT NULL,
  UNIQUE(table_id, version)
);
"""
_MISSION_COLUMNS_V2 = {
    "notebook_id": "TEXT",
    "seq": "INTEGER",
    "parent_turn_id": "TEXT",
    "cancel_requested": "INTEGER NOT NULL DEFAULT 0",
    "archived": "INTEGER NOT NULL DEFAULT 0",
}
ACTIVE = ("created", "running", "waiting_user")


def canonical(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)


def idem_key(mission_id: str, step_idx: int, tool: str, args: Any) -> str:
    raw = f"{mission_id}:{step_idx}:{tool}:{canonical(args)}"
    return hashlib.sha256(raw.encode()).hexdigest()


class LeaseHeld(RuntimeError):
    """Another runtime holds the mission lease and it has not expired."""


class LeaseLost(RuntimeError):
    """This runtime's lease on a mission expired and was taken over. It must stop without writing further results."""


class NotebookBusy(RuntimeError):
    """A turn was started in a notebook that already has an active turn."""


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
            found = int(self.conn.execute("PRAGMA user_version").fetchone()[0])
            if found > SCHEMA_VERSION:
                raise RuntimeError(f"database schema {found} is newer than this version of Prophet ({SCHEMA_VERSION})")
            self.conn.executescript(_DDL)
            have = {r["name"] for r in self.conn.execute("PRAGMA table_info(missions)")}
            for col, decl in _MISSION_COLUMNS_V2.items():
                if col not in have:
                    self.conn.execute(f"ALTER TABLE missions ADD COLUMN {col} {decl}")
            self.conn.executescript(_DDL_V2)
            self.conn.execute("CREATE INDEX IF NOT EXISTS missions_by_notebook ON missions(notebook_id, seq)")
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
                    if exc_type:
                        store.conn.execute("ROLLBACK")
                    else:
                        try:
                            store.conn.execute("COMMIT")
                        except BaseException:
                            # A failed COMMIT (disk full, I/O error, busy) leaves the transaction open; roll it back so
                            # the shared connection is not wedged for every later write.
                            if store.conn.in_transaction:
                                store.conn.execute("ROLLBACK")
                            raise
                finally:
                    store.lock.release()
                return False

        return _Tx()

    # -- missions and leases -----------------------------------------------------------------------------------

    def create_mission(self, goal: str, budget: dict, *, notebook_id: str | None = None,
                       parent_turn_id: str | None = None) -> str:
        """A mission is one turn. In a notebook it gets the next sequence number, and only one turn may be active."""
        mission_id = uuid.uuid4().hex
        now = time.time()
        with self._tx() as c:
            seq = None
            if notebook_id is not None:
                busy = c.execute("SELECT 1 FROM missions WHERE notebook_id=? AND status IN (?,?,?)",
                                 (notebook_id, *ACTIVE)).fetchone()
                if busy:
                    raise NotebookBusy("this notebook is still working on the previous message")
                seq = int(c.execute("SELECT COALESCE(MAX(seq),0)+1 FROM missions WHERE notebook_id=?",
                                    (notebook_id,)).fetchone()[0])
            c.execute("INSERT INTO missions (id, goal, status, budget_json, created, updated, notebook_id, seq, "
                      "parent_turn_id) VALUES (?,?,?,?,?,?,?,?,?)",
                      (mission_id, goal, "created", canonical(budget), now, now, notebook_id, seq, parent_turn_id))
        self.append_event(mission_id, "mission.created", {"goal": goal, "budget": budget, "notebook_id": notebook_id})
        return mission_id

    def request_cancel(self, mission_id: str) -> bool:
        """Ask a running mission to stop at its next step. False if it is not active."""
        with self._tx() as c:
            cur = c.execute("UPDATE missions SET cancel_requested=1, updated=? WHERE id=? AND status IN (?,?,?)",
                            (time.time(), mission_id, *ACTIVE))
            return cur.rowcount == 1

    def is_cancel_requested(self, mission_id: str) -> bool:
        with self.lock:
            row = self.conn.execute("SELECT cancel_requested FROM missions WHERE id=?", (mission_id,)).fetchone()
        return bool(row and row["cancel_requested"])

    def close_open_questions(self, mission_id: str) -> None:
        with self._tx() as c:
            c.execute("UPDATE questions SET status='cancelled' WHERE mission_id=? AND status='open'", (mission_id,))

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
        """Extend the lease. Raises LeaseLost when this runtime no longer holds it (it expired and another runtime
        took the mission), so the old runtime stops instead of running on beside the new one."""
        with self._tx() as c:
            cur = c.execute("UPDATE missions SET lease_until=? WHERE id=? AND lease_owner=?",
                            (time.time() + ttl_s, mission_id, owner))
            if cur.rowcount != 1:
                raise LeaseLost(f"mission {mission_id} lease was taken over by another runtime")

    def interrupt_orphans(self) -> list[str]:
        """At startup: a mission left as created or running by a process that died, whose lease has lapsed, is
        marked interrupted. That frees its notebook (active turns block new ones) and lets it be started again;
        a non-idempotent call left started is still stopped as outcome_unknown when the mission resumes."""
        now = time.time()
        with self._tx() as c:
            ids = [r["id"] for r in c.execute(
                "SELECT id FROM missions WHERE status IN ('created','running') "
                "AND (lease_owner IS NULL OR lease_until<?)", (now,)).fetchall()]
            for mid in ids:
                c.execute("UPDATE missions SET status='interrupted', lease_owner=NULL, lease_until=0, updated=? "
                          "WHERE id=?", (now, mid))
        for mid in ids:
            self.append_event(mid, "mission.status", {"status": "interrupted",
                                                      "reason": "the runtime stopped before this mission finished"})
        return ids

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

    def steps(self, mission_id: str) -> list[dict]:
        with self.lock:
            rows = self.conn.execute("SELECT idx, name, status, output_json FROM steps WHERE mission_id=? "
                                     "ORDER BY idx", (mission_id,)).fetchall()
        return [{"idx": r["idx"], "name": r["name"], "status": r["status"],
                 "output": (json.loads(r["output_json"]) or {}).get("v") if r["output_json"] else None}
                for r in rows]

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
