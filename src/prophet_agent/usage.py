"""Token usage ledger: one row per successful model call, rolling windows per model, and the budgets Settings holds.

Motivation vs Logic
Motivation: a person paying for several model providers wants to see how many tokens each model has used in the last
day, week and month, and to set a ceiling per model. Every mission and sub-agent call goes through the shared client, so
one hook there covers all of them.
Logic: a small SQLite table in WAL mode (data/usage.sqlite, kept by "clear data" because it is not mission data). Used
tokens for a model are prompt_tokens + completion_tokens as the vendor reported them; cache and reasoning counts are
already inside those two totals, so they are not added again. Windows are rolling: an event counts in a window while
``now - event < window``. The reset time for a window is the moment its oldest counted event leaves it; with no events,
it is now plus the window. The clock is injected so the window maths can be tested without waiting.
"""
from __future__ import annotations

import sqlite3
import threading
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .nim.providers import ModelInfo

DAY = 86_400
WINDOWS: dict[str, int] = {"1d": DAY, "7d": 7 * DAY, "30d": 30 * DAY}
USAGE_FILE = "usage.sqlite"


def iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _count(value: Any) -> int:
    """A reported token count, or 0 when it is missing or not a non-negative whole number."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return 0
    return value


class UsageLedger:
    def __init__(self, path: Path, clock: Callable[[], float] = time.time):
        self.path = path
        self.clock = clock
        self._lock = threading.Lock()
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._db() as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.execute("CREATE TABLE IF NOT EXISTS calls ("
                       "id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL NOT NULL, at TEXT NOT NULL, "
                       "provider TEXT NOT NULL, model TEXT NOT NULL, "
                       "prompt_tokens INTEGER NOT NULL, completion_tokens INTEGER NOT NULL)")
            db.execute("CREATE INDEX IF NOT EXISTS calls_model_ts ON calls(model, ts)")

    @contextmanager
    def _db(self) -> Iterator[sqlite3.Connection]:
        """One short connection per operation, serialised by the lock. Thread-safe for the process."""
        with self._lock:
            conn = sqlite3.connect(self.path, timeout=10)
            try:
                with conn:
                    yield conn
            finally:
                conn.close()

    def record(self, provider: str, model: str, usage: Mapping[str, Any] | None, *, at: float | None = None) -> None:
        ts = self.clock() if at is None else at
        usage = usage or {}
        with self._db() as db:
            db.execute("INSERT INTO calls (ts, at, provider, model, prompt_tokens, completion_tokens) "
                       "VALUES (?, ?, ?, ?, ?, ?)",
                       (ts, iso(ts), provider, model, _count(usage.get("prompt_tokens")),
                        _count(usage.get("completion_tokens"))))

    def used(self, model: str, window: str, now: float | None = None) -> int:
        now = self.clock() if now is None else now
        with self._db() as db:
            row = db.execute("SELECT COALESCE(SUM(prompt_tokens + completion_tokens), 0) FROM calls "
                             "WHERE model = ? AND ts > ?", (model, now - WINDOWS[window])).fetchone()
        return int(row[0])

    def reset_at(self, model: str, window: str, now: float | None = None) -> float:
        """Epoch seconds when the oldest counted event leaves the window, or now + window when nothing is counted."""
        now = self.clock() if now is None else now
        with self._db() as db:
            row = db.execute("SELECT MIN(ts) FROM calls WHERE model = ? AND ts > ?",
                             (model, now - WINDOWS[window])).fetchone()
        oldest = row[0]
        return (oldest if oldest is not None else now) + WINDOWS[window]

    def summary(self, models: Sequence[ModelInfo], budgets: Mapping[str, Mapping[str, int | None]],
                now: float | None = None) -> list[dict]:
        now = self.clock() if now is None else now
        out = []
        for m in models:
            budget = budgets.get(m.id, {})
            out.append({
                "id": m.id, "provider": m.provider, "label": m.label,
                "used": {w: self.used(m.id, w, now) for w in WINDOWS},
                "budget": {w: budget.get(w) for w in WINDOWS},
                "resets": {w: iso(self.reset_at(m.id, w, now)) for w in WINDOWS},
            })
        return out


_LEDGERS: dict[str, UsageLedger] = {}
_LEDGERS_LOCK = threading.Lock()


def ledger_for(data_root: Path) -> UsageLedger:
    """One ledger per data folder for this process, so the transport hook and the API share the same object."""
    key = str((data_root / USAGE_FILE).resolve())
    with _LEDGERS_LOCK:
        if key not in _LEDGERS:
            _LEDGERS[key] = UsageLedger(Path(key))
        return _LEDGERS[key]


__all__ = ["DAY", "UsageLedger", "WINDOWS", "iso", "ledger_for"]
