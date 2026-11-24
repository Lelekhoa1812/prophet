"""create_turn reads the shared SQLite connection; every read must hold store.lock.

Defect (round 8, 2026-11-24): the last-mission read in create_turn and the title read in _name_notebook ran without the
lock. When four workflows posted turns at once, a turn request failed with "sqlite3.InterfaceError: bad parameter or other
API misuse" (HTTP 500). This test is deterministic: it records whether the lock is held for every execute in create_turn.
"""
from __future__ import annotations

from prophet_agent.api import titles
from prophet_agent.api.service import MissionService
from prophet_agent.nim.errors import NotConfigured


def _no_planner(*args, **kwargs):
    raise NotConfigured("no planner for this test")


class _LockSpy:
    def __init__(self, conn, lock, seen):
        self._conn, self._lock, self._seen = conn, lock, seen

    def execute(self, sql, params=()):
        self._seen.append((" ".join(sql.split()[:3]), self._lock._is_owned()))
        return self._conn.execute(sql, params)

    def __getattr__(self, name):
        return getattr(self._conn, name)


def test_every_connection_read_in_create_turn_holds_the_store_lock(tmp_path, monkeypatch):
    # The title model would call the network from a background thread; the fallback title is what this test checks.
    monkeypatch.setattr(titles, "name_in_background", lambda *a, **k: None)
    svc = MissionService(tmp_path / "turn.sqlite", _no_planner, {})
    nb = svc.notebooks.create()
    seen = []
    svc.store.conn = _LockSpy(svc.store.conn, svc.store.lock, seen)
    svc.create_turn(nb["id"], "Forecast the next 30 days of the CO2 series")
    reads = [owned for sql, owned in seen if sql.upper().startswith("SELECT")]
    assert reads, "the spy recorded no reads, so the test proves nothing"
    assert all(reads), seen
