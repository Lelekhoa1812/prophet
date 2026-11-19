"""A reviewer changing their mind must not leave stale suppressions or double-count in the agent stats."""
import json
from datetime import datetime, timezone

import pytest
from fastapi.testclient import TestClient

from lab import store
from lab.api import main

c = TestClient(main.app, base_url="http://localhost:8000")


@pytest.fixture
def lab(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "LAB_DB", tmp_path / "lab.duckdb")
    con = store.open_lab()
    con.execute("insert into proposals (proposal_id, dataset, agent, as_of, entity_id, status, dedupe_key, created_at) values ('PR-1','twin','shortfall','2026-01-05','P-1','open','k',?)", [datetime.now(timezone.utc)])
    con.close()
    main._governors.clear()


def lessons():
    con = store.open_lab()
    r = con.execute("select active from lessons").fetchall()
    con.close()
    return [x[0] for x in r]


def test_changing_mind_retires_the_old_lesson(lab):
    c.post("/api/proposals/PR-1/decision", json={"decision": "dismiss", "reason": "wrong_alert"})
    assert lessons() == [True]
    c.post("/api/proposals/PR-1/decision", json={"decision": "approve"})
    assert lessons() == [False]


def test_repeat_dismissal_leaves_one_active_lesson(lab):
    for _ in range(3):
        c.post("/api/proposals/PR-1/decision", json={"decision": "dismiss", "reason": "wrong_alert"})
    assert lessons().count(True) == 1


def test_agent_stats_count_latest_decision_once(lab):
    for d in ("dismiss", "approve", "approve"):
        c.post("/api/proposals/PR-1/decision", json={"decision": d, "reason": "wrong_alert" if d == "dismiss" else None})
    sf = [a for a in c.get("/api/agents", params={"dataset": "twin"}).json() if a["key"] == "shortfall"][0]
    assert sf["operator_decisions"] == {"approve": 1}
