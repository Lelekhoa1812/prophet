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
    monkeypatch.setattr(main, "_available", lambda: ["twin"])      # no dataset files needed (CI has none)
    monkeypatch.setattr(main, "_scorecard", lambda ds: {})
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


def who():
    con = store.open_lab()
    r = con.execute("select decided_by from decisions order by decided_at desc").fetchone()[0]
    con.close()
    return r


def test_operator_header_names_who_decided_and_beats_the_body(lab):
    c.post("/api/proposals/PR-1/decision", json={"decision": "approve", "decided_by": "reviewer"}, headers={"X-Operator": "Sam Nguyen"})
    assert who() == "unverified:Sam Nguyen"


def test_operator_name_is_validated(lab):
    for bad in ("x" * 100, "name;drop", "<script>"):
        r = c.post("/api/proposals/PR-1/decision", json={"decision": "approve"}, headers={"X-Operator": bad})
        assert r.status_code == 400
    r = c.post("/api/proposals/PR-1/decision", json={"decision": "approve", "decided_by": "<b>x</b>"})
    assert r.status_code == 400


def test_default_identity_is_labelled_unverified(lab):
    c.post("/api/proposals/PR-1/decision", json={"decision": "approve"})
    assert who().startswith("unverified:")


def test_dismissal_lesson_runs_on_the_data_clock(lab):
    # Motivation vs Logic: a dismissal on a proposal dated 2026-01-05 must suppress from that day, not from today's wall clock.
    from datetime import date
    c.post("/api/proposals/PR-1/decision", json={"decision": "dismiss", "reason": "one_off"})
    ls = main._governor("twin").lessons
    assert len(ls) == 1 and ls[0].created_on == date(2026, 1, 5)
