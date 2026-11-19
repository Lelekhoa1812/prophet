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


def _lesson_row():
    con = store.open_lab()
    r = con.execute("select lesson_id, active, effect from lessons").fetchall()
    con.close()
    return r


def test_agent_wide_lesson_waits_for_a_different_approver(lab):
    import json
    con = store.open_lab()
    con.execute("update proposals set dollars_at_stake=100 where proposal_id='PR-1'")
    for pid, ent, d in (("PR-2", "P-2", 80.0), ("PR-3", "P-3", 900.0)):
        con.execute("insert into proposals (proposal_id, dataset, agent, as_of, entity_id, dollars_at_stake, status, dedupe_key, created_at) values (?,?,?,?,?,?,?,?,?)",
                    [pid, "twin", "shortfall", "2026-01-06", ent, d, "open", f"shortfall|{ent}", datetime.now(timezone.utc)])
    con.close()
    r = c.post("/api/proposals/PR-1/decision", json={"decision": "dismiss", "reason": "not_useful", "scope": "agent", "decided_by": "ann"})
    assert r.json()["lesson"]["pending_approval"] is True
    lid = r.json()["lesson"]["lesson_id"]
    assert _lesson_row()[0][1] is False and main._governor("twin").lessons == []        # pending lessons do nothing
    assert c.post(f"/api/lessons/{lid}/approve", headers={"X-Operator": "ann"}).status_code == 403   # not the requester
    ok = c.post(f"/api/lessons/{lid}/approve", headers={"X-Operator": "bob"})
    assert ok.status_code == 200 and ok.json()["audit"]["violations"] == []
    ls = main._governor("twin").lessons
    assert len(ls) == 1 and ls[0].scope_type == "agent" and ls[0].effect["max_dollars"] == 100.0
    assert c.post(f"/api/lessons/{lid}/approve", headers={"X-Operator": "bob"}).status_code == 409   # already approved


def test_wide_lesson_never_holds_back_a_bigger_alert():
    from datetime import date
    from types import SimpleNamespace as NS
    from lab.policy.governor import Governor, Lesson, audit_lesson
    ls = Lesson("L", "shortfall", "agent", "*", {"suppress_days": 30, "max_dollars": 100.0}, date(2026, 1, 5))
    small = NS(proposal_id="a", agent="shortfall", entity_id="P-2", dedupe_key="shortfall|P-2", as_of=date(2026, 1, 8), dollars_at_stake=80.0)
    big = NS(proposal_id="b", agent="shortfall", entity_id="P-3", dedupe_key="shortfall|P-3", as_of=date(2026, 1, 8), dollars_at_stake=900.0)
    early = NS(proposal_id="c", agent="shortfall", entity_id="P-4", dedupe_key="shortfall|P-4", as_of=date(2026, 1, 1), dollars_at_stake=10.0)
    other = NS(proposal_id="d", agent="margin", entity_id="P-5", dedupe_key="margin|P-5", as_of=date(2026, 1, 8), dollars_at_stake=10.0)
    a = audit_lesson(ls, [small, big, early, other])
    assert a["violations"] == [] and a["would_suppress"] == 1
    g = Governor(lessons=[ls])
    assert g.filter([big], date(2026, 1, 8), commit=False) == [big]
