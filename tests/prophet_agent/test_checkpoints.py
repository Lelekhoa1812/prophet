"""Fork and Rewind at a user message (docs/prophet/fork_rewind_copy.md).

The promises under test are the runtime's: a finished tool call is never run again, a call a crash left "started" is never
retried, model decisions and results are replayed from the ledger, and a rewind never loses what it hides. No network, no
model, no key.
"""
from __future__ import annotations

import json
import time

import pytest

from prophet_agent import cleanup, notebooks as notebooks_module
from prophet_agent.api import titles
from prophet_agent.api.service import MissionService
from prophet_agent.checkpoints import Busy, Refused
from prophet_agent.runtime.runner import Runtime, Tool
from prophet_agent.runtime.store import idem_key
from prophet_agent.tools import default_tools

RUN = "ab" * 16


@pytest.fixture(autouse=True)
def no_title_model(monkeypatch):
    monkeypatch.setattr(titles, "name_in_background", lambda *a, **k: None)


@pytest.fixture()
def svc(tmp_path):
    s = MissionService(tmp_path / "ck.sqlite", lambda: None, default_tools())
    yield s
    s.close()


class Counts:
    def __init__(self):
        self.decide = 0
        self.tool = 0


def _program(n: int, counts: Counts, *, idempotent=True):
    """Two steps: a stored decision, then a tool call that writes a result naming a run folder."""
    def tool_fn(args, key):
        counts.tool += 1
        return {"value": args["n"], "run_id": RUN}

    tool = Tool("work", tool_fn, idempotent=idempotent)

    def program(ctx):
        d = ctx.step("decide", lambda c: (setattr(counts, "decide", counts.decide + 1) or {"n": n}))
        r = ctx.step("act", lambda c: c.call(tool, {"n": d["n"]}))
        return {"status": "met", "answer": f"answer {r['value']}"}

    return program


def _finish_turn(svc, nid: str, n: int, counts: Counts, goal: str | None = None) -> str:
    mid = svc.create_turn(nid, goal or f"question number {n}")
    out = Runtime(svc.store).run(mid, _program(n, counts))
    assert out["status"] == "completed"
    return mid


def _notebook(svc, turns: int = 3):
    nid = svc.notebooks.create("Sales")["id"]
    counts = Counts()
    ids = [_finish_turn(svc, nid, n + 1, counts) for n in range(turns)]
    return nid, ids, counts


def _dump(svc, tables=("missions", "events", "steps", "tool_calls", "questions", "agent_notes", "sources", "derived_tables",
                       "derived_versions", "notebooks")):
    """Every row of the ledger, for a before/after comparison."""
    with svc.store.lock:
        return {t: sorted(json.dumps(dict(r), sort_keys=True, default=str)
                          for r in svc.store.conn.execute(f"SELECT * FROM {t}")) for t in tables}


def _table_version(svc, nid, turn_id, name="clean", version=1, content="a,b\n1,2\n"):
    """A derived table version written by a turn, without running a sandbox."""
    path = svc.notebooks.root / nid / "tables" / name / f"v{version}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    with svc.store._tx() as c:
        t = c.execute("SELECT id FROM derived_tables WHERE notebook_id=? AND name=?", (nid, name)).fetchone()
        tid = t["id"] if t else name + "-id-" + nid[:6]
        if not t:
            c.execute("INSERT INTO derived_tables (id, notebook_id, name, description, created) VALUES (?,?,?,?,?)",
                      (tid, nid, name, "", time.time()))
        c.execute("INSERT INTO derived_versions (id, table_id, version, rows, columns_json, sha256, file_path, lineage_json, "
                  "turn_id, created) VALUES (?,?,?,?,?,?,?,?,?,?)",
                  (f"{tid}-v{version}", tid, version, 1, json.dumps(["a", "b"]), f"sha{version}", str(path),
                   json.dumps({"run_id": RUN, "file": "out/t.csv"}), turn_id, time.time()))


# -- rewind ---------------------------------------------------------------------------------------------------------

def test_rewind_hides_the_later_questions_and_deletes_nothing(svc):
    nid, ids, _ = _notebook(svc)
    before = _dump(svc)
    out = svc.checkpoints.rewind_notebook(nid, ids[1])
    assert out["prefill"] == "question number 2"
    assert out["dropped"] == ids[1:]
    assert [t["id"] for t in svc.notebooks.turns(nid)] == [ids[0]]
    assert svc.notebooks.list()[0]["turns"] == 1
    after = _dump(svc)
    # every original row is still there, byte for byte, except the flag on the dropped missions and their marker events
    for table in ("steps", "tool_calls", "questions", "agent_notes", "sources"):
        assert after[table] == before[table], table
    assert set(before["events"]) <= set(after["events"])
    assert len(after["missions"]) == len(before["missions"])
    kinds = [e["kind"] for e in svc.store.events(ids[2])]
    assert kinds[-1] == "mission.dropped"
    assert svc.store.mission(ids[2])["dropped_by"] == out["rewind_id"]
    assert svc.store.mission(ids[0])["dropped_by"] is None
    listing = svc.checkpoints.rewinds(nid)
    assert listing[0]["dropped"][0] == {"id": ids[1], "goal": "question number 2"}


def test_the_next_question_after_a_rewind_takes_the_place_and_parent_of_the_cut(svc):
    nid, ids, counts = _notebook(svc)
    svc.checkpoints.rewind_notebook(nid, ids[1])
    new = svc.create_turn(nid, "a different second question")
    m = svc.store.mission(new)
    assert m["seq"] == 2 and m["parent_turn_id"] == ids[0]
    assert [t["seq"] for t in svc.notebooks.turns(nid)] == [1, 2]
    ctx = svc.notebooks.context(nid)
    assert "a different second question" not in ctx          # not finished yet
    assert "question number 3" not in ctx and "question number 2" not in ctx   # dropped turns do not reach the planner


def test_a_dropped_question_cannot_be_resumed_or_rewound_again(svc):
    nid, ids, _ = _notebook(svc)
    svc.checkpoints.rewind_notebook(nid, ids[1])
    with pytest.raises(Refused):
        svc.checkpoints.rewind_notebook(nid, ids[2])
    with pytest.raises(Refused):
        svc.checkpoints.fork_notebook(nid, ids[1])
    svc.store.set_status(ids[2], "interrupted")
    with pytest.raises(ValueError, match="rewind"):
        svc.start(ids[2])


def test_rewind_refuses_while_a_question_is_running_and_changes_nothing(svc):
    nid, ids, _ = _notebook(svc, 2)
    running = svc.create_turn(nid, "still working on this")      # status created: active
    before = _dump(svc)
    with pytest.raises(Busy):
        svc.checkpoints.rewind_notebook(nid, ids[0])
    assert _dump(svc) == before
    assert running


def test_rewind_hides_the_tables_the_dropped_turns_saved_and_undo_restores_them(svc):
    nid, ids, _ = _notebook(svc)
    _table_version(svc, nid, ids[0], "clean", 1)
    _table_version(svc, nid, ids[1], "clean", 2, content="a,b\n9,9\n")
    _table_version(svc, nid, ids[2], "extra", 1)
    assert {t["name"]: t["version"] for t in svc.notebooks.tables(nid)} == {"clean": 2, "extra": 1}
    out = svc.checkpoints.rewind_notebook(nid, ids[1])
    assert {(h["name"], h["version"]) for h in out["tables_hidden"]} == {("clean", 2), ("extra", 1)}
    assert {t["name"]: t["version"] for t in svc.notebooks.tables(nid)} == {"clean": 1}
    assert svc.notebooks.table_path(nid, "clean").read_text() == "a,b\n1,2\n"
    assert [v["version"] for v in svc.notebooks.table_versions(nid, "clean")] == [1]
    with pytest.raises(KeyError):
        svc.notebooks.table_path(nid, "extra")
    # the files of hidden versions are still on disk
    assert (svc.notebooks.root / nid / "tables" / "clean" / "v2.csv").is_file()
    svc.checkpoints.undo_rewind(out["rewind_id"])
    assert {t["name"]: t["version"] for t in svc.notebooks.tables(nid)} == {"clean": 2, "extra": 1}


def test_a_table_saved_after_a_rewind_gets_a_new_version_number_and_is_never_called_unchanged(svc, tmp_path, monkeypatch):
    monkeypatch.setattr(notebooks_module, "SANDBOX_ROOT", tmp_path / "sandbox")
    nid, ids, _ = _notebook(svc)
    out_dir = tmp_path / "sandbox" / RUN / notebooks_module.OUT_DIR
    out_dir.mkdir(parents=True)
    (out_dir / "t.csv").write_text("a,b\n1,2\n")
    first = svc.notebooks.promote(nid, RUN, "t.csv", "clean", turn_id=ids[0])
    (out_dir / "t.csv").write_text("a,b\n7,7\n")
    svc.notebooks.promote(nid, RUN, "t.csv", "clean", turn_id=ids[1])
    svc.checkpoints.rewind_notebook(nid, ids[1])
    (out_dir / "t.csv").write_text("a,b\n7,7\n")        # the same bytes the hidden version holds
    again = svc.notebooks.promote(nid, RUN, "t.csv", "clean", turn_id=svc.create_turn(nid, "second try"))
    assert first["version"] == 1 and again["version"] == 3 and "unchanged" not in again
    assert [v["version"] for v in svc.notebooks.table_versions(nid, "clean")] == [1, 3]


def test_undo_works_until_a_question_is_sent_then_never(svc):
    nid, ids, _ = _notebook(svc)
    out = svc.checkpoints.rewind_notebook(nid, ids[1])
    done = svc.checkpoints.undo_rewind(out["rewind_id"])
    assert done["restored"] == ids[1:]
    assert [t["id"] for t in svc.notebooks.turns(nid)] == ids
    assert svc.store.mission(ids[1])["dropped_by"] is None
    assert [e["kind"] for e in svc.store.events(ids[1])][-1] == "mission.restored"
    with pytest.raises(Refused):
        svc.checkpoints.undo_rewind(out["rewind_id"])        # already undone
    out2 = svc.checkpoints.rewind_notebook(nid, ids[1])
    svc.create_turn(nid, "sent after the rewind")
    with pytest.raises(Refused, match="no longer"):
        svc.checkpoints.undo_rewind(out2["rewind_id"])
    assert [t["seq"] for t in svc.notebooks.turns(nid)] == [1, 2]


def test_only_the_latest_rewind_can_be_undone(svc):
    nid, ids, _ = _notebook(svc, 4)
    first = svc.checkpoints.rewind_notebook(nid, ids[2])
    second = svc.checkpoints.rewind_notebook(nid, ids[1])
    with pytest.raises(Refused, match="later rewind"):
        svc.checkpoints.undo_rewind(first["rewind_id"])
    svc.checkpoints.undo_rewind(second["rewind_id"])
    assert [t["id"] for t in svc.notebooks.turns(nid)] == ids[:2]
    svc.checkpoints.undo_rewind(first["rewind_id"])
    assert [t["id"] for t in svc.notebooks.turns(nid)] == ids


def test_a_quick_forecast_rewind_hides_it_and_undo_shows_it_again(svc):
    mid = svc.create("forecast next month sales")
    Runtime(svc.store).run(mid, _program(1, Counts()))
    out = svc.checkpoints.rewind_forecast(mid)
    assert out["prefill"] == "forecast next month sales"
    assert mid not in [m["id"] for m in svc.list()]
    assert svc.store.mission(mid)["status"] == "completed"       # the ledger is as it was
    with pytest.raises(Refused):
        svc.checkpoints.rewind_forecast(mid)
    svc.checkpoints.undo_rewind(out["rewind_id"])
    assert mid in [m["id"] for m in svc.list()]
    running = svc.create("a forecast that is still running")
    with pytest.raises(Busy):
        svc.checkpoints.rewind_forecast(running)
    nid, ids, _ = _notebook(svc, 1)
    with pytest.raises(KeyError):
        svc.checkpoints.rewind_forecast(ids[0])                  # a notebook question has its own route


# -- fork -----------------------------------------------------------------------------------------------------------

def test_fork_copies_the_earlier_turns_under_new_ids_and_leaves_the_original_untouched(svc):
    nid, ids, _ = _notebook(svc)
    before = _dump(svc)
    out = svc.checkpoints.fork_notebook(nid, ids[2])
    assert out["prefill"] == "question number 3" and out["copied_turns"] == 2
    assert out["title"] == "Sales (Fork)"
    after = _dump(svc)
    for table, rows in before.items():
        assert set(rows) <= set(after[table]), table              # nothing of the original changed or went
    new = svc.notebooks.turns(out["notebook_id"])
    assert [t["seq"] for t in new] == [1, 2] and not {t["id"] for t in new} & set(ids)
    assert [t["goal"] for t in new] == ["question number 1", "question number 2"]
    assert [t["answer"]["answer"] for t in new] == ["answer 1", "answer 2"]
    m = [svc.store.mission(t["id"]) for t in new]
    assert m[0]["parent_turn_id"] is None and m[1]["parent_turn_id"] == m[0]["id"]
    assert [x["forked_from"] for x in m] == ids[:2]
    nb = svc.notebooks
    assert nb.get(out["notebook_id"])["title"] == "Sales (Fork)"
    with svc.store.lock:
        row = svc.store.conn.execute("SELECT forked_from, forked_at_seq FROM notebooks WHERE id=?", (out["notebook_id"],)).fetchone()
    assert (row["forked_from"], row["forked_at_seq"]) == (nid, 3)
    assert svc.checkpoints.fork_notebook(nid, ids[2])["title"] == "Sales (Fork 2)"


def test_fork_at_the_first_question_is_an_empty_notebook(svc):
    nid, ids, _ = _notebook(svc, 2)
    out = svc.checkpoints.fork_notebook(nid, ids[0])
    assert out["copied_turns"] == 0 and svc.notebooks.turns(out["notebook_id"]) == []


def test_a_finished_turn_in_a_fork_never_runs_a_tool_or_asks_the_model_again(svc):
    nid, ids, counts = _notebook(svc)
    out = svc.checkpoints.fork_notebook(nid, ids[2])
    new_ids = [t["id"] for t in svc.notebooks.turns(out["notebook_id"])]
    calls, decides = counts.tool, counts.decide
    # run each copied turn's own program again: every step and every tool call must come from the copied ledger
    for n, mid in enumerate(new_ids, start=1):
        res = Runtime(svc.store).run(mid, _program(n, counts))
        assert res["status"] == "completed"
    assert (counts.tool, counts.decide) == (calls, decides)


def test_every_copied_tool_call_has_its_own_key_and_events_name_the_new_keys(svc):
    nid, ids, _ = _notebook(svc, 2)
    out = svc.checkpoints.fork_notebook(nid, svc.create_turn(nid, "third"))
    old_keys = {r["idem_key"] for r in svc.store.conn.execute("SELECT idem_key FROM tool_calls WHERE mission_id IN (?,?)", ids)}
    (copy,) = [t["id"] for t in svc.notebooks.turns(out["notebook_id"])][:1]
    rows = svc.store.conn.execute("SELECT * FROM tool_calls WHERE mission_id=?", (copy,)).fetchall()
    assert len(rows) == 1
    key = rows[0]["idem_key"]
    assert key not in old_keys
    assert key == idem_key(copy, rows[0]["step_idx"], "work", json.loads(rows[0]["args_json"]))
    assert rows[0]["status"] == "done" and json.loads(rows[0]["result_json"])["value"] == 1
    named = [e["payload"].get("key") for e in svc.store.events(copy) if e["kind"] in ("tool.started", "tool.done")]
    assert named == [key, key]


def test_a_call_a_crash_left_started_stays_started_and_is_not_retried_in_the_fork(svc):
    nid = svc.notebooks.create("Crash")["id"]
    counts = Counts()
    first = _finish_turn(svc, nid, 1, counts)
    crashed = svc.create_turn(nid, "this one crashed mid-tool")
    # the second turn: a stored decision, then a non-idempotent call that started and never recorded an outcome
    store = svc.store
    store.finish_step(crashed, 0, "decide", {"v": {"n": 2}})
    key = idem_key(crashed, 1, "work", {"n": 2})
    store.tool_start(crashed, 1, "work", {"n": 2}, key)
    store.set_status(crashed, "running")
    assert crashed in store.interrupt_orphans()
    # the crashed turn is the last one, so the fork is cut at a question sent after it
    nxt = svc.create_turn(nid, "next question")
    out = svc.checkpoints.fork_notebook(nid, nxt)
    copies = [t["id"] for t in svc.notebooks.turns(out["notebook_id"])]
    assert len(copies) == 2
    copy = copies[1]
    row = store.conn.execute("SELECT * FROM tool_calls WHERE mission_id=?", (copy,)).fetchone()
    assert row["status"] == "started" and row["result_json"] is None
    assert store.mission(copy)["status"] == "interrupted"
    res = Runtime(store).run(copy, _program(2, counts, idempotent=False))
    assert res["status"] == "outcome_unknown"
    assert counts.tool == 1                                      # only the first turn's call ever ran
    assert first


def test_a_turn_still_active_when_copied_is_stored_interrupted_with_its_questions_closed(svc):
    nid = svc.notebooks.create("Waiting")["id"]
    counts = Counts()
    _finish_turn(svc, nid, 1, counts)
    waiting = svc.create_turn(nid, "ask me something")
    key = idem_key(waiting, 0, "ask_user", {"q": "which?"})
    svc.store.tool_start(waiting, 0, "ask_user", {"q": "which?"}, key)
    qid = svc.store.open_question(waiting, key, {"q": "which?"})
    # an active turn that sits before the cut (a leftover of a crash): numbered first, with a finished turn after it
    with svc.store._tx() as c:
        c.execute("UPDATE missions SET status='waiting_user' WHERE id=?", (waiting,))
        c.execute("UPDATE missions SET seq=1 WHERE id=?", (waiting,))
        c.execute("UPDATE missions SET seq=2 WHERE notebook_id=? AND id!=?", (nid, waiting))
    later = svc.store.conn.execute("SELECT id FROM missions WHERE notebook_id=? AND id!=?", (nid, waiting)).fetchone()["id"]
    out = svc.checkpoints.fork_notebook(nid, later)
    (copy,) = [t["id"] for t in svc.notebooks.turns(out["notebook_id"])]
    assert svc.store.mission(copy)["status"] == "interrupted"
    assert svc.store.open_questions(copy) == []
    st = svc.store.conn.execute("SELECT status FROM questions WHERE mission_id=?", (copy,)).fetchone()["status"]
    assert st == "cancelled"
    assert svc.store.conn.execute("SELECT status FROM questions WHERE id=?", (qid,)).fetchone()["status"] == "open"


def test_fork_copies_sources_and_tables_from_before_the_cut_only_and_the_sharing_policy(svc):
    nid = svc.notebooks.create("Files")["id"]
    counts = Counts()
    a = svc.notebooks.add_source(nid, "early.csv", b"x,y\n1,2\n", profile_async=False)
    t1 = _finish_turn(svc, nid, 1, counts)
    _table_version(svc, nid, t1, "clean", 1)
    time.sleep(0.01)
    t2 = _finish_turn(svc, nid, 2, counts)
    _table_version(svc, nid, t2, "late", 1)
    time.sleep(0.01)
    cut = svc.create_turn(nid, "third")
    time.sleep(0.01)
    svc.notebooks.add_source(nid, "late.csv", b"x,y\n3,4\n", profile_async=False)   # added after the question being cut at
    with svc.store._tx() as c:
        c.execute("UPDATE sources SET status='ready'")
    svc.set_sharing(nid, {"web_research": False, "providers": ["nvidia"]})
    out = svc.checkpoints.fork_notebook(nid, cut)
    new = out["notebook_id"]
    assert [s["name"] for s in svc.notebooks.sources(new)] == ["early.csv"]
    assert svc.notebooks.sources(new)[0]["id"] != a["id"]
    assert svc.notebooks.sources(new)[0]["dataset_id"] == a["dataset_id"]   # the stored file is shared, not copied
    assert {t["name"] for t in svc.notebooks.tables(new)} == {"clean", "late"}
    path = svc.notebooks.table_path(new, "clean")
    assert path.read_text() == "a,b\n1,2\n" and str(svc.notebooks.root / new) in str(path)
    assert svc.get_sharing(new) == {"web_research": False, "providers": ["nvidia"]}
    # a fork at the second question does not carry the table the second turn saved
    early = svc.checkpoints.fork_notebook(nid, t2)["notebook_id"]
    assert {t["name"] for t in svc.notebooks.tables(early)} == {"clean"}


def test_a_failed_fork_leaves_no_notebook_and_no_files(svc, monkeypatch):
    nid, ids, _ = _notebook(svc)
    _table_version(svc, nid, ids[0], "clean", 1)
    def boom(*a, **k):
        raise RuntimeError("disk full")
    monkeypatch.setattr("prophet_agent.checkpoints.shutil.copyfile", boom)
    before = _dump(svc)
    with pytest.raises(RuntimeError):
        svc.checkpoints.fork_notebook(nid, ids[2])
    assert _dump(svc) == before
    assert sorted(p.name for p in svc.notebooks.root.iterdir()) == [nid]


def test_deleting_one_side_keeps_the_run_folder_the_other_still_names(svc, tmp_path, monkeypatch):
    monkeypatch.setattr(cleanup, "SANDBOX_ROOT", tmp_path / "sandbox")
    folder = tmp_path / "sandbox" / RUN
    folder.mkdir(parents=True)
    (folder / "chart.png").write_bytes(b"png")
    nid, ids, _ = _notebook(svc)
    fork = svc.checkpoints.fork_notebook(nid, ids[2])["notebook_id"]
    svc.notebooks.delete(nid)
    assert folder.is_dir()                                         # the fork's turns still name it
    svc.notebooks.delete(fork)
    assert not folder.exists()                                     # nothing names it any more


def test_deleting_a_notebook_removes_its_hidden_turns_and_rewind_records_too(svc):
    nid, ids, _ = _notebook(svc)
    svc.checkpoints.rewind_notebook(nid, ids[1])
    svc.notebooks.delete(nid)
    with svc.store.lock:
        left = [svc.store.conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
                for t in ("missions", "events", "tool_calls", "rewinds")]
    assert left == [0, 0, 0, 0]
