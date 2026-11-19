import importlib
import time

from lab.api import jobs


def fresh(tmp_path, monkeypatch):
    monkeypatch.setattr(jobs, "JOB_DIR", tmp_path)
    jobs._jobs.clear()
    return jobs


def wait(jid):
    for _ in range(100):
        j = jobs.get(jid)
        if j["status"] != "running":
            return j
        time.sleep(0.02)
    raise AssertionError("job did not finish")


def test_result_survives_a_restart(tmp_path, monkeypatch):
    fresh(tmp_path, monkeypatch)
    jid = jobs.submit("t", lambda progress, **kw: {"answer": 42})
    assert wait(jid)["result"] == {"answer": 42}
    jobs._jobs.clear()                                  # process restart
    jobs.load_saved()
    assert jobs.get(jid)["status"] == "done" and jobs.get(jid)["result"] == {"answer": 42}


def test_running_job_is_marked_interrupted_after_restart(tmp_path, monkeypatch):
    fresh(tmp_path, monkeypatch)
    jobs._save({"id": "J-dead", "kind": "replay", "status": "running", "progress": 0.4, "message": "", "started": "2026-01-01T00:00:00+00:00", "result": None, "error": None})
    jobs._jobs.clear()
    jobs.load_saved()
    j = jobs.get("J-dead")
    assert j["status"] == "interrupted" and "restart" in j["error"]


def test_failed_job_keeps_its_error(tmp_path, monkeypatch):
    fresh(tmp_path, monkeypatch)
    def boom(progress, **kw): raise ValueError("bad")
    j = wait(jobs.submit("t", boom))
    assert j["status"] == "error" and "bad" in j["error"]
