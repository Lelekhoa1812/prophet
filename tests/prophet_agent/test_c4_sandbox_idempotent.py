"""Round 4 C4, audit A5-10: sandbox_python is declared idempotent, so a re-run with the same idempotency key must give the same run.

Motivation vs Logic
Motivation: a crash after a sandbox run but before the ledger records it leaves the call "started". The tool is declared
idempotent, so the runner re-runs it with the same key. Before the fix the key was ignored: the code ran again in a new run
folder, the outputs were written twice, and the work was charged twice.
Logic: the key seeds the run id (its first 32 hex characters; the key already includes the mission, step, tool and args). A
run folder whose manifest exists is complete, and its stored result is returned without running the code again. A folder
without a manifest is an interrupted run: it is cleared and the code runs once under the same id.
"""
from __future__ import annotations

import hashlib

from prophet_agent.sandbox.runner import SANDBOX_ROOT, run_python
from prophet_agent.tools.sandbox_tool import sandbox_python_tool

APPEND_ONCE = "open('out/count.txt', 'a').write('x')\nprint('ran')\n"


def _key(label: str) -> str:
    return hashlib.sha256(f"c4-test:{label}".encode()).hexdigest()


def test_rerun_with_same_key_returns_the_same_run_and_does_not_run_the_code_again():
    tool = sandbox_python_tool()
    key = _key("rerun")
    first = tool.fn({"code": APPEND_ONCE}, key)
    second = tool.fn({"code": APPEND_ONCE}, key)
    assert second["run_id"] == first["run_id"] == key[:32]
    assert second["stdout"] == first["stdout"]
    assert second["outputs"] == first["outputs"]
    count = (SANDBOX_ROOT / first["run_id"] / "out" / "count.txt").read_text()
    assert count == "x", f"the code ran again on the re-run: count.txt holds {count!r}"


def test_an_interrupted_run_is_cleared_and_run_once_under_the_same_id():
    key = _key("interrupted")
    run_id = key[:32]
    folder = SANDBOX_ROOT / run_id
    if folder.exists():
        import shutil
        shutil.rmtree(folder)
    (folder / "out").mkdir(parents=True)
    (folder / "out" / "count.txt").write_text("stale")       # a half-written earlier attempt, no manifest
    tool = sandbox_python_tool()
    res = tool.fn({"code": APPEND_ONCE}, key)
    assert res["run_id"] == run_id
    assert (folder / "out" / "count.txt").read_text() == "x"


def test_a_plain_run_without_a_key_still_gets_a_fresh_id():
    a = run_python("print(1)")
    b = run_python("print(1)")
    assert a.run_id != b.run_id
