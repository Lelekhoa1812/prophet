"""Round 6h (2026-11-23): a sandbox cell that prints nothing and writes no file says so.

Live Beijing 0137871e (build cf1f0cc): two silent cells with the same code gave the planner nothing to act on, and the run
stopped as a stall after 103 calls. Every series here is synthetic.
"""
from __future__ import annotations

from prophet_agent.tools import sandbox_tool


def _result(**kw):
    base = {"ok": True, "run_id": "r", "stdout": "", "stderr": "", "timed_out": False, "outputs": []}
    base.update(kw)
    return base


def test_a_silent_cell_with_no_file_is_told_what_to_do_instead():
    out = sandbox_tool.observe(_result())
    assert "printed nothing" in out["note"] and "next cell" in out["note"]


def test_a_cell_that_printed_or_wrote_a_file_gets_no_such_note():
    assert "note" not in sandbox_tool.observe(_result(stdout="mean 72.7"))
    assert "note" not in sandbox_tool.observe(_result(outputs=[{"path": "out/x.csv", "bytes": 3, "sha256": "a"}]))


def test_a_failed_silent_cell_is_not_given_the_silent_note():
    assert "note" not in sandbox_tool.observe(_result(ok=False, returncode=1))
