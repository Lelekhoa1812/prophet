"""Round 6k (2026-11-23): a relative path given with dir_fd cannot reach a folder outside the run.

Audit (sandbox/runner.py:83-88, HIGH): the audit hook checks the relative name against the working directory, but the kernel
resolves it against the descriptor's folder. os.remove("f.txt", dir_fd=fd) with fd on a folder outside the run removed the
file outside. The sandbox now resolves the descriptor's folder and applies the same check. Test files are in pytest's temp
folder; nothing here touches the real project.
"""
from __future__ import annotations

import os

from prophet_agent.sandbox.runner import run_python


def _victim_folder(tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "keep.txt").write_text("keep")
    return outside


def test_removing_a_file_outside_the_run_through_a_descriptor_is_refused(tmp_path):
    outside = _victim_folder(tmp_path)
    code = (f"import os\nfd = os.open({str(outside)!r}, os.O_RDONLY)\n"
            "os.remove('keep.txt', dir_fd=fd)\n")
    r = run_python(code)
    assert not r.ok
    assert "changing files outside the run folder is blocked" in r.stderr
    assert (outside / "keep.txt").read_text() == "keep"


def test_making_a_folder_outside_the_run_through_a_descriptor_is_refused(tmp_path):
    outside = _victim_folder(tmp_path)
    code = (f"import os\nfd = os.open({str(outside)!r}, os.O_RDONLY)\n"
            "os.mkdir('made', dir_fd=fd)\n")
    r = run_python(code)
    assert not r.ok and not (outside / "made").exists()


def test_writing_a_file_outside_the_run_through_a_descriptor_is_refused(tmp_path):
    outside = _victim_folder(tmp_path)
    code = (f"import os\nfd = os.open({str(outside)!r}, os.O_RDONLY)\n"
            "os.open('new.txt', os.O_CREAT | os.O_WRONLY, dir_fd=fd)\n")
    r = run_python(code)
    assert not r.ok and not (outside / "new.txt").exists()


def test_a_descriptor_on_the_run_folder_still_works(tmp_path):
    # The guard must not block legitimate use: an out/ folder reached through a descriptor inside the run folder.
    code = ("import os\nfd = os.open('out', os.O_RDONLY)\n"
            "g = os.open('made.txt', os.O_CREAT | os.O_WRONLY, dir_fd=fd); os.close(g)\n"
            "print('made' if os.path.exists('out/made.txt') else 'missing')\n")
    r = run_python(code)
    assert r.ok and "made" in r.stdout
