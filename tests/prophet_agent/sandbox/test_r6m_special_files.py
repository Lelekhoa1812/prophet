"""Round 6m, item 3 (audit-hook bypass): special files and file flags outside the run folder.

Probe (before the fix): os.mkfifo() and os.mknod() created a node outside the run folder and raised no audit event, so
the hook could not see them. os.chflags() on an existing file outside the run changed its flags. The fix wraps mkfifo and
mknod in the os and posix modules (the same function, reachable by either name), and checks chflags and lchflags like
the other metadata calls. Writes inside the run folder keep working.
"""
from __future__ import annotations

import os
import stat
import sys

import pytest

from prophet_agent.sandbox.runner import run_python


def test_a_fifo_cannot_be_created_outside_the_run_folder(tmp_path):
    target = tmp_path / "fifo_outside"
    r = run_python(f"import os\ntry:\n    os.mkfifo({str(target)!r})\n    print('MADE')\n"
                   f"except PermissionError:\n    print('REFUSED')\n")
    assert r.ok, r.stderr
    assert "MADE" not in r.stdout and "REFUSED" in r.stdout and not target.exists()


def test_a_fifo_cannot_be_created_through_the_posix_module(tmp_path):
    target = tmp_path / "fifo_posix"
    r = run_python(f"import posix\ntry:\n    posix.mkfifo({str(target)!r})\n    print('MADE')\n"
                   f"except PermissionError:\n    print('REFUSED')\n")
    assert r.ok, r.stderr
    assert "MADE" not in r.stdout and not target.exists()


def test_a_device_node_cannot_be_created_outside_the_run_folder(tmp_path):
    target = tmp_path / "node_outside"
    r = run_python(f"import os, stat\ntry:\n    os.mknod({str(target)!r}, 0o600 | stat.S_IFIFO)\n    print('MADE')\n"
                   f"except PermissionError:\n    print('REFUSED')\n")
    assert r.ok, r.stderr
    assert "MADE" not in r.stdout and not target.exists()


def test_a_fifo_inside_the_run_folder_still_works():
    r = run_python("import os\nos.mkfifo('out/pipe')\nprint('MADE', os.path.exists('out/pipe'))\n")
    assert r.ok, r.stderr
    assert "MADE True" in r.stdout


@pytest.mark.skipif(sys.platform != "darwin", reason="file flags (chflags) are a BSD and macOS call")
def test_file_flags_cannot_be_changed_outside_the_run_folder(tmp_path):
    target = tmp_path / "flags_target"
    target.write_text("x")
    before = os.stat(target).st_flags
    r = run_python(f"import os, stat\ntry:\n    os.chflags({str(target)!r}, stat.UF_NODUMP)\n    print('CHANGED')\n"
                   f"except PermissionError:\n    print('REFUSED')\n")
    assert r.ok, r.stderr
    assert "CHANGED" not in r.stdout and "REFUSED" in r.stdout
    assert os.stat(target).st_flags == before


def test_permissions_cannot_be_changed_outside_the_run_folder(tmp_path):
    target = tmp_path / "mode_target"
    target.write_text("x")
    os.chmod(target, 0o600)
    r = run_python(f"import os\ntry:\n    os.chmod({str(target)!r}, 0o777)\n    print('CHANGED')\n"
                   f"except PermissionError:\n    print('REFUSED')\n")
    assert r.ok, r.stderr
    assert "CHANGED" not in r.stdout
    assert stat.S_IMODE(os.stat(target).st_mode) == 0o600
