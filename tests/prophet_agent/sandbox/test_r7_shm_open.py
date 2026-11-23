"""Round 7, item 2 (audit gap): POSIX shared memory is not restricted.

shm_open creates a named object outside the run folder (on Linux, a file in /dev/shm), and it raises no audit event, so the
hook never sees it. Probe (before the fix): _posixshmem.shm_open created a named object and multiprocessing's
SharedMemory worked inside the sandbox. The fix refuses shm_open and shm_unlink the way the other blocked calls are
refused: a PermissionError with a message, from a wrapper installed before user code runs.
"""
from __future__ import annotations

import os
import sys

import pytest

from prophet_agent.sandbox.runner import run_python

NAME = "/prophet_r7_shm_probe"


@pytest.fixture(autouse=True)
def _no_leftover_shared_memory():
    yield
    try:
        import _posixshmem
        _posixshmem.shm_unlink(NAME)     # clear a segment a test left behind when the fix was missing
    except (ImportError, OSError):
        pass


def test_shm_open_is_refused_in_the_sandbox():
    code = (f"import _posixshmem, os\n"
            f"try:\n    fd = _posixshmem.shm_open({NAME!r}, os.O_CREAT | os.O_RDWR, mode=0o600)\n"
            f"    print('OPENED')\n    os.close(fd)\n"
            f"except PermissionError as e:\n    print('REFUSED', e)\n")
    r = run_python(code)
    assert r.ok, r.stderr
    assert "OPENED" not in r.stdout and "REFUSED" in r.stdout


def test_shared_memory_objects_cannot_be_created_through_multiprocessing():
    code = (f"from multiprocessing import shared_memory\n"
            f"try:\n    shm = shared_memory.SharedMemory(name={NAME.lstrip('/')!r}, create=True, size=16)\n"
            f"    print('CREATED')\n    shm.close(); shm.unlink()\n"
            f"except PermissionError as e:\n    print('REFUSED', e)\n")
    r = run_python(code)
    assert r.ok, r.stderr
    assert "CREATED" not in r.stdout and "REFUSED" in r.stdout


def test_shm_unlink_is_refused_in_the_sandbox():
    code = (f"import _posixshmem\n"
            f"try:\n    _posixshmem.shm_unlink({NAME!r})\n    print('UNLINKED')\n"
            f"except PermissionError as e:\n    print('REFUSED', e)\n")
    r = run_python(code)
    assert r.ok, r.stderr
    assert "UNLINKED" not in r.stdout and "REFUSED" in r.stdout


@pytest.mark.skipif(sys.platform not in ("darwin", "linux"), reason="POSIX shared memory exists on macOS and Linux")
def test_the_host_can_still_use_shared_memory_outside_the_sandbox():
    # The guard lives in the child only: the parent's own shared memory keeps working.
    import _posixshmem
    fd = _posixshmem.shm_open(NAME, os.O_CREAT | os.O_RDWR, mode=0o600)
    os.close(fd)
    _posixshmem.shm_unlink(NAME)
