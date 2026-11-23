"""Round 7, item 3 (audit gap): RLIMIT_NPROC is not restricted.

RLIMIT_NPROC bounds the processes and threads of the user, so a runaway thread pool or a fork bomb is stopped by the kernel
rather than by the audit hook alone. On Linux the parent counts the user's current tasks and the child gets a soft and hard
limit of that count plus PROCESS_HEADROOM (the child's threads and BLAS workers still start). The limit counts tasks of the
whole user, so a fixed small number would break a busy machine; the headroom is relative to the count at run start.

On macOS the limit is not set: the count is not read there, and the limit counts processes only through fork. The run must
still work and must record that no process limit was applied. The Linux tests are skipped on macOS for that reason.
"""
from __future__ import annotations

import sys

import pytest

from prophet_agent.sandbox.runner import PROCESS_HEADROOM, run_python

LINUX_ONLY = pytest.mark.skipif(
    sys.platform != "linux",
    reason="RLIMIT_NPROC is set only on Linux; on macOS the sandbox records no process limit (see the runner docstring)")
DARWIN_ONLY = pytest.mark.skipif(
    sys.platform != "darwin", reason="checks the macOS code path, where no process limit is applied")


@LINUX_ONLY
def test_the_child_runs_with_a_process_limit_on_linux():
    code = ("import resource\n"
            "soft, hard = resource.getrlimit(resource.RLIMIT_NPROC)\n"
            "print('NPROC', soft, hard)\n")
    r = run_python(code)
    assert r.ok, r.stderr
    soft, hard = (int(v) for v in r.stdout.split()[1:3])
    assert soft == hard == r.limits["processes_limit"]
    assert 0 < soft < 2 ** 32                  # finite: the child cannot start unbounded processes or threads
    assert soft >= PROCESS_HEADROOM            # headroom above the user's count at start


@LINUX_ONLY
def test_threads_stop_at_the_process_limit_on_linux():
    # Thread creation fails with RuntimeError once the limit is reached. The address-space cap is lifted (mem_mb) so that
    # it cannot be what stops the threads: glibc arenas would hit a 1 GiB cap after a few dozen threads. The run must
    # create most of the headroom before it stops, which only the process limit explains.
    code = ("import resource, threading\n"
            "soft = resource.getrlimit(resource.RLIMIT_NPROC)[0]\n"
            "threading.stack_size(262144)\n"
            "stop = threading.Event(); made = []\n"
            "try:\n"
            "    for _ in range(soft + 64):\n"
            "        t = threading.Thread(target=stop.wait, daemon=True); t.start(); made.append(t)\n"
            "    print('UNLIMITED', len(made), soft)\n"
            "except RuntimeError:\n"
            "    print('LIMITED', len(made), soft)\n"
            "finally:\n"
            "    stop.set()\n")
    r = run_python(code, timeout_s=60, mem_mb=4096)
    assert r.ok, r.stderr
    parts = r.stdout.split()
    assert parts[0] == "LIMITED", r.stdout
    made, soft = int(parts[1]), int(parts[2])
    assert made < soft
    assert made >= PROCESS_HEADROOM // 2, r.stdout


@DARWIN_ONLY
def test_on_darwin_no_process_limit_is_applied_and_the_run_still_works():
    r = run_python("import threading\nt = threading.Thread(target=lambda: None); t.start(); t.join()\nprint('RAN')\n")
    assert r.ok, r.stderr
    assert "RAN" in r.stdout
    assert r.limits["processes_limit"] is None
