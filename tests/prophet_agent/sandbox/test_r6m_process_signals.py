"""Round 6m, item 3 (audit-hook bypass): process creation and signals outside the run.

Probe (before the fix): os.forkpty() created a child process (its audit event was not in the refused list), and os.kill()
and os.killpg() signalled any process of the same user, the parent included. The fix refuses forkpty with the other
process-creating calls, and lets a program signal only its own process and its own process group.
"""
from __future__ import annotations

from prophet_agent.sandbox.runner import run_python


def test_forkpty_cannot_start_a_process():
    code = ("import os\n"
            "try:\n    pid, fd = os.forkpty()\nexcept PermissionError:\n    print('REFUSED')\n    raise SystemExit(0)\n"
            "if pid == 0:\n    os._exit(0)\n"
            "os.waitpid(pid, 0)\nprint('FORKED')\n")
    r = run_python(code)
    assert r.ok, r.stderr
    assert "FORKED" not in r.stdout and "REFUSED" in r.stdout


def test_spawning_with_posix_spawnp_is_still_refused():
    code = ("import os\n"
            "try:\n    os.posix_spawnp('true', ['true'], {})\n    print('SPAWNED')\nexcept PermissionError:\n    print('REFUSED')\n")
    r = run_python(code)
    assert r.ok, r.stderr
    assert "SPAWNED" not in r.stdout and "REFUSED" in r.stdout


def test_a_signal_to_the_parent_process_is_refused():
    code = ("import os\n"
            "try:\n    os.kill(os.getppid(), 0)\n    print('SIGNALLED')\n"
            "except PermissionError:\n    print('REFUSED')\n")
    r = run_python(code)
    assert r.ok, r.stderr
    assert "SIGNALLED" not in r.stdout and "REFUSED" in r.stdout


def test_a_signal_to_another_process_group_is_refused():
    code = ("import os\n"
            "try:\n    os.killpg(os.getpgid(os.getppid()), 0)\n    print('SIGNALLED')\n"
            "except PermissionError:\n    print('REFUSED')\n")
    r = run_python(code)
    assert r.ok, r.stderr
    assert "SIGNALLED" not in r.stdout and "REFUSED" in r.stdout


def test_a_signal_to_the_program_itself_and_its_own_group_still_works():
    code = ("import os\n"
            "os.kill(os.getpid(), 0); os.killpg(os.getpgrp(), 0); os.kill(0, 0)\n"
            "print('SELF_OK')\n")
    r = run_python(code)
    assert r.ok, r.stderr
    assert "SELF_OK" in r.stdout
