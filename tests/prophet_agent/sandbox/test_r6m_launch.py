"""Round 6m, item 3 (launch and limits): the child runs in its own session and its limits are in force before its code runs.

The runner used to apply the limits and the session with preexec_fn. Python documents preexec_fn as unsafe when the parent
has threads, and the parent does (the goal loop, notebooks and the API service all start threads): a fork can deadlock in
the child on a lock another thread held. The fix moves the limits into the child's boot code and starts the child with
start_new_session=True, which runs no Python between fork and exec. A deadlock cannot be reproduced on demand, so these
tests pin the behaviour that the move must keep: the session, the limits, and concurrent launches from threads.
"""
from __future__ import annotations

import concurrent.futures

from prophet_agent.sandbox.runner import run_python


def test_the_program_leads_its_own_session_and_process_group():
    r = run_python("import os\nprint('LEADER', os.getsid(0) == os.getpid(), os.getpgrp() == os.getpid())\n")
    assert r.ok, r.stderr
    assert "LEADER True True" in r.stdout


def test_the_limits_are_in_force_before_the_program_runs():
    code = ("import resource\n"
            "print('LIMITS', resource.getrlimit(resource.RLIMIT_CPU), resource.getrlimit(resource.RLIMIT_CORE),\n"
            "      resource.getrlimit(resource.RLIMIT_NOFILE), resource.getrlimit(resource.RLIMIT_FSIZE))\n")
    r = run_python(code, timeout_s=20, file_mb=64)
    assert r.ok, r.stderr
    assert "LIMITS (22, 22) (0, 0) (256, 256) (67108864, 67108864)" in r.stdout


def test_runs_launched_from_several_threads_all_finish():
    def one(i: int):
        return run_python(f"print('RUN', {i})\n", timeout_s=20)
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(one, range(8)))
    assert all(r.ok for r in results), [r.stderr for r in results if not r.ok]
    assert sorted(r.stdout.strip() for r in results) == [f"RUN {i}" for i in range(8)]
