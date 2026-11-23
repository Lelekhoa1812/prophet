"""Round 6m, item 5 (output cap): the total bytes a run writes are capped, not only the size of each file.

Probe (before the fix): RLIMIT_FSIZE caps each file at 64 MB, but nothing caps the total. A loop that keeps creating
1 MB files runs until the timeout, so a 20-second run can fill the disk. The fix: a watcher thread samples the bytes in
the run folder (outside in/, which holds the copied inputs) and kills the process group when they pass ``disk_mb``. A
run that wrote too much is not ok and says so in ``over_disk_limit``.
"""
from __future__ import annotations

from prophet_agent.sandbox.runner import run_python


def test_writing_past_the_disk_quota_kills_the_run():
    code = ("i = 0\n"
            "while True:\n"
            "    open(f'out/f{i}.bin', 'wb').write(b'x' * 1000000)\n"
            "    i += 1\n")
    r = run_python(code, timeout_s=20, disk_mb=8)
    assert not r.ok and r.over_disk_limit is True
    assert r.timed_out is False


def test_a_run_within_the_quota_is_not_flagged():
    r = run_python("open('out/small.bin', 'wb').write(b'x' * 2000000)\nprint('FINE')\n", disk_mb=8)
    assert r.ok, r.stderr
    assert r.over_disk_limit is False and "FINE" in r.stdout


def test_the_disk_quota_is_recorded_with_the_other_limits():
    r = run_python("print('X')\n")
    assert r.limits["disk_mb"] == 512
