"""Round 7, item 4: there is no working memory cap on macOS, and the run's result must say so.

Investigation (measured on Darwin 25.6.0, Python 3.12): setrlimit fails with EINVAL for every finite value of RLIMIT_AS,
RLIMIT_DATA and RLIMIT_RSS (soft only, or soft and hard). Only the unlimited value is accepted. So no address-space cap can
be set on this macOS, and the run cannot pretend otherwise. The result records the gap in ``not_enforced``
(``memory_mb`` on macOS), next to the existing ``memory_limit_enforced`` flag.
"""
from __future__ import annotations

import platform
import uuid

import pytest

from prophet_agent.sandbox.runner import run_python

DARWIN_ONLY = pytest.mark.skipif(platform.system() != "Darwin",
                                 reason="the macOS memory record: on Linux the address-space cap is set and enforced")
LINUX_ONLY = pytest.mark.skipif(platform.system() != "Linux",
                                reason="the enforced memory cap exists on Linux only")


@DARWIN_ONLY
def test_the_result_records_that_the_memory_cap_is_not_enforced_on_darwin():
    r = run_python("print('RAN')\n")
    assert r.ok, r.stderr
    assert r.memory_limit_enforced is False
    assert r.limits["memory_limit_enforced"] is False
    assert "memory_mb" in r.not_enforced
    assert r.to_dict()["not_enforced"] == r.not_enforced


@DARWIN_ONLY
def test_the_memory_record_is_kept_in_the_stored_manifest():
    run_id = uuid.uuid4().hex
    first = run_python("print('RAN')\n", run_id=run_id)
    again = run_python("print('SHOULD NOT RUN')\n", run_id=run_id)   # a finished run returns its stored result
    assert "memory_mb" in first.not_enforced
    assert again.not_enforced == first.not_enforced
    assert "SHOULD NOT RUN" not in again.stdout


@LINUX_ONLY
def test_the_memory_cap_is_enforced_and_not_recorded_as_missing_on_linux():
    r = run_python("print('RAN')\n")
    assert r.ok, r.stderr
    assert r.memory_limit_enforced is True
    assert "memory_mb" not in r.not_enforced
