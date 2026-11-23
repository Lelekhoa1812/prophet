"""A failed tool call whose error is only whitespace (for example "\n") made _error_signature split a stripped text into no
lines and index the empty list, so the goal program raised IndexError. The failure is still a failure: the signature must be
a stable, non-empty string, and the functions that count repeated failures must not raise.
"""
from __future__ import annotations

import pytest

from prophet_agent.runtime.goal import _error_recurrence, _error_run, _error_signature, _failure_text


@pytest.mark.parametrize("error", ["\n", "   ", "\n\n\t", " \r\n "])
def test_a_whitespace_only_error_gives_a_stable_signature_not_an_exception(error):
    call = {"kind": "tool", "error": error}
    first = _error_signature(call)
    assert isinstance(first, str) and first.strip(), "a failed call must still have a signature"
    assert _error_signature(dict(call)) == first


def test_a_whitespace_only_stderr_tail_is_handled_the_same_way():
    call = {"kind": "tool", "result": {"ok": False, "stderr_tail": "\n"}}
    sig = _error_signature(call)
    assert isinstance(sig, str) and sig.strip()
    assert sig == _error_signature({"kind": "tool", "error": "\n"})


def test_repeated_whitespace_errors_are_counted_without_raising():
    history = [{"kind": "tool", "error": "\n"} for _ in range(3)]
    assert _error_run(history) == 3
    assert isinstance(_error_recurrence(history), int)


def test_a_real_error_line_still_gives_the_same_signature_as_before():
    assert _error_signature({"kind": "tool", "error": "KeyError: 'qty'\n"}) == "KeyError: 'qty'"


def test_failure_text_for_a_whitespace_error_does_not_raise():
    assert isinstance(_failure_text({"kind": "tool", "error": "\n"}), str)
