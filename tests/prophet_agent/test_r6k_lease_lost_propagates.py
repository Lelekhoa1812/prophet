"""Round 6k (2026-11-23): a lost lease stops the mission; it is not a tool error.

Audit (runtime/goal.py:1177-1178, HIGH): the renewal that detects a takeover runs inside Ctx.call, and _act caught every
exception as tool evidence. The loop then went on writing after another runtime owned the mission. A lost lease now
propagates to the runner, which stops without writing more. An ordinary tool failure is still evidence for the planner.
All missions here are synthetic.
"""
import pytest

from prophet_agent.runtime import goal
from prophet_agent.runtime.store import LeaseLost


class _Tool:
    name = "probe"
    observe = None


class _Ctx:
    def __init__(self, exc: Exception):
        self._exc = exc

    def call(self, tool, args):
        raise self._exc


def test_a_lost_lease_is_not_turned_into_tool_evidence():
    with pytest.raises(LeaseLost):
        goal._act(_Ctx(LeaseLost("mission m1 lease was taken over")), _Tool(), "probe", {})


def test_an_ordinary_tool_failure_is_still_evidence_for_the_planner():
    obs = goal._act(_Ctx(ValueError("bad column")), _Tool(), "probe", {})
    assert obs["kind"] == "tool" and "bad column" in obs["error"]
