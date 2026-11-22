"""Round 4 C4, audit A5-06: a model call must not outlive the segment's time allowance, and a long call must keep the lease alive.

Motivation vs Logic
Motivation: live electricity segment 1 ran 922 s against a 900 s allowance, and two single tool-call steps took 162 s and 171 s.
The runtime checked time only between steps, so one model call could overrun. A streamed answer also never called the
checkpoint (which renews the lease and feels a cancel), so a call longer than the 300 s lease could lose its lease.
Logic: the goal passes the segment's remaining time to the transport as ``budget_s``, and the transport calls the
checkpoint between lines of a streamed answer, so a spent allowance or a cancel stops the call there. The stop is a
checkpoint (BudgetExceeded), never a failed step, so a resumed run asks the model again for that step.
"""
from __future__ import annotations

import json
import random

import httpx
import pytest

from prophet_agent.nim import NimClient
from prophet_agent.nim.transport import BUDGET_S
from prophet_agent.runtime.goal import NimBrain
from prophet_agent.runtime.runner import BudgetExceeded

KEYS = ("nvapi-AAAA1111", "nvapi-BBBB2222", "nvapi-CCCC3333")
M1 = "nvidia/nemotron-3-ultra-550b-a55b"
MSGS = [{"role": "system", "content": "You are terse."}, {"role": "user", "content": "Say hi."}]


class FakeClock:
    def __init__(self) -> None:
        self.t = 1000.0

    def __call__(self) -> float:
        return self.t

    def sleep(self, seconds: float) -> None:
        self.t += seconds


class NoShuffle(random.Random):
    def shuffle(self, x) -> None:
        return None


class SlowStream(httpx.SyncByteStream):
    """A streamed answer whose lines arrive 10 fake seconds apart."""

    def __init__(self, clock: FakeClock, lines: int, read: list):
        self.clock, self.lines, self.read = clock, lines, read

    def __iter__(self):
        for i in range(self.lines):
            self.clock.t += 10.0
            self.read.append(i)
            chunk = {"choices": [{"delta": {"content": "x"}, "finish_reason": None}]}
            yield f"data: {json.dumps(chunk)}\n\n".encode()
        yield b"data: [DONE]\n\n"


def _slow_client(clock: FakeClock, read: list, lines: int = 100) -> NimClient:
    def handler(request):
        return httpx.Response(200, stream=SlowStream(clock, lines, read), headers={"content-type": "text/event-stream"})

    return NimClient(list(KEYS), transport=httpx.MockTransport(handler), clock=clock, sleep=clock.sleep, rng=NoShuffle())


def test_a_streamed_call_stops_at_the_segment_deadline_not_at_its_end():
    clock = FakeClock()
    read: list = []
    deadline = clock.t + 60.0          # the segment's allowance: 60 fake seconds

    def checkpoint():
        if clock() > deadline:
            raise BudgetExceeded("max_seconds")

    client = _slow_client(clock, read)
    with pytest.raises(BudgetExceeded):
        client.complete([M1], MSGS, checkpoint=checkpoint)
    assert len(read) < 10, f"the call read {len(read)} of 100 lines past the allowance"


def test_a_long_streamed_call_calls_the_checkpoint_while_it_reads():
    clock = FakeClock()
    read: list = []
    calls = {"n": 0}

    def checkpoint():
        calls["n"] += 1

    reply = _slow_client(clock, read, lines=20).complete([M1], MSGS, checkpoint=checkpoint)
    assert reply.text == "x" * 20
    assert calls["n"] >= 15, f"the checkpoint ran {calls['n']} times during a 200 s call; the lease would expire"


class _Client:
    chain = (M1,)

    def __init__(self):
        self.kwargs = None

    def complete(self, models, messages, **kwargs):
        self.kwargs = kwargs
        from types import SimpleNamespace
        return SimpleNamespace(text=json.dumps({"ok": True}))


def test_the_brain_passes_the_segment_time_left_as_the_call_budget():
    c = _Client()
    brain = NimBrain(c)
    brain.bind(lambda: None, None, remaining=lambda: 120.0)
    brain.decide([], {}, "plan")
    assert c.kwargs["budget_s"] == 120.0


def test_the_call_budget_never_exceeds_the_transport_default():
    c = _Client()
    brain = NimBrain(c)
    brain.bind(lambda: None, None, remaining=lambda: 10_000.0)
    brain.decide([], {}, "plan")
    assert c.kwargs["budget_s"] == BUDGET_S


def test_an_unbounded_mission_passes_no_budget():
    c = _Client()
    brain = NimBrain(c)
    brain.bind(lambda: None, None, remaining=lambda: None)
    brain.decide([], {}, "plan")
    assert "budget_s" not in c.kwargs
