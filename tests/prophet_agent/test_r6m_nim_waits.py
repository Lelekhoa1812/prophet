"""A caller's checkpoint must be able to stop a call while it sits in a pacing sleep (NIM.md 4.3 pacing)."""
from __future__ import annotations

import json
import random

import httpx
import pytest

from prophet_agent.nim import NimClient

KEYS = ("nvapi-AAAA1111", "nvapi-BBBB2222", "nvapi-CCCC3333")
M1 = "nvidia/nemotron-3-ultra-550b-a55b"
MSGS = [{"role": "user", "content": "Say hi."}]


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


class Stop(Exception):
    pass


def reply(text: str = "ok") -> httpx.Response:
    chunk = {"choices": [{"delta": {"content": text}, "finish_reason": None}]}
    done = {"choices": [{"delta": {}, "finish_reason": "stop"}]}
    body = "".join(f"data: {json.dumps(c)}\n\n" for c in (chunk, done)) + "data: [DONE]\n\n"
    return httpx.Response(200, text=body, headers={"content-type": "text/event-stream"})


def test_a_checkpoint_stops_a_pacing_sleep_within_a_second_or_two():
    # One key at 1 request a minute: the second call must wait ~60 s for the window. The caller's checkpoint (a cancel,
    # or a spent allowance) must be consulted during that wait, not only before it.
    clock = FakeClock()
    client = NimClient(KEYS[:1], transport=httpx.MockTransport(lambda r: reply()), clock=clock, sleep=clock.sleep,
                       rng=NoShuffle(), per_minute=1)
    client.complete([M1], MSGS)

    def checkpoint():
        if clock.t >= 1010.0:
            raise Stop()

    with pytest.raises(Stop):
        client.complete([M1], MSGS, checkpoint=checkpoint)
    assert clock.t <= 1011.0

