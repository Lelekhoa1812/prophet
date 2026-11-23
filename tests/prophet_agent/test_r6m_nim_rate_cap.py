"""Without a caller checkpoint nothing can stop a call that keeps waiting, so it gets one hard end: the work budget plus
RATE_WAIT_LIMIT_S (NIM.md 4.3). Pacing sleeps and rate-limit waits both count against that end (T6b)."""
from __future__ import annotations

import random

import httpx
import pytest

from prophet_agent.nim import Exhausted, NimClient
from prophet_agent.nim.transport import RATE_WAIT_LIMIT_S

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


def test_an_endless_rate_limit_without_a_checkpoint_ends_within_the_wait_limit_plus_one_budget():
    # The provider refuses every call for good. Without a checkpoint the caller has no other way to stop the call, so it
    # must end on its own: RATE_WAIT_LIMIT_S of waiting plus one work budget, however the waits are split up.
    clock = FakeClock()

    def refused(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, text="slow down", headers={"retry-after": "60"})

    client = NimClient(list(KEYS), transport=httpx.MockTransport(refused), clock=clock, sleep=clock.sleep,
                       rng=NoShuffle())
    with pytest.raises(Exhausted):
        client.complete([M1], MSGS, budget_s=240.0)
    # Pacing sleeps inside key selection are waits too; counting only the rate-limit waits let one call reach 5,460 s.
    assert clock.t - 1000.0 <= RATE_WAIT_LIMIT_S + 240.0 + 5.0
