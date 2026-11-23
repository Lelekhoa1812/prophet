"""A read cut short by the call's own budget is not evidence about the model or the key (NIM.md 4.2). The server did not
fail; the caller ran out of time. Such a cut must not demote a model or cool a key."""
from __future__ import annotations

import dataclasses
import random

import httpx
import pytest

from prophet_agent.nim import Exhausted, NimClient
from prophet_agent.nim.providers import PROVIDERS

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


def make(handler, *, provider=None, clock=None) -> tuple[NimClient, FakeClock]:
    clock = clock or FakeClock()
    kwargs = {"provider": provider} if provider is not None else {}
    client = NimClient(list(KEYS), transport=httpx.MockTransport(handler), clock=clock, sleep=clock.sleep,
                       rng=NoShuffle(), **kwargs)
    return client, clock


def _silent_provider(seconds: float):
    return dataclasses.replace(PROVIDERS["nim"], silence_s=seconds)


def test_a_read_cut_by_the_budget_does_not_demote_a_model_that_was_not_silent():
    # The budget leaves 20 s; the model is given only that, and its read is cut at 20 s. It was not silent for its
    # 35 s silence limit, so it must not be demoted for ten minutes.
    clock = FakeClock()

    def slow(request: httpx.Request) -> httpx.Response:
        clock.t += request.extensions["timeout"]["read"]
        raise httpx.ReadTimeout("cut by the budget")

    client, _ = make(slow, provider=_silent_provider(35.0), clock=clock)
    with pytest.raises(Exhausted):
        client.complete([M1], MSGS, budget_s=20.0)
    assert M1 not in client.demoted


def test_a_read_cut_by_the_budget_does_not_cool_a_healthy_key():
    clock = FakeClock()

    def slow(request: httpx.Request) -> httpx.Response:
        clock.t += request.extensions["timeout"]["read"]
        raise httpx.ReadTimeout("cut by the budget")

    client, _ = make(slow, clock=clock)
    with pytest.raises(Exhausted):
        client.complete([M1], MSGS, budget_s=20.0)
    assert client.cool == {}
