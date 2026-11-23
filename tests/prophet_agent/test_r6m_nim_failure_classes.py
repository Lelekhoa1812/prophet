"""Each failure has one consequence (NIM.md 4.2). These tests pin the consequence for an error frame inside a 200 stream,
a case the table leaves to the code."""
from __future__ import annotations

import json
import random

import httpx

from prophet_agent.nim import NimClient

KEYS = ("nvapi-AAAA1111", "nvapi-BBBB2222", "nvapi-CCCC3333")
M1 = "nvidia/nemotron-3-ultra-550b-a55b"
M2 = "poolside/laguna-xs-2.1"
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


def raw_sse(*events: object) -> httpx.Response:
    body = "".join(f"data: {json.dumps(e)}\n\n" for e in events) + "data: [DONE]\n\n"
    return httpx.Response(200, text=body, headers={"content-type": "text/event-stream"})


def answer(text: str = "served") -> httpx.Response:
    return raw_sse({"choices": [{"delta": {"content": text}, "finish_reason": None}]},
                   {"choices": [{"delta": {}, "finish_reason": "stop"}]})


def key_of(request: httpx.Request) -> str:
    return request.headers.get("authorization", "").removeprefix("Bearer ")


def make(handler, *, provider=None, clock=None) -> tuple[NimClient, FakeClock]:
    clock = clock or FakeClock()
    kwargs = {"provider": provider} if provider is not None else {}
    client = NimClient(list(KEYS), transport=httpx.MockTransport(handler), clock=clock, sleep=clock.sleep,
                       rng=NoShuffle(), **kwargs)
    return client, clock


# -- an error frame inside a 200 stream names a model-scoped refusal ---------------------------------------------------

def test_model_forbidden_frame_does_not_retire_the_key_for_every_model():
    # "model_forbidden" says this one model is not open to this account. The key is fine: retiring it for ten minutes
    # would take it away from every other model too.
    def handler(request: httpx.Request) -> httpx.Response:
        if json.loads(request.content)["model"] == M1:
            return raw_sse({"error": {"code": "model_forbidden", "message": "not entitled"}})
        return answer()

    client, _ = make(handler)
    assert client.complete([M1, M2], MSGS).model == M2
    assert KEYS[0] not in client.bad


def test_model_forbidden_frame_moves_past_that_model_for_this_call():
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content)["model"])
        if seen[-1] == M1:
            return raw_sse({"error": {"code": "model_forbidden", "message": "not entitled"}})
        return answer()

    client, _ = make(handler)
    client.complete([M1, M2], MSGS)
    assert seen == [M1, M2]

