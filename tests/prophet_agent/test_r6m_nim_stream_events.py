"""Malformed but parseable stream events must fail over like unreadable JSON (NIM.md 4.2: unreadable JSON -> key cool
30 s -> next key), never crash the whole call. No network: an httpx.MockTransport scripts every answer."""
from __future__ import annotations

import json
import random

import httpx

from prophet_agent.nim import NimClient

KEYS = ("nvapi-AAAA1111", "nvapi-BBBB2222", "nvapi-CCCC3333")
M1 = "nvidia/nemotron-3-ultra-550b-a55b"
MSGS = [{"role": "system", "content": "You are terse."}, {"role": "user", "content": "Say hi."}]
TOOLS = [{"name": "record", "description": "record a fact", "parameters": {"type": "object"}}]


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
    """Each event is a JSON value (encoded) or a string (sent as written after ``data: ``)."""
    lines = [f"data: {e}" if isinstance(e, str) else f"data: {json.dumps(e)}" for e in events]
    body = "".join(f"{line}\n\n" for line in lines) + "data: [DONE]\n\n"
    return httpx.Response(200, text=body, headers={"content-type": "text/event-stream"})


def good_answer() -> httpx.Response:
    return raw_sse({"choices": [{"delta": {"content": "served"}, "finish_reason": None}]},
                   {"choices": [{"delta": {}, "finish_reason": "stop"}]})


def key_of(request: httpx.Request) -> str:
    return request.headers.get("authorization", "").removeprefix("Bearer ")


def bad_then_good(bad_event: object, *, tools: bool = False):
    """Key 1 sends the malformed event; every other key answers properly. Returns (handler, log of keys used)."""
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        key = key_of(request)
        seen.append(key)
        if key == KEYS[0]:
            return raw_sse(bad_event)
        return good_answer()

    return handler, seen


def make(handler) -> tuple[NimClient, FakeClock]:
    clock = FakeClock()
    client = NimClient(list(KEYS), transport=httpx.MockTransport(handler), clock=clock, sleep=clock.sleep,
                       rng=NoShuffle())
    return client, clock


def test_a_stream_event_that_is_not_an_object_fails_over_to_the_next_key():
    handler, seen = bad_then_good([1, 2])
    client, _ = make(handler)
    reply = client.complete([M1], MSGS)
    assert reply.text == "served"
    assert seen[:2] == [KEYS[0], KEYS[1]]


def test_a_choice_that_is_not_an_object_fails_over_to_the_next_key():
    handler, seen = bad_then_good({"choices": ["not-an-object"]})
    client, _ = make(handler)
    assert client.complete([M1], MSGS).text == "served"
    assert seen[:2] == [KEYS[0], KEYS[1]]


def test_a_delta_that_is_not_an_object_fails_over_to_the_next_key():
    handler, seen = bad_then_good({"choices": [{"delta": "text", "finish_reason": None}]})
    client, _ = make(handler)
    assert client.complete([M1], MSGS).text == "served"
    assert seen[:2] == [KEYS[0], KEYS[1]]


def test_tool_call_arguments_of_the_wrong_type_fail_over_instead_of_crashing_the_call():
    event = {"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"name": "record", "arguments": 5}}]},
                          "finish_reason": None}]}
    handler, seen = bad_then_good(event)
    client, _ = make(handler)
    reply = client.complete([M1], MSGS, tools=TOOLS)
    assert reply.key_slot == 2
    assert seen[:2] == [KEYS[0], KEYS[1]]
