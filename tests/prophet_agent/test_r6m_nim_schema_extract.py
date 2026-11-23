"""A schema-bound reply with prose around the JSON: the JSON value that satisfies the schema is the answer.

Prose often carries bracketed references such as "[1]". The extractor must not stop at the first bracketed value
when a later JSON object is the one the schema asks for (NIM.md 3.2 and the local-check rule in 3.1)."""
from __future__ import annotations

import json
import random

import httpx

from prophet_agent.nim import NimClient

KEYS = ("nvapi-AAAA1111", "nvapi-BBBB2222")
M1 = "nvidia/nemotron-3-ultra-550b-a55b"
MSGS = [{"role": "user", "content": "Extract the claims."}]
SCHEMA = {
    "type": "object",
    "properties": {"claims": {"type": "array", "items": {
        "type": "object",
        "properties": {"claim": {"type": "string"}, "confidence": {"type": "string", "enum": ["low", "high"]}},
        "required": ["claim", "confidence"],
    }}},
    "required": ["claims"],
}
ANSWER = {"claims": [{"claim": "The warehouse ships 40 pallets a day.", "confidence": "high"}]}


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


def reply(text: str) -> httpx.Response:
    chunk = {"choices": [{"delta": {"content": text}, "finish_reason": None}]}
    done = {"choices": [{"delta": {}, "finish_reason": "stop"}]}
    body = "".join(f"data: {json.dumps(c)}\n\n" for c in (chunk, done)) + "data: [DONE]\n\n"
    return httpx.Response(200, text=body, headers={"content-type": "text/event-stream"})


def test_a_bracketed_reference_before_the_json_does_not_hide_the_answer():
    text = "Claims from source [1], as requested:\n" + json.dumps(ANSWER)
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return reply(text) if len(calls) == 1 else reply("ok")

    clock = FakeClock()
    client = NimClient(list(KEYS), transport=httpx.MockTransport(handler), clock=clock, sleep=clock.sleep,
                       rng=NoShuffle())
    out = client.complete([M1], MSGS, schema=SCHEMA, schema_name="claims")
    assert json.loads(out.text) == ANSWER
    assert len(calls) == 1, "the answer was valid, so no repair turn should have been sent"
