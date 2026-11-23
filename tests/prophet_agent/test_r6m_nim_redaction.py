"""Text that came from the model is data: a key-shaped string in a refused reply must not reach the logs (NIM.md 8:
any text derived from upstream is redacted before it is stored or raised)."""
from __future__ import annotations

import json
import logging
import random

import httpx

from prophet_agent.nim import NimClient, SchemaViolation

KEYS = ("nvapi-AAAA1111", "nvapi-BBBB2222")
M1 = "nvidia/nemotron-3-ultra-550b-a55b"
LEAK = "nvapi-LEAKEDVALUE12345"
SCHEMA = {"type": "object", "properties": {"confidence": {"type": "string", "enum": ["low", "medium", "high"]}},
          "required": ["confidence"]}
MSGS = [{"role": "user", "content": "Rate it."}]


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


def test_a_schema_refusal_that_quotes_a_key_shaped_value_is_not_logged_raw(caplog):
    # The model answers with a value outside the enum; jsonschema's message quotes that value back.
    clock = FakeClock()
    client = NimClient(list(KEYS), transport=httpx.MockTransport(lambda r: reply(json.dumps({"confidence": LEAK}))),
                       clock=clock, sleep=clock.sleep, rng=NoShuffle())
    caplog.set_level(logging.DEBUG, logger="prophet.nim")
    try:
        client.complete([M1], MSGS, schema=SCHEMA, schema_name="rate")
    except SchemaViolation as exc:
        assert LEAK not in str(exc)
    client.close()
    logged = "\n".join(record.getMessage() for record in caplog.records)
    assert logged, "the refusal should be logged"
    assert LEAK not in logged
