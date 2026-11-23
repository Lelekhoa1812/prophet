"""check_model reports a think-only reply as a failed check instead of raising, so verify_models can store one row per
model and reach the next one (NIM.md 8)."""
from __future__ import annotations

import json
import random

import httpx

from prophet_agent.nim import NimClient

KEYS = ("nvapi-AAAA1111", "nvapi-BBBB2222")
M1 = "poolside/laguna-xs-2.1"
MSGS = [{"role": "user", "content": "Reply with the single word: ok"}]


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


def reply(text: str, finish: str = "stop") -> httpx.Response:
    events = [{"choices": [{"delta": {"content": text}, "finish_reason": None}]},
              {"choices": [{"delta": {}, "finish_reason": finish}]}]
    body = "".join(f"data: {json.dumps(e)}\n\n" for e in events) + "data: [DONE]\n\n"
    return httpx.Response(200, text=body, headers={"content-type": "text/event-stream"})


def test_a_think_only_reply_is_reported_failed_and_never_raises():
    clock = FakeClock()
    client = NimClient(list(KEYS), transport=httpx.MockTransport(lambda r: reply("<think>working it out</think>")),
                       clock=clock, sleep=clock.sleep, rng=NoShuffle())
    out = client.check_model(M1, MSGS, max_tokens=64)
    client.close()
    assert out["status"] == "failed"
    assert out["error"]
