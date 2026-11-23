"""NimClient.check_model is the one-call model verification behind data/model_checks.json (NIM.md 8). Its contract: a
200 with a completion is "verified"; anything else is "failed" with a short redacted reason; it never raises."""
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


def raw_sse(*events: object) -> httpx.Response:
    body = "".join(f"data: {json.dumps(e)}\n\n" for e in events) + "data: [DONE]\n\n"
    return httpx.Response(200, text=body, headers={"content-type": "text/event-stream"})


def reply(text: str, finish: str = "stop") -> httpx.Response:
    return raw_sse({"choices": [{"delta": {"content": text}, "finish_reason": None}]},
                   {"choices": [{"delta": {}, "finish_reason": finish}],
                    "usage": {"prompt_tokens": 9, "completion_tokens": 64}})


def make(handler) -> NimClient:
    clock = FakeClock()
    return NimClient(list(KEYS), transport=httpx.MockTransport(handler), clock=clock, sleep=clock.sleep,
                     rng=NoShuffle())


def test_a_completion_cut_by_the_64_token_allowance_is_verified_because_the_model_answered():
    # 200 with real text: the model is reachable and answering. The allowance cut it, which is not a failure of the model.
    client = make(lambda request: reply("ok, here is a longer answer that the small allowance cuts off", finish="length"))
    out = client.check_model(M1, MSGS, max_tokens=64)
    client.close()
    assert out["status"] == "verified"
    assert out["http_status"] == 200

