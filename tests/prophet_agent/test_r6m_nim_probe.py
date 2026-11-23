"""Behaviour of the live NIM probe CLI (prophet_agent.nim.probe) with a mocked network.

No network, no real keys, no repo .env: every call goes to an httpx.MockTransport, the clock is a fake one, and the
keys are fixed test values. Covers what the probe prints (never key values), how it reports schema failures, how
long a model probe may wait, and the exit codes: 0 all good, 1 a probe failed, 2 not configured or a usage error.
"""
from __future__ import annotations

import json
import random

import httpx

from prophet_agent.nim import probe as probe_mod
from prophet_agent.nim import transport as transport_mod
from prophet_agent.nim.transport import NimClient

KEYS = ("nvapi-PROBETESTKEY1", "nvapi-PROBETESTKEY2")
GOOD_CLAIMS = '{"claims": [{"claim": "The warehouse ships 40 pallets a day.", "confidence": "high"}]}'


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


def sse_answer(text: str, finish: str = "stop") -> httpx.Response:
    chunks = [{"choices": [{"delta": {"content": text}, "finish_reason": None}]},
              {"choices": [{"delta": {}, "finish_reason": finish}],
               "usage": {"prompt_tokens": 5, "completion_tokens": 9}}]
    body = "".join(f"data: {json.dumps(c)}\n\n" for c in chunks) + "data: [DONE]\n\n"
    return httpx.Response(200, text=body, headers={"content-type": "text/event-stream"})


def key_of(request: httpx.Request) -> str:
    return request.headers.get("authorization", "").removeprefix("Bearer ")


def install(monkeypatch, handler, keys=KEYS) -> FakeClock:
    """Make the probe build its client from ``keys`` on the mocked transport instead of the environment."""
    clock = FakeClock()

    class FakeNim(NimClient):
        @classmethod
        def from_env(cls, **kwargs):
            return cls(list(keys), transport=httpx.MockTransport(handler), clock=clock, sleep=clock.sleep,
                       rng=NoShuffle())

    monkeypatch.setattr(probe_mod, "NimClient", FakeNim)
    return clock


def every_key_answers(status_by_key: dict[str, int], body: str = "") -> object:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status_by_key[key_of(request)], text=body)
    return handler


def rows(output: str) -> list[dict]:
    return [json.loads(line) for line in output.splitlines() if line.startswith("{")]


# -- key status: never a key value ----------------------------------------------------------------------------------

def test_keys_output_has_one_status_row_per_slot_and_never_a_key_value(monkeypatch, capsys):
    # A refused key's error body echoes the key back. The probe must not print the body, so no key value appears.
    handler = every_key_answers({KEYS[0]: 200, KEYS[1]: 403}, body=f"invalid key {KEYS[1]}")
    install(monkeypatch, handler)
    probe_mod.main(["keys"])
    out = capsys.readouterr().out
    assert "nvapi-" not in out
    for key in KEYS:
        assert key not in out
    found = rows(out)
    assert [r["slot"] for r in found] == [1, 2]
    assert [r["status"] for r in found] == [200, 403]


def test_keys_exit_zero_when_every_key_answers_200_or_is_cooling_429(monkeypatch, capsys):
    install(monkeypatch, every_key_answers({KEYS[0]: 200, KEYS[1]: 429}))
    assert probe_mod.main(["keys"]) == 0


def test_keys_exit_one_when_a_key_is_refused(monkeypatch, capsys):
    install(monkeypatch, every_key_answers({KEYS[0]: 200, KEYS[1]: 403}))
    assert probe_mod.main(["keys"]) == 1


def test_keys_exit_one_when_a_key_cannot_reach_the_service(monkeypatch, capsys):
    def unreachable(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError(f"refused for {key_of(request)}")

    install(monkeypatch, unreachable)
    assert probe_mod.main(["keys"]) == 1
    out = capsys.readouterr().out
    assert "nvapi-" not in out
    assert [r["error"] for r in rows(out)] == ["ConnectError", "ConnectError"]


# -- model probe: schema failures are reported as such ----------------------------------------------------------------

def test_models_all_verified_exit_zero_and_each_row_is_schema_valid(monkeypatch, capsys):
    install(monkeypatch, lambda request: sse_answer(GOOD_CLAIMS))
    assert probe_mod.main(["models"]) == 0
    found = rows(capsys.readouterr().out)
    assert len(found) == len(transport_mod.DEFAULT_CHAIN)
    assert all(r["ok"] and r["schema_valid"] for r in found)


def test_models_report_a_schema_failure_clearly_and_exit_one(monkeypatch, capsys):
    install(monkeypatch, lambda request: sse_answer("I cannot extract claims from that text."))
    assert probe_mod.main(["models"]) == 1
    found = rows(capsys.readouterr().out)
    assert found and all(r["ok"] is False for r in found)
    for r in found:
        assert r["error"] == "SchemaViolation"
        assert r["schema_valid"] is False
        assert "local schema check failed" in r["detail"]


def test_models_output_never_carries_a_key_value_even_when_an_error_names_one(monkeypatch, capsys):
    def leaky(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError(f"upstream echoed {key_of(request)}")

    install(monkeypatch, leaky)
    probe_mod.main(["models"])
    out = capsys.readouterr().out
    assert "nvapi-" not in out
    for key in KEYS:
        assert key not in out

