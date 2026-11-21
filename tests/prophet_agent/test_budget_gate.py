"""Token budgets are enforced before a call: a model over its 1d/7d/30d budget is skipped, and a call with no model left
stops with Exhausted. The Settings data size is cached for a short time. No network: a MockTransport answers every call.
Run with: PYTHONPATH=src .venv/bin/python -m pytest tests/prophet_agent/test_budget_gate.py -q
"""
from __future__ import annotations

import json

import httpx
import pytest

from prophet_agent import settings as settings_mod
from prophet_agent import usage as usage_mod
from prophet_agent.nim import Exhausted, NimClient
from prophet_agent.nim import transport as transport_mod

KEYS = ("nvapi-AAAA1111", "nvapi-BBBB2222")
M1 = "nvidia/nemotron-3-ultra-550b-a55b"
M2 = "poolside/laguna-xs-2.1"


def _sse(text: str) -> httpx.Response:
    chunks = [{"choices": [{"delta": {"content": text}, "finish_reason": None}]},
              {"choices": [{"delta": {}, "finish_reason": "stop"}],
               "usage": {"prompt_tokens": 5, "completion_tokens": 2}}]
    body = "".join(f"data: {json.dumps(c)}\n\n" for c in chunks) + "data: [DONE]\n\n"
    return httpx.Response(200, text=body, headers={"content-type": "text/event-stream"})


class _Recorder:
    def __init__(self) -> None:
        self.models: list[str] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.models.append(json.loads(request.content)["model"])
        return _sse("ok")


def _client(recorder: _Recorder, allow) -> NimClient:
    return NimClient(list(KEYS), transport=httpx.MockTransport(recorder), sleep=lambda _s: None, allow=allow)


def test_a_model_over_budget_is_skipped_and_the_next_model_answers():
    recorder = _Recorder()
    client = _client(recorder, allow=lambda m: f"{m} over budget" if m == M1 else None)
    reply = client.complete([M1, M2], [{"role": "user", "content": "hi"}])
    assert reply.text == "ok"
    assert recorder.models == [M2]


def test_no_model_left_stops_the_call_before_any_request():
    recorder = _Recorder()
    client = _client(recorder, allow=lambda m: f"{m} over budget")
    with pytest.raises(Exhausted, match="token budget used up"):
        client.complete([M1, M2], [{"role": "user", "content": "hi"}])
    assert recorder.models == []


def test_without_a_gate_every_model_is_eligible():
    recorder = _Recorder()
    client = _client(recorder, allow=None)
    client.complete([M1], [{"role": "user", "content": "hi"}])
    assert recorder.models == [M1]


class _FakeLedger:
    clock = staticmethod(lambda: 1_000_000.0)

    def __init__(self, used: dict[str, int]) -> None:
        self._used = used

    def used(self, model: str, window: str, now: float | None = None) -> int:
        return self._used.get(window, 0)


def test_gate_reports_the_window_that_is_used_up(monkeypatch):
    budgets = {M1: {"1d": 1_000, "7d": None, "30d": 10_000}, M2: {"1d": None, "7d": None, "30d": None}}
    monkeypatch.setattr(settings_mod.Settings, "budgets", lambda self: budgets)
    monkeypatch.setattr(usage_mod, "ledger_for", lambda _root: _FakeLedger({"1d": 1_500, "30d": 2_000}))
    gate = transport_mod._budget_gate()
    reason = gate(M1)
    assert reason is not None and "1d 1,500/1,000 tokens" in reason and "30d" not in reason
    assert gate(M2) is None


def test_gate_passes_a_model_with_no_budget_or_budget_not_reached(monkeypatch):
    budgets = {M1: {"1d": 1_000, "7d": None, "30d": None}}
    monkeypatch.setattr(settings_mod.Settings, "budgets", lambda self: budgets)
    monkeypatch.setattr(usage_mod, "ledger_for", lambda _root: _FakeLedger({"1d": 999}))
    assert transport_mod._budget_gate()(M1) is None


def test_data_size_is_cached_until_the_ttl_or_a_clear(tmp_path, monkeypatch):
    root = tmp_path / "data"
    root.mkdir()
    (root / "a.bin").write_bytes(b"x" * 10)
    clock = {"now": 100.0}
    monkeypatch.setattr(settings_mod.time, "monotonic", lambda: clock["now"])
    settings_mod._forget_data_bytes(root)

    assert settings_mod._data_bytes(root) == 10
    (root / "b.bin").write_bytes(b"y" * 20)
    clock["now"] += settings_mod.SIZE_TTL_S / 2
    assert settings_mod._data_bytes(root) == 10          # still inside the TTL: the walk is skipped

    clock["now"] += settings_mod.SIZE_TTL_S
    assert settings_mod._data_bytes(root) == 30          # TTL passed: recomputed

    (root / "c.bin").write_bytes(b"z" * 5)
    settings_mod._forget_data_bytes(root)                # what a clear does
    assert settings_mod._data_bytes(root) == 35
