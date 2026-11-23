"""A model probe is bounded by its own budget (PROBE_BUDGET_S), not by the provider's rate-limit wait. Behaviour of the live
NIM probe CLI with a mocked network: no network, no real keys, a fake clock."""
from __future__ import annotations

import random

import httpx

from prophet_agent.nim import probe as probe_mod
from prophet_agent.nim import transport as transport_mod
from prophet_agent.nim.transport import NimClient

KEYS = ("nvapi-PROBETESTKEY1", "nvapi-PROBETESTKEY2")
PROBE_BUDGET_S = 240.0            # the per-model budget the probe passes to the transport (probe_models default)


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


def test_a_rate_limited_model_probe_stops_at_its_budget_instead_of_waiting_half_an_hour():
    clock = FakeClock()
    started = clock.t
    rate_limited = httpx.MockTransport(lambda request: httpx.Response(429, text="slow down", headers={"retry-after": "60"}))
    client = NimClient(list(KEYS), transport=rate_limited, clock=clock, sleep=clock.sleep, rng=NoShuffle())
    found = probe_mod.probe_models(client, thinking=False)
    client.close()
    assert clock.t - started <= len(transport_mod.DEFAULT_CHAIN) * PROBE_BUDGET_S + 30
    assert all(r["ok"] is False and r["error"] == "ProbeTimeout" for r in found)
