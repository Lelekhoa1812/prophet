"""The DNS-over-HTTPS fallback when the system resolver fails (no network: the lookups and the DoH client are stubbed)."""
import socket

import httpx
import pytest

from prophet_agent.research import netguard


class _Client:
    def __init__(self, answers, calls):
        self.answers, self.calls = answers, calls

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def get(self, url, params, headers):
        self.calls.append((url, params["type"]))
        status, body = self.answers(url, params["type"])
        return httpx.Response(status, json=body)


def _patch(monkeypatch, answers, calls):
    monkeypatch.setattr(netguard.httpx, "Client", lambda **kw: _Client(answers, calls))

    def fail(*a, **k):
        raise socket.gaierror(8, "nodename nor servname provided")
    monkeypatch.setattr(netguard.socket, "getaddrinfo", fail)


def test_failed_system_lookup_falls_back_and_answers_are_still_checked(monkeypatch):
    calls = []

    def answers(url, rtype):
        if "1.1.1.1" in url:
            return 503, {}
        return 200, {"Status": 0, "Answer": [{"type": 5, "data": "alias.example.vn."},
                                             {"type": 1, "data": "160.25.148.3"}] if rtype == "A" else []}
    _patch(monkeypatch, answers, calls)
    assert netguard._default_resolver("www.nso.gov.vn", 443) == ["160.25.148.3"]
    assert calls[0][0].startswith("https://1.1.1.1") and any("8.8.8.8" in c[0] for c in calls)

    _patch(monkeypatch, lambda url, rtype: (200, {"Status": 0, "Answer": [{"type": 1, "data": "10.0.0.5"}]}), [])
    with pytest.raises(netguard.UnsafeURL, match="private|not fetched|address"):
        netguard.resolve_checked("https://www.nso.gov.vn/x")      # a private answer is refused like any other


def test_nxdomain_and_reserved_names_are_not_asked_twice(monkeypatch):
    calls = []
    _patch(monkeypatch, lambda url, rtype: (200, {"Status": 3}), calls)
    with pytest.raises(netguard.UnsafeURL, match="did not resolve"):
        netguard._default_resolver("nothing-here.gov.vn", 443)
    assert len(calls) == 1
    calls.clear()
    with pytest.raises(netguard.UnsafeURL):
        netguard._default_resolver("host.invalid", 443)
    assert calls == []
