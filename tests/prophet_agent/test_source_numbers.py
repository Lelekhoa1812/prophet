"""Mission-wide source numbers (live AURATE, round 20261128-063203): each web_research call numbered its sources from S1, so
"S1" named a different page in every call and a claim was checked against the wrong page."""
from __future__ import annotations

from types import SimpleNamespace

from prophet_agent.runtime.claims import check_report_claims
from prophet_agent.runtime.goal import _number_sources
from prophet_agent.runtime.store import Store


def test_the_same_address_keeps_its_number_across_calls(tmp_path):
    store = Store(tmp_path / "s.sqlite")
    mid = store.create_mission("g", {})
    ctx = SimpleNamespace(store=store, mission_id=mid)
    first = {"sources": [{"id": "S1", "url": "https://www.bankofengland.co.uk/x"}, {"id": "S2", "url": "https://rba.gov.au/a"}],
             "events": [{"source_id": "S2", "event": "x"}]}
    _number_sources(ctx, first)
    second = {"sources": [{"id": "S1", "url": "https://rba.gov.au/a"}, {"id": "S2", "url": "https://asx.com.au/t"}],
              "events": [{"source_id": "S1", "event": "y"}]}
    full = {"sources": [dict(s) for s in second["sources"]], "events": [dict(e) for e in second["events"]]}
    _number_sources(ctx, second, full)
    assert [s["id"] for s in first["sources"]] == ["S1", "S2"]
    assert [s["id"] for s in second["sources"]] == ["S2", "S3"], "rba.gov.au/a is S2 wherever it appears"
    assert second["events"][0]["source_id"] == "S2"
    assert [s["id"] for s in full["sources"]] == ["S2", "S3"]
    fetched = {"url": "https://rba.gov.au/b", "text": "page"}
    _number_sources(ctx, fetched)
    assert fetched["id"] == "S4", "a fetched page gets a number so it can be cited"
    other = store.create_mission("g2", {})
    lone = {"sources": [{"id": "S1", "url": "https://rba.gov.au/a"}]}
    _number_sources(SimpleNamespace(store=store, mission_id=other), lone)
    assert lone["sources"][0]["id"] == "S1", "numbers are per mission"
    store.close()


def test_a_reused_id_resolves_to_the_page_that_holds_the_quote(monkeypatch):
    evidence = [{"kind": "tool", "result": {"sources": [{"id": "S1", "url": "https://rba.gov.au/smp", "sha256": "aaa"}]}},
                {"kind": "tool", "result": {"sources": [{"id": "S1", "url": "https://www.bankofengland.co.uk/r",
                                                         "sha256": "bbb"}]}}]
    read = []
    texts = {"https://rba.gov.au/smp": "the Board decided to increase the cash rate to 4.85 per cent",
             "https://www.bankofengland.co.uk/r": "Bank Rate is 3.75%"}
    monkeypatch.setattr("prophet_agent.runtime.claims.stored_page",
                        lambda url: read.append(url) or {"text": texts[url], "sha256": "x"})
    report = ('CLAIM | value=4.85 | unit=per cent | as_of=2026-11-03 | entity=cash rate | source_id=S1 | '
              'quote="the Board decided to increase the cash rate to 4.85 per cent"')
    out = check_report_claims(report, evidence)
    assert out["claims"], out
    assert out["claims"][0]["url"] == "https://rba.gov.au/smp", "the page that holds the quote, not the last S1"
