"""Mission 9 (task 1): a researcher's claims are checked against the stored page text, deterministically.

The checks are literal: the quote must be in the cached page, the value inside the quote, a value printed more than once needs a
locator, and a changed page is unverified. Each test runs offline; the research cache is a temporary folder set by
PROPHET_RESEARCH_CACHE. A replay test runs the sub-agent once, changes the cached page, and resumes: the stored verification is read
back, the page is not read again.
"""
from __future__ import annotations

import json

import pytest

from prophet_agent.research.search import cache_key
from prophet_agent.runtime.claims import Claim, check_report_claims, parse_claims, verify_claim
from prophet_agent.runtime.goal import GoalConfig, _one_agent
from prophet_agent.runtime.runner import Runtime, Tool
from prophet_agent.runtime.store import Store

URL = "https://data.example.org/brent-2026-11-20"
PAGE = ("Market report, 20 November 2026.\n\nBrent crude settled at 78.50 dollars a barrel on Thursday.\n"
        "Dated series: 2026-11-19 closed at 77.10.")
SHA = "a" * 64


def _claim(**over) -> Claim:
    base = dict(value="78.50", unit="USD/bbl", as_of="2026-11-20", entity="Brent crude", url=URL, source_id="S1",
                quote="Brent crude settled at 78.50 dollars a barrel on Thursday.")
    return Claim(**{**base, **over})


def _cache(tmp_path, monkeypatch, url=URL, text=PAGE, sha=SHA):
    """A stored page in the research cache, in the form fetch_page writes."""
    monkeypatch.setenv("PROPHET_RESEARCH_CACHE", str(tmp_path / "cache"))
    path = tmp_path / "cache" / "pages" / f"{cache_key('page', url)}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"ok": True, "url": url, "final_url": url, "text": text, "sha256": sha}), encoding="utf-8")


# ---------------------------------------------------------------- the report's claim lines

def test_a_claim_line_is_parsed_into_its_fields_and_its_status_is_never_read():
    report = ('Findings\n- CLAIM | value=78.50 | unit=USD/bbl | as_of=2026-11-20 | entity=Brent crude | source_id=S1 | '
              'quote="Brent crude settled at 78.50 dollars a barrel" | status=verified\n'
              '**CLAIM**: value=12 | url=https://x.example/a | quote="it was 12"\n'
              "A sentence without the marker is not a claim.")
    first, second = parse_claims(report)
    assert (first.value, first.unit, first.as_of, first.entity, first.source_id) == (
        "78.50", "USD/bbl", "2026-11-20", "Brent crude", "S1")
    assert first.quote == "Brent crude settled at 78.50 dollars a barrel"
    assert first.status == "" and first.method == "report_line"
    assert (second.value, second.url, second.quote) == ("12", "https://x.example/a", "it was 12")
    assert parse_claims(None) == [] and parse_claims("no claims here") == []


def test_a_quote_may_hold_a_pipe_when_no_key_follows_it():
    (claim,) = parse_claims('CLAIM | value=3 | quote="rate 3 | fixed in 2026" | source_id=S2')
    assert claim.quote == "rate 3 | fixed in 2026" and claim.source_id == "S2"


# ---------------------------------------------------------------- verify_claim

def test_a_quote_with_its_value_in_the_stored_page_is_verified():
    got = verify_claim(_claim(), PAGE, SHA)
    assert got.status == "verified" and "quote and the value" in got.reason


def test_whitespace_and_typographic_quotes_are_normalised_before_matching():
    page = "The agency said “prices will\nease” and Brent  settled at 78.50."
    got = verify_claim(_claim(quote='The agency said "prices will ease" and Brent settled at 78.50.'), page, SHA)
    assert got.status == "verified"


def test_a_quote_the_page_does_not_hold_is_unverified():
    got = verify_claim(_claim(quote="Brent crude settled at 79.00 dollars a barrel on Thursday."), PAGE, SHA)
    assert got.status == "unverified" and "does not appear" in got.reason


def test_a_value_outside_the_quote_is_unverified_even_when_the_page_holds_it():
    got = verify_claim(_claim(value="77.10", quote="Brent crude settled at 78.50 dollars a barrel on Thursday."), PAGE, SHA)
    assert got.status == "unverified" and "not inside the quote" in got.reason


def test_a_value_is_matched_as_a_whole_number_not_inside_a_longer_one():
    page = "Brent crude settled at 178.5 dollars a barrel on Thursday."
    got = verify_claim(_claim(value="78.5", quote="Brent crude settled at 178.5 dollars a barrel on Thursday."), page, SHA)
    assert got.status == "unverified"


def test_a_value_printed_twice_on_the_page_is_ambiguous_without_a_locator():
    page = PAGE + " Brent crude settled at 78.50 dollars a barrel on Thursday."
    got = verify_claim(_claim(), page, SHA)
    assert got.status == "ambiguous" and "appears 2 times" in got.reason


def test_a_locator_next_to_the_quote_resolves_an_ambiguous_value():
    page = "Table. WTI: 78.50 on Friday.\nBrent crude settled at 78.50 dollars a barrel on Thursday."
    got = verify_claim(_claim(locator="Brent crude"), page, SHA)
    assert got.status == "verified"


def test_a_locator_that_is_not_next_to_the_quote_is_unverified():
    page = "Brent crude index 78.50.\n" + ("filler words " * 80) + "Brent crude settled at 78.50 dollars a barrel on Thursday."
    got = verify_claim(_claim(locator="Brent crude index"), page, SHA)
    assert got.status == "unverified" and "next to the quote" in got.reason


def test_a_changed_page_is_unverified_even_when_the_quote_is_still_there():
    got = verify_claim(_claim(snapshot_sha256="b" * 64), PAGE, SHA)
    assert got.status == "unverified" and "changed" in got.reason
    assert verify_claim(_claim(snapshot_sha256=SHA), PAGE, SHA).status == "verified"


def test_no_stored_text_no_quote_or_no_value_is_unverified():
    assert verify_claim(_claim(), None, None).status == "unverified"
    assert verify_claim(_claim(), "   ", None).status == "unverified"
    assert verify_claim(_claim(quote=""), PAGE, SHA).status == "unverified"
    assert verify_claim(_claim(value=""), PAGE, SHA).status == "unverified"


# ---------------------------------------------------------------- the report against the cache

def test_claims_are_checked_against_the_cached_page_and_the_source_id_is_resolved(tmp_path, monkeypatch):
    _cache(tmp_path, monkeypatch)
    evidence = [{"kind": "tool", "tool": "web_research", "result": {"sources": [
        {"id": "S1", "url": URL, "final_url": URL, "sha256": SHA, "passages": []}]}}]
    report = ('CLAIM | value=78.50 | unit=USD/bbl | source_id=S1 | quote="Brent crude settled at 78.50 dollars a barrel on Thursday."\n'
              'CLAIM | value=79.00 | source_id=S1 | quote="Brent crude settled at 79.00 dollars a barrel"\n'
              'CLAIM | value=5 | source_id=S9 | quote="five"')
    result = check_report_claims(report, evidence)
    assert [c["status"] for c in result["claims"]] == ["verified", "unverified", "unverified"]
    assert result["claims"][0]["url"] == URL and result["claims"][0]["snapshot_sha256"] == SHA
    assert "source id" in result["claims"][2]["reason"]
    assert result["summary"] == "3 claims: 1 verified, 2 unverified, 0 ambiguous"


def test_a_report_without_claims_adds_nothing(tmp_path, monkeypatch):
    _cache(tmp_path, monkeypatch)
    assert check_report_claims("The market was calm.", []) == {"claims": [], "summary": ""}


def test_a_page_read_by_the_sub_agent_that_has_since_changed_is_unverified(tmp_path, monkeypatch):
    _cache(tmp_path, monkeypatch, sha="c" * 64)
    evidence = [{"kind": "tool", "tool": "web_fetch", "result": {"url": URL, "sha256": SHA, "text": PAGE}}]
    (c,) = check_report_claims('CLAIM | value=78.50 | url=%s | quote="Brent crude settled at 78.50 dollars a barrel on Thursday."'
                               % URL, evidence)["claims"]
    assert c["status"] == "unverified" and "changed" in c["reason"]


# ---------------------------------------------------------------- the lead's view and replay

class Queue:
    def __init__(self, plans, evals):
        self.plans, self.evals = list(plans), list(evals)

    def decide(self, messages, schema, name):
        queue = self.plans if name == "plan" else self.evals
        if not queue:
            raise AssertionError(f"the brain was asked for {name} but no decision is scripted")
        return queue.pop(0)


class Silent:
    def decide(self, messages, schema, name):
        raise AssertionError(f"a resumed run asked the model for {name}")


REPORT = ("Brent settled at 78.50 on Thursday.\n"
          'CLAIM | value=78.50 | unit=USD/bbl | as_of=2026-11-20 | entity=Brent crude | source_id=S1 | '
          'quote="Brent crude settled at 78.50 dollars a barrel on Thursday."')


def _source_tool(reads: dict):
    def fn(args, key):
        reads["n"] += 1
        return {"sources": [{"id": "S1", "url": URL, "final_url": URL, "sha256": SHA, "title": "Market report",
                             "passages": [{"text": PAGE}]}]}
    return Tool("web_research", fn, idempotent=True)


@pytest.fixture()
def store(tmp_path):
    s = Store(tmp_path / "claims.sqlite")
    yield s
    s.close()


def test_the_lead_sees_each_claim_with_its_status_and_a_summary(store, tmp_path, monkeypatch):
    _cache(tmp_path, monkeypatch)
    reads = {"n": 0}
    tools = {"web_research": _source_tool(reads)}
    brain = Queue(plans=[{"action": "call_tool", "tool": "web_research", "args": {}, "rationale": "read"}, {"action": "finish", "rationale": "done"}, {"action": "finish", "rationale": "done"}],
                  evals=[{"verdict": "met", "answer": REPORT, "reason": "done", "findings": ["Brent settled at 78.50"],
                          "figures": [{"name": "Brent close", "value": "78.50"}]}])
    seen = {}
    spec = {"agent_type": "researcher", "task": "find the Brent close on 20 November 2026"}

    def program(ctx):
        seen["obs"] = _one_agent(ctx, brain, spec, 1, tools, GoalConfig(max_iterations=10, tool_budget=4), "", [])
        return {"status": "done"}

    mid = store.create_mission("find the Brent close", {"max_tool_calls": 4, "max_seconds": 900})
    Runtime(store, owner="p1").run(mid, program)
    obs = seen["obs"]
    assert obs["claims"][0]["status"] == "verified", obs
    assert obs["claim_summary"] == "1 claim: 1 verified, 0 unverified, 0 ambiguous"


def test_a_resumed_run_reuses_the_stored_verification_and_does_not_read_the_cache_again(store, tmp_path, monkeypatch):
    _cache(tmp_path, monkeypatch)
    reads = {"n": 0}
    tools = {"web_research": _source_tool(reads)}
    brain = Queue(plans=[{"action": "call_tool", "tool": "web_research", "args": {}, "rationale": "read"}, {"action": "finish", "rationale": "done"}, {"action": "finish", "rationale": "done"}],
                  evals=[{"verdict": "met", "answer": REPORT, "reason": "done"}])
    seen = {}
    spec = {"agent_type": "researcher", "task": "find the Brent close on 20 November 2026"}

    def program(brain_, record):
        def run_it(ctx):
            seen.setdefault(record, _one_agent(ctx, brain_, spec, 1, tools, GoalConfig(max_iterations=10, tool_budget=4), "", []))
            return {"status": "done"}
        return run_it

    mid = store.create_mission("find the Brent close", {"max_tool_calls": 4, "max_seconds": 900})
    Runtime(store, owner="p1").run(mid, program(brain, "first"))
    first = seen["first"]
    assert first["claims"][0]["status"] == "verified"

    # The cached page changes after the first run: the quote is gone. A resumed run must still read the stored verification.
    _cache(tmp_path, monkeypatch, text="The page was rewritten and no longer quotes Brent.", sha="d" * 64)
    Runtime(store, owner="p2").run(mid, program(Silent(), "resumed"))
    assert seen["resumed"]["claims"] == first["claims"]
    assert seen["resumed"]["claim_summary"] == first["claim_summary"]
    assert reads["n"] == 1, "the source tool ran again on resume"

    # Verifying from scratch against the changed page would have refused the claim, so the reuse is what keeps it.
    assert verify_claim(parse_claims(REPORT)[0], "The page was rewritten and no longer quotes Brent.", "d" * 64).status == "unverified"

