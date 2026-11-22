"""Round 3 gate B4, goal 4 (A3-02): the research item ticks only on a usable briefing.

Live evidence (measured): WTI rep4, the researcher returned answer "" with reason "research briefing is incomplete",
yet the checklist item ticked.
"""
from prophet_agent.runtime import contracts


def _research_item(history, **kw):
    return next(i for i in contracts.progress(history, run_root=None, external=True, **kw)
                if i["item"].startswith("outside context"))


def _researcher(answer, evidence=None):
    return {"kind": "subagent", "agent_id": "r1", "agent_type": "researcher", "status": "met", "answer": answer,
            "evidence": evidence or [], "files": []}



def test_an_empty_researcher_briefing_does_not_tick_the_research_item():
    # Before the fix: any researcher run ticked the item, so an empty answer passed.
    assert contracts.researched([_researcher("")]) is False
    assert _research_item([_researcher("")])["status"] == "todo"
    assert contracts.researched([_researcher("   ")]) is False


def test_a_researcher_with_a_briefing_ticks_the_research_item():
    assert contracts.researched([_researcher("OPEC+ meets on 29 November; EIA weekly stocks on Wednesdays.")]) is True
    assert _research_item([_researcher("OPEC+ meets on 29 November.")])["status"] == "done"


def test_a_researcher_whose_own_web_calls_returned_sources_counts_even_with_an_empty_answer():
    web = {"kind": "tool", "tool": "web_search", "args": {"query": "WTI regime"},
           "result": {"ok": True, "results": [{"title": "EIA weekly", "date": "2026-11-18", "text": "stocks fell"}]}}
    assert contracts.researched([_researcher("", evidence=[web])]) is True


def test_a_web_call_that_failed_does_not_tick_the_research_item():
    # An attempt that returned nothing is not a briefing. The answer must say plainly why research was not possible.
    failed = {"kind": "tool", "tool": "web_search", "args": {}, "error": "offline"}
    empty = {"kind": "tool", "tool": "web_search", "args": {}, "result": {"ok": True, "results": []}}
    assert contracts.researched([failed]) is False
    assert contracts.researched([empty]) is False


def test_an_analyst_is_not_research():
    assert contracts.researched([{"kind": "subagent", "agent_type": "analyst", "answer": "a briefing"}]) is False
