"""A failed download starts the principal analyst once. The check does not name a subject."""
from prophet_agent.runtime import contracts
from prophet_agent.runtime.goal import _harness_plan


def test_a_failed_download_asks_for_a_table_claim_once():
    tools = {"state_table_claim": object(), "sandbox_python": object()}
    failed = {"kind": "tool", "tool": "download_web_data", "args": {"url": "https://example.test/x.csv"},
              "error": "HTTP 401; only a 200 response is used"}
    plan = _harness_plan("Forecast the series the question names", [failed], tools)
    assert plan["agent_type"] == "principal_analyst" and plan["auto"] == "table-claim"
    done = [failed, {"kind": "subagent", "auto": "table-claim", "agent_type": "principal_analyst"}]
    assert _harness_plan("Forecast the series the question names", done, tools) is None
    claimed = [failed, {"kind": "tool", "tool": "state_table_claim", "result": {"series_name": "alpha_daily"}}]
    assert _harness_plan("Forecast the series the question names", claimed, tools) is None


def test_html_download_asks_for_extract_on_the_stored_page_and_hides_model_items():
    history = [
        {"kind": "tool", "tool": "sandbox_python", "args": {},
         "result": {"run_id": "r1", "outputs": ["out/scan.json", "out/ledger.json"]}},
        {"kind": "tool", "tool": "web_fetch", "args": {"url": "https://example.test/table"},
         "result": {"ok": True, "final_url": "https://example.test/table"}},
        {"kind": "tool", "tool": "download_web_data", "args": {"url": "https://example.test/table"},
         "error": "the body is HTML, not a CSV or JSON table"},
        {"kind": "tool", "tool": "web_fetch", "args": {"queries": ["standings"]},
         "error": "queries must be a list of 1 to 8 search strings, each a different angle"},
    ]
    text = " ".join(f"{it['item']} {it.get('detail', '')}" for it in contracts.progress(history, run_root=None))
    assert "extract_page_tables" in text and "https://example.test/table" in text
    assert "backtest_series" not in text and "timesfm_forecast" not in text
    tools = {"state_table_claim": object(), "extract_page_tables": object(), "sandbox_python": object()}
    stuck = ("compare the forecast method with the simple baselines (backtest_series)", 16)
    plan = _harness_plan("Forecast the series the question names", history, tools, stuck=stuck)
    assert plan["agent_type"] == "principal_analyst" and plan["auto"] == "table-claim"
    assert "extract_page_tables" in plan["task"] and "https://example.test/table" in plan["task"]
    saved = history + [{"kind": "tool", "tool": "sandbox_python", "args": {},
                        "result": {"run_id": "r2", "outputs": ["out/series.json"]}}]
    later = " ".join(it["item"] for it in contracts.progress(saved, run_root=None))
    assert "backtest_series" in later
