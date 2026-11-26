"""A failed download starts the principal analyst once. The check does not name a subject."""
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
