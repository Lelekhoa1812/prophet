"""Three open questions exercise the same loop. The library does not name them."""
from prophet_agent.datasets import DatasetStore
from prophet_agent.source_table.claim import build_claim
from prophet_agent.source_table.evaluate import process_score
from prophet_agent.source_table.extract import extract_html
from prophet_agent.source_table.publish import bind_series
from prophet_agent.source_table.validate import validate_table


QUESTIONS = (
    {"question": "Melbourne crime rate forecasting", "entity": "Melbourne", "series_name": "city_rate", "measure": "rate"},
    {"question": "EPL champion in the 2026/27 season", "entity": "Champion", "series_name": "season_table", "measure": "points"},
    {"question": "RBA cash rate over the next five years", "entity": "Cash", "series_name": "policy_rate", "measure": "rate"},
)


def _page(entity: str, measure: str) -> str:
    return (
        "<html><body><table>"
        f"<tr><th>date</th><th>name</th><th>{measure}</th></tr>"
        f"<tr><td>2024-01-01</td><td>{entity}</td><td>1.5</td></tr>"
        f"<tr><td>2024-02-01</td><td>{entity}</td><td>1.7</td></tr>"
        "</table></body></html>"
    )


def _other() -> bytes:
    return b"date,ticker,name,close\n2024-01-01,BBB,Beta,9\n"


def test_each_open_question_accepts_its_own_table_and_refuses_a_substitute(tmp_path):
    for item in QUESTIONS:
        claim = build_claim({"series_name": item["series_name"], "entity": item["entity"], "measure": item["measure"]})
        page = _page(item["entity"], item["measure"])
        candidate = extract_html(page, final_url=f"https://example.test/{item['entity'].casefold()}")["candidates"][0]
        accepted = validate_table(claim, candidate["csv"].encode(), final_url=f"https://example.test/{item['entity'].casefold()}")
        assert accepted["status"] == "accepted", item["question"]
        store = DatasetStore(tmp_path / item["series_name"])
        bound = bind_series(store, claim, candidate["csv"].encode(), final_url=f"https://example.test/{item['entity'].casefold()}")
        assert bound["name"] == f"{item['series_name']}.csv"
        refused = bind_series(store, claim, _other(), final_url="https://example.test/beta.csv")
        assert refused["status"] == "wrong_entity" and refused["bound_name"] is None
        score = process_score([
            {"tool": "state_table_claim", "series_name": claim["series_name"]},
            {"tool": "validate_table", **accepted},
            {"tool": "bind_series", **bound},
            {"tool": "validate_table", **refused},
            {"tool": "bind_series", **refused},
        ])
        assert score["passed"] is True, item["question"]
        assert item["question"].split()[0] not in score["checks"].__repr__()
