"""A substitute series cannot keep the requested name. No network and no model."""
from pathlib import Path

from prophet_agent.datasets import DatasetStore
from prophet_agent.source_table.claim import build_claim
from prophet_agent.source_table.publish import bind_series
from prophet_agent.source_table.validate import validate_table

CSV = b"date,ticker,name,close\n2020-01-02,BBB,Beta,10\n2020-01-03,BBB,Beta,11\n"
URL = "https://example.test/beta.csv"


def test_substitute_series_cannot_keep_the_requested_name(tmp_path):
    claim = build_claim({"series_name": "alpha_daily", "entity": "Alpha", "aliases": ["Alpha", "AAA"], "measure": "close"})
    verdict = validate_table(claim, CSV, final_url=URL)
    assert verdict["status"] == "wrong_entity"
    assert verdict["bound_name"] is None
    assert verdict["observed_entity"].casefold() in {"beta", "bbb"}
    published = bind_series(DatasetStore(tmp_path / "datasets"), claim, CSV, final_url=URL)
    assert published["dataset_id"] is None
    assert published["bound_name"] is None
    assert published.get("name") is None
    names = [path.name for path in Path(tmp_path).rglob("*") if path.is_file()]
    assert "alpha_daily.csv" not in names
    assert claim["series_name"] == "alpha_daily"


def test_an_api_address_that_names_the_ticker_is_not_a_rival():
    """Live NVDA 2026-11-26: api.nasdaq.com/api/quote/NVDA/historical was refused as naming API and QUOTE."""
    from prophet_agent.tools.opendata_tools import name_entity_refusal
    body = b"date,close\n2026-11-24,181.2\n2026-11-25,183.0\n"
    url = "https://api.nasdaq.com/api/quote/NVDA/historical?assetclass=stocks"
    assert name_entity_refusal("NVDA_daily.csv", body, url) is None
    assert name_entity_refusal("NVDA_daily.csv", body, "https://api.nasdaq.com/api/quote/AAPL/historical") is not None
