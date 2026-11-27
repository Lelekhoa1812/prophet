"""Daily market prices: the endpoint cascade, the parsers, the entity checks, the date range and the name guard.

Offline. The network seam (market.open_client) is an httpx.MockTransport routed by the Host header, and names resolve to a
public address. The Nasdaq fixture is a trimmed copy of a live NVDA answer (measured 2026-11-26). The Yahoo bodies are
built here from the documented chart shape, because Yahoo answered 429 on that day and no live body was measured.
The live smoke test runs only with PROPHET_LIVE=1.
"""
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import httpx
import pytest

from prophet_agent.datasets import DatasetStore
from prophet_agent.opendata import fetch, market
from prophet_agent.research import netguard
from prophet_agent.tools import default_tools
from prophet_agent.tools import opendata_tools as od
from prophet_agent.tools.opendata_tools import OpenDataContext

FIXTURES = Path(__file__).parent / "fixtures"
NASDAQ_BYTES = (FIXTURES / "nasdaq_nvda_head.json").read_bytes()
PUBLIC = "93.184.216.34"
Y1, Y2, NQ, ST = "query1.finance.yahoo.com", "query2.finance.yahoo.com", "api.nasdaq.com", "stooq.com"
RATE = (429, "text/html", b"Too Many Requests")
BOT_PAGE = (200, "text/html; charset=utf-8",
            b"<!DOCTYPE html><html><body><noscript>This site requires JavaScript to verify your browser.</noscript>"
            b"</body></html>")
PLAIN_PAGE = (200, "text/html", b"<html><body>Not found</body></html>")
STOOQ_CSV = (b"Date,Open,High,Low,Close,Volume\n2024-11-26,137.7,139.3,135.67,136.92,190287700\n"
             b"2024-11-27,140,141,139,140.5,1\n")
APPLE_COLUMNS = (b"Date,AAPL.Open,AAPL.High,AAPL.Low,AAPL.Close,AAPL.Volume,AAPL.Adjusted\n"
                 b"2024-11-25,230.1,231.0,228.2,229.9,40000000,229.9\n")
APPLE_URL = "https://raw.githubusercontent.com/plotly/datasets/master/finance-charts-apple.csv"
START, END = "2024-11-26", "2026-11-25"


def _stamp(day: str, hour: int = 14, minute: int = 30) -> int:
    year, month, dom = (int(part) for part in day.split("-"))
    return int(datetime(year, month, dom, hour, minute, tzinfo=timezone.utc).timestamp())


def yahoo_body(*, stamps, closes, adj=None, symbol="NVDA", zone="America/New_York") -> bytes:
    """A chart v8 answer with open, high, low and close set to the close, so the expected rows are easy to read."""
    quote = {"open": closes, "high": closes, "low": closes, "close": closes, "volume": [1000] * len(stamps)}
    indicators = {"quote": [quote]}
    if adj is not None:
        indicators["adjclose"] = [{"adjclose": adj}]
    result = {"meta": {"symbol": symbol, "currency": "USD", "exchangeName": "NMS", "exchangeTimezoneName": zone},
              "timestamp": stamps, "indicators": indicators}
    return json.dumps({"chart": {"result": [result], "error": None}}).encode()


def nasdaq_bytes(symbol="NVDA") -> bytes:
    data = json.loads(NASDAQ_BYTES)
    data["data"]["symbol"] = symbol
    return json.dumps(data).encode()


class Net:
    """Routes by Host header. Every request's host is recorded in order, so the cascade order can be asserted."""

    def __init__(self):
        self.routes: dict[str, tuple[int, str, bytes]] = {}
        self.calls: list[str] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        host = request.headers.get("host", "").split(":")[0]
        self.calls.append(host)
        status, media, body = self.routes.get(host, (404, "text/plain", b"not found"))
        return httpx.Response(status, headers={"content-type": media}, content=body)


@pytest.fixture
def net(monkeypatch):
    state = Net()
    monkeypatch.setattr(market, "open_client", lambda timeout_s: httpx.Client(
        transport=httpx.MockTransport(state.handler), follow_redirects=False))
    monkeypatch.setattr(netguard, "_default_resolver", lambda host, port: [PUBLIC])
    return state


def _all_yahoo_limited(net, **routes):
    net.routes = {Y1: RATE, Y2: RATE, **routes}


# --- symbols, ranges and the endpoint plan ----------------------------------------------------------------------------

def test_symbols_are_upper_cased_and_a_malformed_one_is_refused():
    assert market.normalise_symbol(" nvda ") == "NVDA"
    assert market.normalise_symbol("^gspc") == "^GSPC"
    with pytest.raises(ValueError, match="ticker"):
        market.normalise_symbol("NV DA")


def test_a_reversed_or_malformed_range_is_refused():
    with pytest.raises(ValueError, match="before start"):
        market.parse_range("2024-02-01", "2024-01-01")
    with pytest.raises(ValueError, match="YYYY-MM-DD"):
        market.parse_range("2024/01/01", "2024-02-01")


def test_the_endpoint_plan_is_yahoo_query1_then_query2_then_nasdaq_then_stooq():
    plan = market.endpoints("NVDA", *market.parse_range(START, END))
    assert [source for source, _, _ in plan] == ["yahoo_chart", "yahoo_chart", "nasdaq_historical", "stooq_csv"]
    assert plan[0][1].startswith("https://query1.finance.yahoo.com/v8/finance/chart/NVDA?period1=")
    assert "events=div%7Csplits" in plan[0][1]
    assert plan[2][1].startswith("https://api.nasdaq.com/api/quote/NVDA/historical?assetclass=stocks")
    assert plan[3][1] == "https://stooq.com/q/d/l/?s=nvda.us&i=d"


def test_stooq_is_not_offered_for_an_index_form_and_that_is_recorded(net):
    _all_yahoo_limited(net, **{NQ: RATE, ST: BOT_PAGE})
    got = market.fetch_daily_prices("^GSPC", START, END)
    assert got["ok"] is False
    assert {"source": "stooq_csv", "endpoint": None,
            "reason": "the endpoint does not list this symbol form"} in got["failed"]
    assert ST not in net.calls


# --- the cascade ------------------------------------------------------------------------------------------------------

def test_cascade_tries_yahoo_then_nasdaq_and_stops_at_the_first_table(net):
    _all_yahoo_limited(net, **{NQ: (200, "application/json", NASDAQ_BYTES), ST: BOT_PAGE})
    got = market.fetch_daily_prices("NVDA", START, END)
    assert got["ok"] is True
    assert got["source"] == "nasdaq_historical"
    assert got["endpoint"].startswith("https://api.nasdaq.com/api/quote/NVDA/historical")
    assert net.calls == [Y1, Y2, NQ]
    assert [item["source"] for item in got["failed"]] == ["yahoo_chart", "yahoo_chart"]
    assert got["failed"][0]["reason"] == "HTTP 429 (rate limited)"


def test_yahoo_is_used_when_it_answers_and_nothing_else_is_asked(net):
    stamps = [_stamp("2024-11-26"), _stamp("2024-11-27")]
    net.routes = {Y1: (200, "application/json", yahoo_body(stamps=stamps, closes=[137.0, 140.0]))}
    got = market.fetch_daily_prices("NVDA", START, END)
    assert got["ok"] is True and got["source"] == "yahoo_chart"
    assert net.calls == [Y1]
    assert got["failed"] == []


def test_query2_is_tried_when_query1_is_refused(net):
    stamps = [_stamp("2024-11-26")]
    net.routes = {Y1: RATE, Y2: (200, "application/json", yahoo_body(stamps=stamps, closes=[137.0]))}
    got = market.fetch_daily_prices("NVDA", START, END)
    assert got["ok"] is True and got["endpoint"].startswith("https://query2.finance.yahoo.com/")
    assert net.calls == [Y1, Y2]


# --- parsers ----------------------------------------------------------------------------------------------------------

def test_yahoo_rows_keep_adjusted_close_sort_them_and_keep_the_last_row_for_a_repeated_day(net):
    stamps = [_stamp("2024-11-26"), _stamp("2024-11-27"), _stamp("2024-11-27"), _stamp("2024-11-29")]
    net.routes = {Y1: (200, "application/json",
                       yahoo_body(stamps=stamps, closes=[137.0, 140.0, 141.0, 142.0], adj=[136.0, 139.0, 140.0, 141.0]))}
    got = market.fetch_daily_prices("NVDA", START, END)
    assert got["columns"] == ["date", "open", "high", "low", "close", "adj_close", "volume"]
    assert got["csv"].decode().splitlines()[1:] == [
        "2024-11-26,137.0,137.0,137.0,137.0,136.0,1000",
        "2024-11-27,141.0,141.0,141.0,141.0,140.0,1000",
        "2024-11-29,142.0,142.0,142.0,142.0,141.0,1000",
    ]
    assert got["row_count"] == 3 and got["currency"] == "USD" and got["exchange"] == "NMS"


def test_a_yahoo_row_without_a_close_is_dropped_and_no_adjusted_column_is_invented(net):
    net.routes = {Y1: (200, "application/json", yahoo_body(stamps=[_stamp("2024-11-26"), _stamp("2024-11-27")],
                                                           closes=[137.0, None]))}
    got = market.fetch_daily_prices("NVDA", START, END)
    assert got["columns"] == ["date", "open", "high", "low", "close", "volume"]
    assert got["row_count"] == 1 and got["last_date"] == "2024-11-26"


def test_yahoo_days_follow_the_exchange_calendar_not_utc(net):
    # 00:30 UTC on 27 November is 19:30 on 26 November in New York, so the trading day is the 26th.
    net.routes = {Y1: (200, "application/json", yahoo_body(stamps=[_stamp("2024-11-27", 0, 30)], closes=[137.0]))}
    got = market.fetch_daily_prices("NVDA", START, END)
    assert got["first_date"] == "2024-11-26"


def test_nasdaq_rows_are_parsed_from_dollar_text_and_sorted_oldest_first(net):
    _all_yahoo_limited(net, **{NQ: (200, "application/json", NASDAQ_BYTES)})
    got = market.fetch_daily_prices("NVDA", START, END)
    lines = got["csv"].decode().splitlines()
    assert lines[0] == "date,open,high,low,close,volume"
    assert lines[1] == "2024-11-26,137.7,139.3,135.67,136.92,190287700"
    assert lines[-1] == "2026-11-25,144.06,150.37,141.17,149.3,366571900"
    assert got["row_count"] == 6 and got["currency"] is None


def test_a_stooq_csv_is_used_last_and_its_entity_is_recorded_as_unverified(net):
    _all_yahoo_limited(net, **{NQ: (404, "application/json", b'{"status":{"rCode":404}}'),
                               ST: (200, "text/csv", STOOQ_CSV)})
    got = market.fetch_daily_prices("NVDA", START, END)
    assert got["ok"] is True and got["source"] == "stooq_csv"
    assert got["entity_check"].startswith("unverified")
    assert got["returned_symbol"] is None
    assert got["row_count"] == 2
    assert got["failed"][-1]["source"] == "nasdaq_historical" and got["failed"][-1]["reason"] == "HTTP 404"


# --- Stooq and other pages --------------------------------------------------------------------------------------------

def test_stooq_html_bot_check_is_recorded_as_a_bot_check_and_not_passed(net):
    _all_yahoo_limited(net, **{NQ: BOT_PAGE, ST: BOT_PAGE})
    got = market.fetch_daily_prices("NVDA", START, END)
    assert got["ok"] is False
    assert net.calls == [Y1, Y2, NQ, ST]
    reasons = {item["source"]: item["reason"] for item in got["failed"]}
    assert reasons["stooq_csv"] == "answered a bot-check page, not data; the page was not passed"
    assert reasons["nasdaq_historical"] == "answered a bot-check page, not data; the page was not passed"


def test_a_plain_html_answer_from_stooq_is_refused_as_html(net):
    _all_yahoo_limited(net, **{NQ: RATE, ST: PLAIN_PAGE})
    got = market.fetch_daily_prices("NVDA", START, END)
    assert got["ok"] is False
    assert got["failed"][-1] == {"source": "stooq_csv", "endpoint": "https://stooq.com/q/d/l/?s=nvda.us&i=d",
                                 "reason": "answered HTML, not data"}


# --- entity checks ----------------------------------------------------------------------------------------------------

def test_a_yahoo_answer_for_another_symbol_is_refused_with_a_reason_and_the_next_endpoint_is_tried(net):
    stamps = [_stamp("2024-11-26")]
    wrong = (200, "application/json", yahoo_body(stamps=stamps, closes=[229.9], symbol="AAPL"))
    net.routes = {Y1: wrong, Y2: wrong, NQ: (200, "application/json", NASDAQ_BYTES)}
    got = market.fetch_daily_prices("NVDA", START, END)
    assert got["ok"] is True and got["source"] == "nasdaq_historical"
    assert net.calls == [Y1, Y2, NQ]
    assert got["failed"][0]["reason"] == "refused: the answer names 'AAPL', not the requested 'NVDA'"


def test_a_nasdaq_table_for_another_symbol_is_refused_and_nothing_is_returned(net):
    _all_yahoo_limited(net, **{NQ: (200, "application/json", nasdaq_bytes("AAPL")), ST: PLAIN_PAGE})
    got = market.fetch_daily_prices("NVDA", START, END)
    assert got["ok"] is False
    assert got["failed"][-2]["reason"] == "refused: the answer names 'AAPL', not the requested 'NVDA'"


def test_an_answer_that_names_no_symbol_cannot_pass_the_entity_check(net):
    _all_yahoo_limited(net, **{NQ: (200, "application/json", nasdaq_bytes("")), ST: PLAIN_PAGE})
    got = market.fetch_daily_prices("NVDA", START, END)
    assert got["ok"] is False
    assert got["failed"][-2]["reason"] == "the answer names no symbol, so the entity cannot be checked"


# --- date range -------------------------------------------------------------------------------------------------------

def test_rows_outside_the_requested_range_are_left_out(net):
    stamps = [_stamp("2024-11-26"), _stamp("2024-11-27"), _stamp("2024-11-29")]
    net.routes = {Y1: (200, "application/json", yahoo_body(stamps=stamps, closes=[137.0, 140.0, 142.0]))}
    got = market.fetch_daily_prices("NVDA", "2024-11-27", "2024-11-28")
    assert got["row_count"] == 1
    assert (got["first_date"], got["last_date"]) == ("2024-11-27", "2024-11-27")


def test_a_range_with_no_closes_is_a_failure_on_each_endpoint(net):
    _all_yahoo_limited(net, **{NQ: (200, "application/json", NASDAQ_BYTES), ST: PLAIN_PAGE})
    got = market.fetch_daily_prices("NVDA", "2015-01-01", "2015-01-31")
    assert got["ok"] is False
    assert got["failed"][2]["reason"] == "no daily closes in the requested range"


# --- the name guard (8eb4342's entity check) --------------------------------------------------------------------------

def test_an_nvda_name_over_apple_columns_is_refused_with_the_rival_ticker_named():
    reason = od.name_entity_refusal("NVDA_daily.csv", APPLE_COLUMNS, APPLE_URL)
    assert reason is not None
    assert "AAPL" in reason and "NVDA" in reason and "nothing was stored" in reason


def test_an_nvda_name_without_a_rival_ticker_is_accepted():
    assert od.name_entity_refusal("NVDA_daily.csv", b"date,close\n2024-11-26,136.92\n2024-11-27,140.5\n") is None


def test_a_name_with_no_ticker_token_is_not_guarded():
    assert od.name_entity_refusal("daily.csv", APPLE_COLUMNS) is None


@pytest.fixture
def served_csv(monkeypatch):
    def serve(body: bytes):
        def handler(request):
            return httpx.Response(200, headers={"content-type": "text/csv"}, content=body)
        monkeypatch.setattr(fetch, "open_client", lambda timeout_s: httpx.Client(
            transport=httpx.MockTransport(handler), follow_redirects=False))
    monkeypatch.setattr(netguard, "_default_resolver", lambda host, port: [PUBLIC])
    return serve


def test_download_web_data_refuses_an_nvda_name_over_apple_columns_and_stores_nothing(served_csv, tmp_path):
    served_csv(APPLE_COLUMNS)
    store = DatasetStore(tmp_path / "datasets")
    tool = default_tools(opendata=OpenDataContext(store=store, attach=lambda _id: None))["download_web_data"]
    with pytest.raises(ValueError, match="AAPL"):
        tool.fn({"url": APPLE_URL, "name": "NVDA_daily.csv"}, "k")
    assert not (tmp_path / "datasets").exists() or list((tmp_path / "datasets").iterdir()) == []


def test_download_web_data_without_a_name_still_stores_the_file_under_its_address_name(served_csv, tmp_path):
    served_csv(APPLE_COLUMNS)
    tool = default_tools(opendata=OpenDataContext(store=DatasetStore(tmp_path / "datasets"),
                                                  attach=lambda _id: None))["download_web_data"]
    out = tool.fn({"url": APPLE_URL}, "k")
    assert out["sandbox_file"] == "in/finance-charts-apple.csv"


# --- the planner tool -------------------------------------------------------------------------------------------------

def _market_tool(tmp_path, attached):
    ctx = OpenDataContext(store=DatasetStore(tmp_path / "datasets"), attach=attached.append)
    return default_tools(opendata=ctx)["download_market_prices"]


def test_download_market_prices_lands_the_file_with_its_provenance_and_a_repeat_reuses_it(net, tmp_path):
    _all_yahoo_limited(net, **{NQ: (200, "application/json", NASDAQ_BYTES)})
    attached = []
    tool = _market_tool(tmp_path, attached)
    out = tool.fn({"symbol": "NVDA", "start": START, "end": END}, "k")
    assert out["sandbox_file"] == "in/NVDA_daily.csv"
    assert out["row_count"] == 6
    assert out["columns"] == ["date", "open", "high", "low", "close", "volume"]
    assert out["first_rows"][0][0] == "2024-11-26"
    assert out["provenance"]["source"] == "market"
    assert out["provenance"]["endpoint_source"] == "nasdaq_historical"
    assert out["provenance"]["symbol"] == "NVDA"
    assert out["provenance"]["failed"][0]["reason"] == "HTTP 429 (rate limited)"
    assert attached == [out["dataset_id"]]
    calls = len(net.calls)
    again = tool.fn({"symbol": "NVDA", "start": START, "end": END}, "k")
    assert again["dataset_id"] == out["dataset_id"] and len(net.calls) == calls


def test_download_market_prices_refuses_a_name_for_another_ticker_and_stores_nothing(net, tmp_path):
    _all_yahoo_limited(net, **{NQ: (200, "application/json", NASDAQ_BYTES)})
    tool = _market_tool(tmp_path, [])
    with pytest.raises(ValueError, match="AAPL"):
        tool.fn({"symbol": "NVDA", "start": START, "end": END, "name": "AAPL_daily.csv"}, "k")
    assert not (tmp_path / "datasets").exists() or list((tmp_path / "datasets").iterdir()) == []


def test_download_market_prices_names_every_failed_endpoint_and_stores_nothing(net, tmp_path):
    _all_yahoo_limited(net, **{NQ: BOT_PAGE, ST: BOT_PAGE})
    with pytest.raises(ValueError, match="nothing was stored") as caught:
        _market_tool(tmp_path, []).fn({"symbol": "NVDA", "start": START, "end": END}, "k")
    assert "stooq_csv: answered a bot-check page" in str(caught.value)


# --- live smoke test (skipped unless PROPHET_LIVE=1) ------------------------------------------------------------------

@pytest.mark.skipif(os.environ.get("PROPHET_LIVE") != "1", reason="live endpoint smoke test: set PROPHET_LIVE=1")
def test_live_nvda_returns_a_checked_table_from_some_keyless_endpoint():
    got = market.fetch_daily_prices("NVDA", "2024-11-26", "2026-11-25")
    assert got["ok"] is True, got["failed"]
    assert got["returned_symbol"] in ("NVDA", None)
    assert got["row_count"] >= 400


def test_an_index_falls_back_to_the_fred_close_and_reads_it():
    # Round 20 (2026-11-27): Yahoo answered ^GSPC with 429 and Stooq now challenges every request.
    from datetime import date as _date
    from prophet_agent.opendata import market as mk
    plan = mk.endpoints("^GSPC", _date(2026, 11, 1), _date(2026, 11, 27))
    assert plan[-1][0] == "fred_close" and "id=SP500&cosd=2026-11-01&coed=2026-11-27" in plan[-1][1]
    assert all(name != "fred_close" for name, _, _ in mk.endpoints("NVDA", _date(2026, 11, 1), _date(2026, 11, 27)))
    body = b"observation_date,SP500\n2026-11-02,7541.40\n2026-11-03,.\n2026-11-04,7688.62\n"
    out = mk.parse_fred_close(body, "text/csv", "^GSPC")
    assert [r["date"] for r in out["rows"]] == ["2026-11-02", "2026-11-04"]
    assert out["rows"][0]["open"] is None and out["rows"][0]["close"] == "7541.40"
    import pytest as _pytest
    with _pytest.raises(mk.MarketError):
        mk.parse_fred_close(b"observation_date,DJIA\n2026-11-02,1\n", "text/csv", "^GSPC")


def test_futures_and_currency_symbols_are_accepted_and_quoted():
    """The owner's gold topic (2026-11-27): GC=F was refused as a symbol, so the gold price had no market source."""
    from datetime import date
    assert market.normalise_symbol("gc=f") == "GC=F" and market.normalise_symbol("AUDUSD=X") == "AUDUSD=X"
    for bad in ("GC=", "GC=Z", "A B", "GC=F=F"):
        with pytest.raises(ValueError):
            market.normalise_symbol(bad)
    plan = market.endpoints("GC=F", date(2025, 1, 1), date(2025, 2, 1))
    assert "/chart/GC%3DF?" in plan[0][1]
    assert dict((s, u) for s, u, _ in plan)["stooq_csv"] is None
