"""Daily prices for one stock, ETF or index, from keyless public endpoints tried in order, each with an entity check.

Motivation vs Logic
Motivation: a forecast of a ticker needs its daily closes, and on 2026-11-26 a run saved another company's sample under the
ticker's name, because the endpoints it tried refused or answered a page. A download must say which endpoint answered,
which failed and why, and must refuse a file whose symbol is not the one requested.
Logic: endpoints are tried in order: Yahoo chart JSON (query1, then query2), Nasdaq historical JSON, Stooq CSV. An answer
must be HTTP 200 with a data body. A page is recorded with its reason and skipped; no challenge on a page is answered.
The symbol the endpoint returns (Yahoo meta.symbol, Nasdaq data.symbol) must equal the requested symbol, or that endpoint
is refused and the next one is tried. Stooq names no symbol, so its rows are marked unverified. Rows are reduced to date,
open, high, low, close, adj_close (only when the endpoint gives it) and volume; they are filtered to the range, sorted,
de-duplicated by date, and a row without a close is dropped. Every request goes through netguard.open_pinned, so the
address and redirect rules apply. The request carries a browser-like User-Agent because Yahoo requires one; that header
is not a challenge answer.
"""
from __future__ import annotations

import csv
import io
import json
import math
import re
import time
from datetime import date, datetime, timezone
from urllib.parse import quote
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import httpx

from prophet_agent.research.netguard import Resolver, UnsafeURL, open_pinned, read_capped

DEFAULT_TIMEOUT_S = 20.0
MAX_BYTES = 5 * 1024 * 1024
BROWSER_AGENT = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) "
                 "Chrome/124.0 Safari/537.36")
REQUEST_HEADERS = {"User-Agent": BROWSER_AGENT, "Accept": "application/json, text/csv;q=0.9, */*;q=0.5"}
SYMBOL = re.compile(r"\^?[A-Z0-9][A-Z0-9.\-]{0,11}")
PLAIN_SYMBOL = re.compile(r"[A-Z]{1,6}")
ISO_DAY = re.compile(r"\d{4}-\d{2}-\d{2}")
NULLS = frozenset({"", "n/a", "na", "n/d", "-", "null", "none", "nan"})
BOT_MARKERS = ("verify your browser", "requires javascript", "captcha")
STOP_REASONS = {
    "size": "the answer is larger than the size limit",
    "time": "the answer did not finish in time",
    "encoding": "the answer uses a content encoding this reader does not decode",
}


class MarketError(ValueError):
    """One endpoint's refusal or failure. The message is safe to show the planner and to store in provenance."""


def open_client(timeout_s: float) -> httpx.Client:
    """The network seam. Tests replace this function with one that returns an httpx.MockTransport client."""
    return httpx.Client(follow_redirects=False, timeout=timeout_s)


def normalise_symbol(symbol) -> str:
    if not isinstance(symbol, str) or not symbol.strip():
        raise ValueError("symbol must be a ticker such as NVDA, SPY or ^GSPC")
    text = symbol.strip().upper()
    if not SYMBOL.fullmatch(text):
        raise ValueError("symbol must be a ticker such as NVDA, SPY or ^GSPC")
    return text


def parse_range(start, end) -> tuple[date, date]:
    """The first and last day, both inclusive. Each must be YYYY-MM-DD and a real date; end is not before start."""
    for value in (start, end):
        if not isinstance(value, str) or not ISO_DAY.fullmatch(value):
            raise ValueError("start and end must be dates in YYYY-MM-DD form")
    try:
        first, last = date.fromisoformat(start), date.fromisoformat(end)
    except ValueError:
        raise ValueError("start and end must be real calendar dates") from None
    if last < first:
        raise ValueError("end is before start")
    return first, last


def _epoch(day: date) -> int:
    return int(datetime(day.year, day.month, day.day, tzinfo=timezone.utc).timestamp())


def endpoints(symbol: str, first: date, last: date) -> list[tuple[str, str | None, object]]:
    """(source, url, parser) in the order they are tried. A url of None is an endpoint that does not list this symbol."""
    quoted = quote(symbol, safe="")
    plan = []
    for host in ("query1.finance.yahoo.com", "query2.finance.yahoo.com"):
        url = (f"https://{host}/v8/finance/chart/{quoted}?period1={_epoch(first)}&period2={_epoch(last) + 86400}"
               "&interval=1d&events=div%7Csplits")
        plan.append(("yahoo_chart", url, parse_yahoo))
    plan.append(("nasdaq_historical",
                 f"https://api.nasdaq.com/api/quote/{quoted}/historical?assetclass=stocks"
                 f"&fromdate={first.isoformat()}&todate={last.isoformat()}&limit=9999", parse_nasdaq))
    stooq_url = f"https://stooq.com/q/d/l/?s={symbol.lower()}.us&i=d" if PLAIN_SYMBOL.fullmatch(symbol) else None
    plan.append(("stooq_csv", stooq_url, parse_stooq))
    # Round 20 (2026-11-27): Yahoo answered ^GSPC with HTTP 429, and Stooq now answers every request with a browser
    # challenge, so an index had no second source. FRED publishes the daily close of the main US indices without a key.
    fred = FRED_INDEX.get(symbol.upper())
    if fred:
        plan.append(("fred_close", f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={fred}&cosd={first.isoformat()}"
                                   f"&coed={last.isoformat()}", parse_fred_close))
    return plan


FRED_INDEX = {"^GSPC": "SP500", "^SPX": "SP500", "^DJI": "DJIA", "^IXIC": "NASDAQCOM", "^NDX": "NASDAQ100",
              "^VIX": "VIXCLS"}


def parse_fred_close(body: bytes, media: str, symbol: str) -> dict:
    """FRED graph CSV of one index's daily close (observation_date, <code>). Open, high, low and volume are not published
    there, so they stay empty; a '.' is a market holiday."""
    _page(body, media)
    try:
        text = body.decode("utf-8-sig")
    except UnicodeDecodeError:
        raise MarketError("the answer is not UTF-8 text") from None
    reader = csv.reader(io.StringIO(text))
    header = next(reader, None)
    code = FRED_INDEX.get(symbol.upper())
    if not header or len(header) != 2 or header[0].strip().lower() not in ("observation_date", "date") \
            or header[1].strip() != code:
        raise MarketError(f"the answer is not FRED's daily table for {code}")
    rows = []
    for record in reader:
        if len(record) != 2 or record[1].strip() in ("", "."):
            continue
        try:
            day = date.fromisoformat(record[0].strip()).isoformat()
        except ValueError:
            raise MarketError("a row date is not YYYY-MM-DD") from None
        rows.append({"date": day, "open": None, "high": None, "low": None, "close": record[1], "adj_close": None,
                     "volume": None})
    return _parsed(rows, has_adj=False, returned=code, entity=f"matched: FRED series {code} is {symbol.upper()}'s close")


def _media(value: str) -> str:
    return (value or "").split(";", 1)[0].strip().lower()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _get(url: str, *, timeout_s: float, resolver: Resolver | None) -> tuple[bytes, str, str]:
    """(body, media type, retrieved_at) for one HTTP 200 answer. Every failure is a MarketError with its reason."""
    deadline = time.monotonic() + timeout_s
    client = open_client(timeout_s)
    try:
        try:
            response = open_pinned(client, url, resolver=resolver, headers=REQUEST_HEADERS, timeout=timeout_s,
                                   deadline=deadline)
        except UnsafeURL as exc:
            raise MarketError(f"refused: {exc}") from None
        except httpx.HTTPError as exc:
            raise MarketError(f"transport error: {type(exc).__name__}") from None
        try:
            if response.status_code != 200:
                note = " (rate limited)" if response.status_code == 429 else ""
                raise MarketError(f"HTTP {response.status_code}{note}")
            body, stopped = read_capped(response, MAX_BYTES, deadline)
            if stopped:
                raise MarketError(STOP_REASONS[stopped])
            if not body:
                raise MarketError("the answer is empty")
            return body, _media(response.headers.get("content-type", "")), _now()
        finally:
            response.close()
    finally:
        client.close()


def _page(body: bytes, media: str) -> None:
    """Refuse an HTML page. A bot-check page is named as one and is not passed."""
    if body.lstrip()[:1] == b"<" or media == "text/html":
        lowered = body[:4000].decode("utf-8", "replace").lower()
        if any(marker in lowered for marker in BOT_MARKERS):
            raise MarketError("answered a bot-check page, not data; the page was not passed")
        raise MarketError("answered HTML, not data")


def _json(body: bytes):
    try:
        return json.loads(body)
    except (ValueError, RecursionError):
        raise MarketError("the answer is not JSON") from None


def _check_entity(returned, symbol: str) -> str:
    """The symbol the endpoint returned must be the one requested. Refuses with the reason, never silently."""
    name = str(returned or "").strip()
    if not name:
        raise MarketError("the answer names no symbol, so the entity cannot be checked")
    if name.upper() != symbol:
        raise MarketError(f"refused: the answer names {name!r}, not the requested {symbol!r}")
    return name


def _parsed(rows: list[dict], *, has_adj: bool, returned, entity: str = "matched", currency=None,
            exchange=None) -> dict:
    return {"rows": rows, "has_adj": has_adj, "returned_symbol": returned, "entity_check": entity,
            "currency": currency, "exchange": exchange}


def _at(values, index: int):
    if not isinstance(values, list) or index >= len(values):
        return None
    return values[index]


def _zone(name) -> ZoneInfo | timezone:
    try:
        return ZoneInfo(str(name)) if name else timezone.utc
    except (ZoneInfoNotFoundError, ValueError):
        return timezone.utc


def parse_yahoo(body: bytes, media: str, symbol: str) -> dict:
    """Yahoo chart v8. Timestamps are converted to the exchange's calendar day, so a 09:30 open stays on its own day."""
    _page(body, media)
    data = _json(body)
    chart = data.get("chart") if isinstance(data, dict) else None
    if not isinstance(chart, dict):
        raise MarketError("the answer has no chart object")
    if chart.get("error"):
        raise MarketError(f"Yahoo reported an error: {str(chart['error'])[:200]}")
    results = chart.get("result") or []
    if not results or not isinstance(results[0], dict):
        raise MarketError("the answer holds no result for the symbol")
    result = results[0]
    meta = result.get("meta") or {}
    returned = _check_entity(meta.get("symbol"), symbol)
    indicators = result.get("indicators") or {}
    quotes = (indicators.get("quote") or [{}])[0] or {}
    adj_block = (indicators.get("adjclose") or [{}])[0] or {}
    adjusted = adj_block.get("adjclose")
    zone = _zone(meta.get("exchangeTimezoneName"))
    rows = []
    for index, stamp in enumerate(result.get("timestamp") or []):
        try:
            day = datetime.fromtimestamp(int(stamp), zone).date().isoformat()
        except (TypeError, ValueError, OverflowError, OSError):
            raise MarketError("a timestamp is not a time") from None
        rows.append({"date": day, "open": _at(quotes.get("open"), index), "high": _at(quotes.get("high"), index),
                     "low": _at(quotes.get("low"), index), "close": _at(quotes.get("close"), index),
                     "adj_close": _at(adjusted, index), "volume": _at(quotes.get("volume"), index)})
    return _parsed(rows, has_adj=isinstance(adjusted, list), returned=returned, currency=meta.get("currency"),
                   exchange=meta.get("exchangeName"))


def _us_day(text) -> str:
    try:
        return datetime.strptime(str(text), "%m/%d/%Y").date().isoformat()
    except ValueError:
        raise MarketError("a row date is not MM/DD/YYYY") from None


def parse_nasdaq(body: bytes, media: str, symbol: str) -> dict:
    """Nasdaq historical JSON: newest first, dates MM/DD/YYYY, prices as text such as $123.45."""
    _page(body, media)
    data = _json(body)
    payload = data.get("data") if isinstance(data, dict) else None
    if not isinstance(payload, dict):
        status = data.get("status") if isinstance(data, dict) else None
        code = status.get("rCode") if isinstance(status, dict) else "unknown"
        raise MarketError(f"the answer holds no data (status {code})")
    returned = _check_entity(payload.get("symbol"), symbol)
    rows = []
    for record in ((payload.get("tradesTable") or {}).get("rows") or []):
        if not isinstance(record, dict):
            raise MarketError("a row is not a record")
        rows.append({"date": _us_day(record.get("date")), "open": record.get("open"), "high": record.get("high"),
                     "low": record.get("low"), "close": record.get("close"), "adj_close": None,
                     "volume": record.get("volume")})
    return _parsed(rows, has_adj=False, returned=returned)


def parse_stooq(body: bytes, media: str, symbol: str) -> dict:
    """Stooq daily CSV. The file names no symbol, so the entity is recorded as unverified, not as matched."""
    _page(body, media)
    try:
        text = body.decode("utf-8-sig")
    except UnicodeDecodeError:
        raise MarketError("the answer is not UTF-8 text") from None
    reader = csv.DictReader(io.StringIO(text))
    fields = {(name or "").strip().lower(): name for name in (reader.fieldnames or [])}
    if any(part not in fields for part in ("date", "open", "high", "low", "close")):
        raise MarketError("the answer is not a daily CSV table with Date, Open, High, Low and Close")
    volume_key = fields.get("volume")
    rows = []
    for record in reader:
        try:
            day = date.fromisoformat(str(record[fields["date"]]).strip()).isoformat()
        except ValueError:
            raise MarketError("a row date is not YYYY-MM-DD") from None
        rows.append({"date": day, "open": record[fields["open"]], "high": record[fields["high"]],
                     "low": record[fields["low"]], "close": record[fields["close"]], "adj_close": None,
                     "volume": record.get(volume_key) if volume_key else None})
    return _parsed(rows, has_adj=False, returned=None,
                   entity="unverified: the CSV names no symbol; the request named it")


def _number(value) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        number = float(value)
    else:
        text = str(value).strip().replace("$", "").replace(",", "")
        if text.lower() in NULLS:
            return None
        try:
            number = float(text)
        except ValueError:
            return None
    return number if math.isfinite(number) else None


def _table(rows: list[dict], first: str, last: str, has_adj: bool) -> list[dict]:
    """Rows in the range with a close, sorted by date, one per date (the last one given wins)."""
    by_day = {}
    for row in rows:
        day = row.get("date")
        if not day or not first <= day <= last:
            continue
        close = _number(row.get("close"))
        if close is None:
            continue
        volume = _number(row.get("volume"))
        by_day[day] = {"date": day, "open": _number(row.get("open")), "high": _number(row.get("high")),
                       "low": _number(row.get("low")), "close": close,
                       "adj_close": _number(row.get("adj_close")) if has_adj else None,
                       "volume": None if volume is None else int(round(volume))}
    return [by_day[day] for day in sorted(by_day)]


def _csv(table: list[dict], has_adj: bool) -> tuple[list[str], bytes]:
    columns = ["date", "open", "high", "low", "close"] + (["adj_close"] if has_adj else []) + ["volume"]
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(columns)
    for row in table:
        writer.writerow(["" if row.get(column) is None else row[column] for column in columns])
    return columns, buffer.getvalue().encode("utf-8")


def _parse(parser, body: bytes, media: str, symbol: str) -> dict:
    try:
        return parser(body, media, symbol)
    except (KeyError, TypeError, IndexError, AttributeError):
        raise MarketError("the answer has an unexpected shape") from None


def fetch_daily_prices(symbol, start, end, *, timeout_s: float = DEFAULT_TIMEOUT_S,
                       resolver: Resolver | None = None) -> dict:
    """Daily closes for ``symbol`` from ``start`` to ``end`` (YYYY-MM-DD, both inclusive), from the first endpoint that
    answers with a table whose symbol matches. Returns {ok: True, ...table facts and provenance} or {ok: False, failed}.
    ``failed`` lists every endpoint tried and its reason, in order."""
    symbol = normalise_symbol(symbol)
    first, last = parse_range(start, end)
    failed: list[dict] = []
    for source, url, parser in endpoints(symbol, first, last):
        if url is None:
            failed.append({"source": source, "endpoint": None, "reason": "the endpoint does not list this symbol form"})
            continue
        try:
            body, media, retrieved_at = _get(url, timeout_s=timeout_s, resolver=resolver)
            parsed = _parse(parser, body, media, symbol)
            table = _table(parsed["rows"], first.isoformat(), last.isoformat(), parsed["has_adj"])
            if not table:
                raise MarketError("no daily closes in the requested range")
        except MarketError as exc:
            failed.append({"source": source, "endpoint": url, "reason": str(exc)})
            continue
        columns, csv_bytes = _csv(table, parsed["has_adj"])
        from prophet_agent.opendata.fetch import record_download
        record_download(url, url, body, retrieved_at)      # the endpoint now has a fetch record a series can cite
        return {"ok": True, "symbol": symbol, "source": source, "endpoint": url, "retrieved_at": retrieved_at,
                "currency": parsed["currency"], "exchange": parsed["exchange"],
                "entity_check": parsed["entity_check"], "returned_symbol": parsed["returned_symbol"],
                "columns": columns, "row_count": len(table), "first_date": table[0]["date"],
                "last_date": table[-1]["date"], "csv": csv_bytes, "failed": failed}
    return {"ok": False, "symbol": symbol, "failed": failed}
