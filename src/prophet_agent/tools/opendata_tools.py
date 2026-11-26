"""Planner tools for open data: list the no-key sources, download one into the mission's files, or download one CSV or JSON
file from an https address. Every stored download keeps its provenance and joins the mission's files, so sandbox_python
reads it at in/<name> in the same run.

Motivation vs Logic
Motivation: a forecast often needs a public series the person did not upload. The planner should fetch it itself, but only
through the checked sources and the guarded download, and the file must be usable by the analysis code straight away.
Logic: list and source downloads go through sources.py and opendata.store, which store the bytes with their provenance and
return the same dataset for the same source and parameters. A web download goes through fetch.download (netguard, size,
type and redirect rules) and is refused unless the body parses as a CSV, JSON or plain-text numeric table. A plain-text
table (space-separated columns, '#' comment lines) is stored whole, so a file larger than the research text cap still
reaches the sandbox. Both paths record the new dataset
on the mission (OpenDataContext.attach) so a later build of the mission's tools mounts it, and LiveMounts lets the sandbox
read it in the current run, since the sandbox's inputs are otherwise fixed when the tools are built.
"""
from __future__ import annotations

import csv
import hashlib
import re
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlsplit

from prophet_agent.datasets import DatasetStore, clean_name
from prophet_agent.opendata import containers, fetch, html_landing, market, pdf_landing, sources
from prophet_agent.opendata.csvlines import table_lines
from prophet_agent.opendata.store import download_to_dataset, preview
from prophet_agent.runtime.runner import ArgumentRejected, Tool
from prophet_agent.source_table.validate import GENERIC, validate_table
from prophet_agent.tools.sandbox_tool import SANDBOX_PYTHON_SPEC, make_run, sandbox_python_tool

MAX_MISSION_FILES = 10          # the most files one mission may hold, uploaded or downloaded (the request allows ten)
UNVERIFIED = "unverified"       # licence_key for a web download whose source has no recorded key
KNOWN_LICENCE_KEYS = frozenset(source.licence_key for source in sources.SOURCES.values())
WEB_NAME_FALLBACK = "web-data"


@dataclass(frozen=True)
class OpenDataContext:
    """What the download tools need from one mission: its dataset store (the root the sandbox mounts from) and a way to
    record a new dataset on the mission. A listing built without a mission has store None and the downloads refuse."""
    store: DatasetStore | None
    attach: Callable[[str], None] | None


class LiveMounts:
    """The files the sandbox mounts for a mission: the ones it started with, then each download as it lands. The sandbox
    reads this at every call, so a file downloaded earlier in the run is an input of the next sandbox_python call."""

    def __init__(self, start: dict[str, Path] | None):
        self._mounts: dict[str, Path] = dict(start or {})

    def mounts(self) -> dict[str, Path]:
        return dict(self._mounts)

    def __len__(self) -> int:
        return len(self._mounts)

    def name_of(self, path: Path) -> str | None:
        return next((name for name, mounted in self._mounts.items() if mounted == path), None)

    def add(self, name: str, path: Path) -> str:
        """Mount path under name, with a numeric suffix when the name is taken (as DatasetStore.mount does)."""
        stem, suffix, mount, n = Path(name).stem, Path(name).suffix, name, 1
        while mount in self._mounts:
            n += 1
            mount = f"{stem}_{n}{suffix}"
        self._mounts[mount] = path
        return mount


def live_sandbox_tool(live: LiveMounts) -> Tool:
    """sandbox_python, reading the mission's live files at each call instead of the set fixed when the tools were built."""
    return replace(sandbox_python_tool(), fn=lambda args, key: make_run(live.mounts())(args, key))


def _arguments(args: Any, allowed: set[str], required: set[str]) -> dict:
    if not isinstance(args, dict):
        raise ArgumentRejected("the arguments must be an object")
    extra = sorted(set(args) - allowed)
    if extra:
        raise ArgumentRejected(f"unknown arguments {extra}; allowed: {', '.join(sorted(allowed)) or 'none'}")
    missing = sorted(required - set(args))
    if missing:
        raise ArgumentRejected(f"missing arguments {missing}")
    return args


def _store(ctx: OpenDataContext) -> DatasetStore:
    if ctx.store is None:
        raise ValueError("open-data downloads are not available outside a mission")
    return ctx.store


def _room(live: LiveMounts) -> None:
    if len(live) >= MAX_MISSION_FILES:
        raise ValueError(f"this mission already holds {MAX_MISSION_FILES} files; sandbox_python can read them, "
                         "but no more can be added")


def _mount(ctx: OpenDataContext, live: LiveMounts, meta: dict) -> str:
    """Record a stored dataset on the mission and mount it; returns the name it is mounted under."""
    path = _store(ctx).path(meta["id"])
    mounted = live.name_of(path)
    if mounted is not None:
        return mounted
    if ctx.attach is not None:
        ctx.attach(meta["id"])      # first, so a dataset that cannot be recorded is not mounted for this run only
    return live.add(meta["name"], path)


def _result(meta: dict, mount: str, extra: dict) -> dict:
    table = meta["preview"]
    result = {**extra, "dataset_id": meta["id"], "name": meta["name"], "sandbox_file": f"in/{mount}",
              "row_count": table["row_count"], "columns": table["columns"], "first_rows": table["rows"]}
    if table.get("last_row"):
        result["last_row"] = table["last_row"]
    return result


def list_sources_tool() -> Tool:
    def run(args: dict, _key: str) -> dict:
        _arguments(args, set(), set())
        return {"sources": sources.describe()}
    return Tool("list_open_data_sources", run, idempotent=True)


def download_open_data_tool(ctx: OpenDataContext, live: LiveMounts) -> Tool:
    def run(args: dict, _key: str) -> dict:
        args = _arguments(args, {"source", "params"}, {"source"})
        source = args["source"]
        params = args.get("params", {})
        if not isinstance(params, dict):
            raise ArgumentRejected("params must be an object of parameter names to values")
        store = _store(ctx)
        sources.get(source)         # an unknown key is refused before anything else is checked
        _room(live)
        meta = download_to_dataset(source, params, store)
        mount = _mount(ctx, live, meta)
        return _result(meta, mount, {"source": source})
    return Tool("download_open_data", run, idempotent=True)


def _as_table(fetched: dict) -> tuple[bytes, str]:
    """CSV or JSON bytes. A ZIP or a workbook becomes one CSV. The original content type is kept on the record."""
    media = fetch.media_type(fetched["content_type"])
    body = fetched["bytes"]
    if media in fetch.CONTAINER_TYPES:
        kind = "xlsx" if media.endswith("sheet") else "zip"
        try:
            body = containers.to_csv(body, kind)
        except containers.ContainerError as exc:
            raise ValueError(str(exc)) from None
        media = "text/csv"
    return body, media


def _web_format(media: str, body: bytes) -> str:
    head = body.lstrip()[:1]
    if head == b"<":
        raise ValueError("the body is HTML, not a CSV or JSON table")
    return "json" if media == "application/json" or head in (b"[", b"{") else "csv"


def _web_table(body: bytes, fmt: str) -> tuple[dict, str]:
    """The preview and the format to store: json, csv, or text (whitespace columns).

    A text/plain body with no commas is one column to the CSV reader. When that happens the whitespace reader counts the
    rows, and the original bytes are what the sandbox opens."""
    if fmt == "json":
        try:
            table = preview(body, "json")
        except ValueError as exc:
            raise ValueError(f"the file is not a CSV or JSON table: {exc}") from None
        if table["row_count"] < 1:
            raise ValueError("the file has no data rows")
        return table, "json"
    table = None
    try:
        table = preview(body, "csv")
    except ValueError:
        table = None
    if table is not None and len(table["columns"]) >= 2 and table["row_count"] >= 1:
        return table, "csv"
    try:
        table = preview(body, "text")
    except ValueError as exc:
        raise ValueError(f"the file is not a CSV, JSON or plain-text table: {exc}") from None
    if len(table["columns"]) < 2:
        raise ValueError("the file is not a table: a table needs at least two columns")
    if table["row_count"] < 1:
        raise ValueError("the file has no data rows")
    return table, "text"


def _web_name(url: str, name: str | None, fmt: str) -> str:
    base = name if name is not None else unquote(urlsplit(url).path.rsplit("/", 1)[-1]) or WEB_NAME_FALLBACK
    base = clean_name(base)
    suffix = {"json": ".json", "csv": ".csv", "text": ".txt"}[fmt]
    return base if base.lower().endswith(suffix) else f"{base}{suffix}"


def _existing_web(store: DatasetStore, url: str, table_request: str | None = None) -> dict | None:
    if not store.root.is_dir():
        return None
    matches = []
    for folder in store.root.iterdir():
        try:
            meta = store.get(folder.name)
        except KeyError:
            continue
        provenance = meta.get("provenance") or {}
        if (provenance.get("source") == "web" and provenance.get("url") == url
                and provenance.get("table_request") == table_request):
            matches.append(meta)
    return max(matches, key=lambda meta: meta["created"]) if matches else None


TICKER_TOKEN = re.compile(r"[A-Z]{2,5}")
PREFIX_TICKER = re.compile(r"[A-Za-z]{2,5}")


def _name_ticker(name: str | None) -> str | None:
    """The first all-capitals token of 2 to 5 letters in a file name that is not a generic word: NVDA in NVDA_daily.csv."""
    for token in re.findall(r"[A-Za-z0-9]+", name or ""):
        if TICKER_TOKEN.fullmatch(token) and token.lower() not in GENERIC:
            return token
    return None


def _dotted_column_tickers(body: bytes) -> set[str]:
    """Tickers that lead a column header such as AAPL.Close. validate_table reads only bare ticker headers, so these are
    read here, with the same comment and blank-line rules."""
    try:
        lines = [line for line in table_lines(body.decode("utf-8-sig")) if line.strip()]
        header = next(csv.reader(lines[:1]), [])
    except (UnicodeDecodeError, csv.Error):
        return set()
    tickers = set()
    for cell in header:
        head, dot, _ = cell.strip().partition(".")
        if dot and PREFIX_TICKER.fullmatch(head):
            tickers.add(head.upper())
    return tickers


def name_entity_refusal(name: str | None, body: bytes, final_url: str = "") -> str | None:
    """Why the bytes contradict the ticker in the name, or None.

    Motivation vs Logic
    Motivation: a file saved as NVDA_daily.csv held Apple's columns on 2026-11-26 and the planner used it for a forecast.
    Logic: the name's ticker is checked against the file's own tickers. Column headers such as AAPL.Close are read first,
    then validate_table (commit 8eb4342) reads the entity cells, comments and address. Only a rival ticker refuses; a file
    that names no entity is not refused, as validate_table decides."""
    ticker = _name_ticker(name)
    if ticker is None:
        return None
    rivals = sorted(_dotted_column_tickers(body) - {ticker})
    if not rivals:
        verdict = validate_table({"claim_id": "download", "entity": ticker, "aliases": [ticker], "series_name": name},
                                 body, final_url=final_url)
        if verdict["status"] != "wrong_entity":
            return None
        rivals = sorted(item.upper() for item in (verdict["rivals"] or [verdict["observed_entity"]]))
    return (f"the name {name!r} names {ticker}, but the file names {', '.join(rivals)}; refused, nothing was stored")


def _guard_name(name, body: bytes, url: str) -> None:
    if isinstance(name, str) and name.strip():
        reason = name_entity_refusal(name, body, url)
        if reason:
            raise ValueError(reason)


def download_web_data_tool(ctx: OpenDataContext, live: LiveMounts) -> Tool:
    def run(args: dict, _key: str) -> dict:
        args = _arguments(args, {"url", "name", "licence_key", "table"}, {"url"})
        url = args["url"]
        if not isinstance(url, str) or not url.strip():
            raise ArgumentRejected("url must be the https address of a CSV, JSON, HTML or PDF file")
        url = url.strip()
        if urlsplit(url).scheme.lower() != "https":
            raise ValueError("url must start with https://")
        name = args.get("name")
        if name is not None and not isinstance(name, str):
            raise ArgumentRejected("name must be text")
        table = args.get("table")
        if table is not None and (isinstance(table, bool) or not isinstance(table, (int, str))):
            raise ArgumentRejected("table must be a candidate index or a locator such as table[2] or __NEXT_DATA__:$.props")
        key = args.get("licence_key") or UNVERIFIED
        if not isinstance(key, str):
            raise ValueError(f"licence_key must be text: {UNVERIFIED} or one of: {', '.join(sorted(KNOWN_LICENCE_KEYS))}")
        if key not in KNOWN_LICENCE_KEYS:
            # Panel CPI (2026-11-26) was refused twice for a FRED CSV because it named a key not in the list. The key is a
            # provenance label, so a key with no record is stored as unverified and the download goes ahead.
            key = UNVERIFIED
        store = _store(ctx)
        _room(live)
        table_request = None if table is None else str(table)
        existing = _existing_web(store, url, table_request)
        if existing is not None:
            body = store.path(existing["id"]).read_bytes()
            _guard_name(name, body, url)
            meta = dict(existing, preview=preview(body, existing["provenance"]["format"]))
        else:
            fetched = fetch.download(url)          # refuses private addresses, redirects to them, and types or sizes it does not accept
            media = fetch.media_type(fetched["content_type"])
            if media in fetch.HTML_TYPES or media in fetch.PDF_TYPES:
                # A page or a PDF lands its best table (or the one named by table), with the other candidates listed on the
                # result. The name guard applies to the landed CSV exactly as it does to any other download.
                land = pdf_landing.land_pdf if media in fetch.PDF_TYPES else html_landing.land_html
                meta, _ = land(fetched, store=store, name=_web_name(url, None, "csv"), url=url,
                               licence_key=key, table=table)
                reason = (name_entity_refusal(name, store.path(meta["id"]).read_bytes(), url)
                          if isinstance(name, str) and name.strip() else None)
                if reason:
                    raise ValueError(reason.replace("nothing was stored", "the table was not mounted"))
            else:
                body, media = _as_table(fetched)
                fmt = _web_format(media, body)
                table_info, fmt = _web_table(body, fmt)
                _guard_name(name, body, url)
                provenance = {"source": "web", "format": fmt, "url": url, "final_url": fetched["final_url"],
                              "content_type": fetched["content_type"], "retrieved_at": fetched["retrieved_at"],
                              "sha256": hashlib.sha256(body).hexdigest(), "bytes": len(body), "licence_key": key,
                              "series_bound": False, "table_request": None}
                # The caller's name is not the series. The address supplies the working file name.
                meta = dict(store.add(_web_name(url, None, fmt), body, provenance=provenance), preview=table_info)
        mount = _mount(ctx, live, meta)
        provenance = meta["provenance"]
        extra = {"source": "web", "url": provenance["url"], "final_url": provenance["final_url"],
                 "licence_key": provenance["licence_key"], "series_bound": False}
        if provenance.get("html") or provenance.get("pdf"):
            extra["extraction"] = provenance["extraction"]
            extra["other_tables"] = provenance["extraction"]["other_tables"]
        if isinstance(name, str) and name.strip():
            extra["caller_name_ignored"] = True
        return _result(meta, mount, extra)
    return Tool("download_web_data", run, idempotent=True)


LIST_SOURCES_SPEC = {
    "name": "list_open_data_sources",
    "title": "Open data sources",
    "activity": "Listing open data sources",
    "description": "List the open-data sources that can be downloaded without a key, with the parameters each one takes. "
                   "Read-only.",
    "args": {"none": "takes no arguments; send an empty object"},
}

DOWNLOAD_OPEN_DATA_SPEC = {
    "name": "download_open_data",
    "title": "Open data download",
    "activity": "Downloading open data",
    "description": "Download one source from list_open_data_sources into the mission's files. The result gives the dataset "
                   "id, the first rows and sandbox_file, the path sandbox_python reads it at (in/<name>) from its next call "
                   "on. The same source and parameters return the stored copy without downloading again.",
    "args": {"source": "a source key from list_open_data_sources",
             "params": "object of that source's parameters, named as list_open_data_sources lists them"},
}

DOWNLOAD_WEB_DATA_SPEC = {
    "name": "download_web_data",
    "title": "Web data download",
    "activity": "Downloading a web file",
    "description": "Download one CSV, JSON, plain-text table, ZIP, workbook, HTML page or PDF from an https address into the "
                   "mission's files. An HTML page lands its best data table (from its markup or its embedded JSON, such as "
                   "__NEXT_DATA__) as CSV, with provenance; a PDF lands its best table (ruled tables first, then aligned "
                   "text columns) as CSV, with its page. The result lists the other tables by index, header and row "
                   "count, and table picks another (a whole number, or a locator such as page3.table1 for a PDF). A page or "
                   "PDF with no data table is refused: call web_fetch, then extract_page_tables. The stored name comes "
                   "from the address. A name you pass is ignored, and series_bound stays false until bind_series accepts "
                   "the file. The full file is mounted at sandbox_file (in/<name>) for sandbox_python. Use this, not "
                   "web_fetch, when the URL is the data file or a page or PDF with a table. Its licence_key is unverified "
                   "unless a known key is given.",
    "args": {"url": "https address of a CSV, JSON, plain-text, ZIP, workbook, HTML or PDF file",
             "name": "optional; ignored. The stored name comes from the address",
             "licence_key": f"optional known key ({', '.join(sorted(KNOWN_LICENCE_KEYS))}); leave out when unknown",
             "table": "optional, for an HTML page or PDF: the index of a candidate (a whole number) or its locator, such as "
                      "table[2], __NEXT_DATA__:$.props.pageProps.standings or page3.table1; leave out for the best table"},
}


MARKET_PRICES_SPEC = {
    "name": "download_market_prices",
    "title": "Daily market prices",
    "activity": "Downloading daily prices",
    "description": "Download daily open, high, low, close and volume for one stock, ETF or index between two dates. The "
                   "endpoints are tried in order (Yahoo, Nasdaq, Stooq). The first table whose symbol matches the request "
                   "is stored in the mission's files. The result names the endpoint that answered and each endpoint that "
                   "failed, with its reason. The file is mounted at sandbox_file (in/<name>) for sandbox_python. Pages and "
                   "bot checks are not passed. A name for another ticker is refused.",
    "args": {"symbol": "ticker such as NVDA, SPY or ^GSPC",
             "start": "first day, YYYY-MM-DD",
             "end": "last day, YYYY-MM-DD",
             "name": "optional file name; refused when it names another ticker. Left out, the file is SYMBOL_daily.csv"},
}


def _market_file(name, symbol: str) -> str:
    if isinstance(name, str) and name.strip():
        base = clean_name(name)
    else:
        base = f"{re.sub(r'[^A-Za-z0-9]', '', symbol)}_daily.csv"
    return base if base.lower().endswith(".csv") else f"{base}.csv"


def _market_refusal(symbol: str, failed: list[dict]) -> str:
    reasons = "; ".join(f"{item['source']}: {item['reason']}" for item in failed) or "no endpoint was tried"
    return f"no keyless endpoint returned a matching table for {symbol} ({reasons}); nothing was stored"


def _existing_market(store: DatasetStore, symbol: str, start: str, end: str) -> dict | None:
    if not store.root.is_dir():
        return None
    matches = []
    for folder in store.root.iterdir():
        try:
            meta = store.get(folder.name)
        except KeyError:
            continue
        provenance = meta.get("provenance") or {}
        if (provenance.get("source") == "market" and provenance.get("symbol") == symbol
                and provenance.get("start") == start and provenance.get("end") == end):
            matches.append(meta)
    return max(matches, key=lambda meta: meta["created"]) if matches else None


def download_market_prices_tool(ctx: OpenDataContext, live: LiveMounts) -> Tool:
    """Daily prices for one symbol through market.fetch_daily_prices, stored with the endpoint that answered. The name is
    checked against the symbol before anything is stored, so a file cannot carry another ticker's name."""
    def run(args: dict, _key: str) -> dict:
        args = _arguments(args, {"symbol", "start", "end", "name"}, {"symbol", "start", "end"})
        symbol = market.normalise_symbol(args["symbol"])
        first, last = market.parse_range(args["start"], args["end"])
        start, end = first.isoformat(), last.isoformat()
        name = args.get("name")
        if name is not None and not isinstance(name, str):
            raise ArgumentRejected("name must be text")
        store = _store(ctx)
        _room(live)
        existing = _existing_market(store, symbol, start, end)
        if existing is not None:
            meta = existing
            body = store.path(existing["id"]).read_bytes()
            endpoint = existing["provenance"]["endpoint"]
        else:
            got = market.fetch_daily_prices(symbol, start, end)
            if not got["ok"]:
                raise ValueError(_market_refusal(symbol, got["failed"]))
            body, endpoint = got["csv"], got["endpoint"]
            meta = None
            provenance = {"source": "market", "endpoint_source": got["source"], "endpoint": endpoint, "symbol": symbol,
                          "start": start, "end": end, "retrieved_at": got["retrieved_at"],
                          "returned_symbol": got["returned_symbol"], "entity_check": got["entity_check"],
                          "currency": got["currency"], "exchange": got["exchange"], "failed": got["failed"],
                          "row_count": got["row_count"], "first_date": got["first_date"],
                          "last_date": got["last_date"], "format": "csv",
                          "sha256": hashlib.sha256(body).hexdigest(), "bytes": len(body),
                          "series_bound": False}
        if isinstance(name, str) and name.strip():
            claimed = _name_ticker(name)
            if claimed is not None and claimed != symbol.lstrip("^"):
                raise ValueError(f"the name {name!r} names {claimed}, but the download is {symbol}; refused, nothing "
                                 "was stored")
        _guard_name(name, body, endpoint)
        if meta is None:
            meta = store.add(_market_file(name, symbol), body, provenance=provenance)
        provenance = meta["provenance"]
        meta = dict(meta, preview=preview(body, "csv"))
        mount = _mount(ctx, live, meta)
        extra = {"source": "market", "symbol": symbol, "endpoint": provenance["endpoint"],
                 "endpoint_source": provenance["endpoint_source"], "entity_check": provenance["entity_check"],
                 "currency": provenance["currency"], "exchange": provenance["exchange"], "series_bound": False,
                 "provenance": provenance}
        return _result(meta, mount, extra)
    return Tool("download_market_prices", run, idempotent=True)


def open_data_tools(datasets: dict[str, Path] | None, context: OpenDataContext | None) -> dict[str, tuple[Tool, dict]]:
    """The open-data tools as (tool, spec) pairs for default_tools. list_open_data_sources is always offered. The downloads
    and the live-mount sandbox are offered only with a mission context; the sandbox then reads the downloads of this run."""
    live = LiveMounts(datasets)
    pairs: dict[str, tuple[Tool, dict]] = {"list_open_data_sources": (list_sources_tool(), LIST_SOURCES_SPEC)}
    if context is not None:
        pairs["sandbox_python"] = (live_sandbox_tool(live), SANDBOX_PYTHON_SPEC)
        pairs["download_open_data"] = (download_open_data_tool(context, live), DOWNLOAD_OPEN_DATA_SPEC)
        pairs["download_web_data"] = (download_web_data_tool(context, live), DOWNLOAD_WEB_DATA_SPEC)
        pairs["download_market_prices"] = (download_market_prices_tool(context, live), MARKET_PRICES_SPEC)
        from prophet_agent.tools.source_table_tools import source_table_tools
        pairs.update(source_table_tools(context, live))
    return pairs
