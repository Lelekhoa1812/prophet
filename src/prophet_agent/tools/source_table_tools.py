"""Planner tools for a claimed table: diagnose, extract, validate, fuse, calibrate, and bind.

Motivation vs Logic
Motivation: a working file is not the series the person asked for until the entity check accepts it.
Logic: each tool is idempotent. bind_series computes the verdict again and writes the requested name only on accept.
HTML is read from bytes the caller already has, or from a page the research cache already stored. No new host is contacted.
"""
from __future__ import annotations

from prophet_agent.opendata.store import preview
from prophet_agent.research.fetch import stored_page
from prophet_agent.runtime.runner import ArgumentRejected, Tool
from prophet_agent.source_table.calibrate import calibrate_overlap
from prophet_agent.source_table.claim import build_claim
from prophet_agent.source_table.diagnose import diagnose_failure
from prophet_agent.source_table.extract import extract_html
from prophet_agent.source_table.fuse import fuse_sources
from prophet_agent.source_table.publish import bind_series
from prophet_agent.source_table.validate import validate_table
from prophet_agent.tools.opendata_tools import LiveMounts, OpenDataContext, _mount, _room, _store

_CLAIM_KEYS = {"series_name", "entity", "aliases", "measure", "time_grain", "time_aliases", "entity_aliases",
               "measure_aliases", "tolerance", "expected_rows", "expected_entities", "table_kind"}


def _args(args, allowed: set[str], required: set[str]) -> dict:
    if not isinstance(args, dict):
        raise ArgumentRejected("the arguments must be an object")
    extra = sorted(set(args) - allowed)
    if extra:
        raise ArgumentRejected(f"unknown arguments {extra}; allowed: {', '.join(sorted(allowed)) or 'none'}")
    missing = sorted(required - set(args))
    if missing:
        raise ArgumentRejected(f"missing arguments {missing}")
    return args


def _claim_from(args: dict) -> dict:
    raw = args.get("claim")
    if raw is None:
        raw = {key: args[key] for key in _CLAIM_KEYS if key in args}
    if not isinstance(raw, dict):
        raise ArgumentRejected("claim must be an object")
    try:
        return build_claim(raw)
    except ValueError as exc:
        raise ArgumentRejected(str(exc)) from None


def _csv_bytes(ctx: OpenDataContext, args: dict) -> tuple[bytes, str]:
    if isinstance(args.get("csv"), str):
        return args["csv"].encode("utf-8"), str(args.get("final_url") or "")
    dataset_id = args.get("dataset_id")
    if not isinstance(dataset_id, str):
        raise ArgumentRejected("pass csv text or a dataset_id")
    store = _store(ctx)
    try:
        meta = store.get(dataset_id)
    except KeyError:
        raise ArgumentRejected("dataset_id is not in the mission store") from None
    url = ""
    provenance = meta.get("provenance") or {}
    if isinstance(provenance, dict):
        url = str(provenance.get("final_url") or provenance.get("url") or "")
    if isinstance(args.get("final_url"), str) and args["final_url"].strip():
        url = args["final_url"].strip()
    return store.path(dataset_id).read_bytes(), url


def _reference_csvs(ctx: OpenDataContext, args: dict) -> list[bytes]:
    """Results tables of the same competition, read from the mission store, used to count a league's teams."""
    ids = args.get("reference_dataset_ids")
    if ids is None:
        return []
    if not isinstance(ids, list) or not all(isinstance(i, str) for i in ids) or len(ids) > 5:
        raise ArgumentRejected("reference_dataset_ids must be a list of up to 5 dataset ids")
    store = _store(ctx)
    out = []
    for dataset_id in ids:
        try:
            store.get(dataset_id)
        except KeyError:
            raise ArgumentRejected(f"reference dataset {dataset_id} is not in the mission store") from None
        out.append(store.path(dataset_id).read_bytes())
    return out


def source_table_tools(ctx: OpenDataContext, live: LiveMounts) -> dict[str, tuple[Tool, dict]]:
    """Tools that share the mission's live file mounts with the download tools."""

    def state(args, _key):
        claim = _claim_from(_args(args, _CLAIM_KEYS | {"claim"}, {"entity", "series_name"}))
        return {"tool": "state_table_claim", "status": "ok", **{k: claim[k] for k in claim}}

    def diagnose(args, _key):
        got = _args(args, {"error", "content_type", "url"}, set())
        error = got.get("error") if isinstance(got.get("error"), str) else ""
        media = got.get("content_type") if isinstance(got.get("content_type"), str) else ""
        out = diagnose_failure(error, content_type=media)
        out["tool"] = "diagnose_source"
        if isinstance(got.get("url"), str):
            out["url"] = got["url"][:500]
        return out

    def extract(args, _key):
        got = _args(args, {"html", "url"}, set())
        extracted = None
        if isinstance(got.get("html"), str) and got["html"].strip():
            html, final_url = got["html"][:500_000], str(got.get("url") or "")
            extracted = extract_html(html, final_url=final_url)
        elif isinstance(got.get("url"), str) and got["url"].strip():
            page = stored_page(got["url"].strip())
            if page is None:
                return {"tool": "extract_page_tables", "candidates": [], "file_links": [],
                        "note": "no stored page for that url; call web_fetch first"}
            final_url = str(page.get("final_url") or got["url"])
            cached_tables = page.get("tables") or []
            if not cached_tables:
                return {"tool": "extract_page_tables", "candidates": [], "file_links": page.get("file_links") or [],
                        "final_url": final_url, "note": "the stored page has no data table"}
            extracted = {"candidates": cached_tables, "file_links": page.get("file_links") or [], "skipped_layout": 0}
        else:
            raise ArgumentRejected("pass html or the url of a page web_fetch already stored")
        stored = []
        for candidate in extracted["candidates"]:
            if len(live) >= 10:
                stored.append({"sha256": candidate["sha256"], "columns": candidate["columns"],
                               "row_count": candidate["row_count"], "note": "mission file cap reached"})
                continue
            _room(live)
            body = candidate["csv"].encode("utf-8")
            name = f"table-{candidate['index']}-{candidate['sha256'][:8]}.csv"
            meta = _store(ctx).add(name, body, provenance={
                "source": "html_table", "final_url": final_url, "locator": candidate["locator"],
                "sha256": candidate["sha256"], "series_bound": False,
            })
            mount = _mount(ctx, live, {**meta, "preview": preview(body, "csv")})
            stored.append({"dataset_id": meta["id"], "sandbox_file": f"in/{mount}", "name": meta["name"],
                           "columns": candidate["columns"], "row_count": candidate["row_count"],
                           "sha256": candidate["sha256"], "preview_rows": candidate["preview_rows"],
                           "series_bound": False})
        return {"tool": "extract_page_tables", "candidates": stored, "file_links": extracted["file_links"],
                "skipped_layout": extracted["skipped_layout"], "final_url": final_url}

    def validate(args, _key):
        got = _args(args, _CLAIM_KEYS | {"claim", "csv", "dataset_id", "final_url", "page_text", "reference_dataset_ids"},
                    {"series_name", "entity"})
        claim = _claim_from(got)
        body, url = _csv_bytes(ctx, got)
        page = got.get("page_text") if isinstance(got.get("page_text"), str) else None
        verdict = validate_table(claim, body, final_url=url, page_text=page, reference_csvs=_reference_csvs(ctx, got))
        verdict["tool"] = "validate_table"
        verdict["mapped"] = {key: int(value) for key, value in (verdict.get("mapped") or {}).items()}
        return verdict

    def bind(args, _key):
        got = _args(args, _CLAIM_KEYS | {"claim", "csv", "dataset_id", "final_url", "page_text", "retrieved_at",
                                         "reference_dataset_ids"}, {"series_name", "entity"})
        claim = _claim_from(got)
        body, url = _csv_bytes(ctx, got)
        page = got.get("page_text") if isinstance(got.get("page_text"), str) else None
        retrieved = got.get("retrieved_at") if isinstance(got.get("retrieved_at"), str) else ""
        result = bind_series(_store(ctx), claim, body, final_url=url, retrieved_at=retrieved, page_text=page,
                             reference_csvs=_reference_csvs(ctx, got))
        result["tool"] = "bind_series"
        if result.get("dataset_id") and result.get("status") == "accepted":
            meta = _store(ctx).get(result["dataset_id"])
            meta["preview"] = preview(body, "csv")
            result["sandbox_file"] = f"in/{_mount(ctx, live, meta)}"
        return result

    def fuse(args, _key):
        got = _args(args, {"tables", "time_column", "measure_column", "tolerance"}, {"tables", "time_column", "measure_column"})
        if not isinstance(got["tables"], list):
            raise ArgumentRejected("tables must be a list of source_id and rows")
        try:
            fused = fuse_sources(got["tables"], time_column=str(got["time_column"]), measure_column=str(got["measure_column"]),
                                 tolerance=str(got.get("tolerance") or "exact"))
        except ValueError as exc:
            raise ArgumentRejected(str(exc)) from None
        fused["tool"] = "fuse_sources"
        fused["series_bound"] = False
        return fused

    def calibrate(args, _key):
        got = _args(args, {"left", "right", "time_column", "measure_column", "source_left", "source_right"},
                    {"left", "right", "time_column", "measure_column", "source_left", "source_right"})
        if not isinstance(got["left"], list) or not isinstance(got["right"], list):
            raise ArgumentRejected("left and right must be lists of rows")
        report = calibrate_overlap(got["left"], got["right"], time_column=str(got["time_column"]),
                                   measure_column=str(got["measure_column"]), source_left=str(got["source_left"]),
                                   source_right=str(got["source_right"]))
        report["tool"] = "calibrate_overlap"
        return report

    specs = {
        "state_table_claim": (Tool("state_table_claim", state, idempotent=True), {
            "name": "state_table_claim", "title": "Table claim", "activity": "Stating the table claim",
            "description": "Record the series name, the entity, and the measure the question asked for. Later tools read this claim. An empty entity is refused.",
            "args": {"series_name": "short name for the series, starting with a letter",
                     "entity": "the entity the question names",
                     "aliases": "optional list of other names for that same entity",
                     "measure": "optional column the question asks for, such as the value to forecast",
                     "time_aliases": "optional header names that mean the time column",
                     "entity_aliases": "optional header names that mean the entity column",
                     "measure_aliases": "optional header names that mean the measure",
                     "time_grain": "optional grain, such as day or month",
                     "tolerance": "optional exact, or a relative gap allowed when two sources are later fused",
                     "expected_rows": "optional number of rows the table must have, such as the number of clubs in a league",
                     "expected_entities": "optional list of names the table must hold, such as the clubs",
                     "table_kind": "optional: league for a standings table (no time column; its row count is checked)"},
        }),
        "diagnose_source": (Tool("diagnose_source", diagnose, idempotent=True), {
            "name": "diagnose_source", "title": "Source diagnosis", "activity": "Diagnosing a source",
            "description": "Classify one download error. auth_denied and not_found close that address. html_page means extract_page_tables on a url web_fetch already stored. Do not invent a replacement address.",
            "args": {"error": "the error text from the failed download",
                     "content_type": "optional content type",
                     "url": "optional address the error belongs to"},
        }),
        "extract_page_tables": (Tool("extract_page_tables", extract, idempotent=True), {
            "name": "extract_page_tables", "title": "Page tables", "activity": "Extracting tables from a stored page",
            "description": "Read HTML tables and file links from html you pass or from a url web_fetch already stored. Each table is a working file. series_bound is false until bind_series accepts it.",
            "args": {"html": "optional HTML of a page already fetched",
                     "url": "optional url of a page web_fetch already stored"},
        }),
        "validate_table": (Tool("validate_table", validate, idempotent=True), {
            "name": "validate_table", "title": "Table check", "activity": "Checking a candidate table",
            "description": "Check a candidate against the claim. wrong_entity means the file is a different series. short means fewer rows than expected, with the missing names when the claim lists them: find the full table, do not forecast from part of it. The requested name is not applied here.",
            "args": {"series_name": "the series name from state_table_claim",
                     "entity": "the entity from state_table_claim",
                     "aliases": "optional aliases from the claim",
                     "measure": "optional measure from the claim",
                     "csv": "optional CSV text of the candidate",
                     "dataset_id": "optional id of a stored candidate",
                     "final_url": "optional address the bytes came from",
                     "page_text": "optional page text; each measure cell must appear in it",
                     "expected_rows": "optional expected row count from the claim",
                     "expected_entities": "optional expected names from the claim",
                     "table_kind": "optional table kind from the claim (league)",
                     "reference_dataset_ids": "optional results tables of the same competition, used to count its teams"},
        }),
        "bind_series": (Tool("bind_series", bind, idempotent=True), {
            "name": "bind_series", "title": "Bind series", "activity": "Binding an accepted series",
            "description": "Store the candidate under the claim's series name only when the entity and the columns match. A rejection leaves that name unbound.",
            "args": {"series_name": "the series name from state_table_claim",
                     "entity": "the entity from state_table_claim",
                     "aliases": "optional aliases from the claim",
                     "measure": "optional measure from the claim",
                     "csv": "optional CSV text",
                     "dataset_id": "optional id of a stored candidate",
                     "final_url": "optional address the bytes came from",
                     "page_text": "optional page text the measure cells must appear in",
                     "retrieved_at": "optional retrieval time",
                     "expected_rows": "optional expected row count from the claim",
                     "expected_entities": "optional expected names from the claim",
                     "table_kind": "optional table kind from the claim (league)",
                     "reference_dataset_ids": "optional results tables of the same competition, used to count its teams"},
        }),
        "fuse_sources": (Tool("fuse_sources", fuse, idempotent=True), {
            "name": "fuse_sources", "title": "Fuse sources", "activity": "Fusing accepted sources",
            "description": "Align accepted rows on the time column. When the measure disagrees, both values stay and the row is marked conflict. This does not bind a series name.",
            "args": {"tables": "list of objects with source_id and rows",
                     "time_column": "the time column name inside each row",
                     "measure_column": "the measure column name inside each row",
                     "tolerance": "optional exact, or a relative gap"},
        }),
        "calibrate_overlap": (Tool("calibrate_overlap", calibrate, idempotent=True), {
            "name": "calibrate_overlap", "title": "Overlap calibration", "activity": "Calibrating the overlap",
            "description": "Compare two accepted series on shared times. The observed column is copied. A calibrated column is added and marked derived.",
            "args": {"left": "rows of the observed series",
                     "right": "rows of the comparison series",
                     "time_column": "time column name",
                     "measure_column": "measure column name",
                     "source_left": "id of the observed source",
                     "source_right": "id of the comparison source"},
        }),
    }
    return specs

