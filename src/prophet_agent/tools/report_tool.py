"""One tool that forecasts a series with TimesFM 3 and builds the chart, spreadsheet, document and PDF pack.

Motivation vs Logic
Motivation: the user should get the forecast as files they can open, not only as numbers in the chat.
Logic: the forecast comes from the same TimesFM call that the timesfm_forecast tool uses, so the files and the numbers
cannot drift apart. The pack is built inside the sandbox, and the result carries the manifest path so the files can be
verified later. The tool is idempotent: it writes only to a fresh run folder with no external effect.
"""
from __future__ import annotations

from typing import Any

from prophet_agent.artifacts import build_forecast_pack
from prophet_agent.runtime.runner import Tool
from prophet_agent.tools.series_io import SOURCE_HELP, VALUES_HELP, resolve_series
from prophet_agent.tools.timesfm_tool import _run as timesfm_run

MAX_TITLE = 120
ALLOWED_ARGS = {"values", "source", "horizon", "title"}


def _run(args: dict, key: str) -> dict[str, Any]:
    extra = sorted(set(args) - ALLOWED_ARGS)
    if extra:
        raise ValueError(f"unknown arguments {extra}; allowed: values or source, horizon, title")
    named = resolve_series(args)
    if len(named) != 1:
        raise ValueError(f"the pack is for one series; choose one with source.name from {[n for n, _ in named][:20]}")
    title = str(args.get("title", "Forecast"))[:MAX_TITLE] or "Forecast"
    fc = timesfm_run({k: args[k] for k in ("values", "source", "horizon") if k in args}, key)
    if "series" in fc:
        fc = next(iter(fc["series"].values()))
    values = [float(v) for v in named[0][1]]
    pack = build_forecast_pack(history=values, point=fc["point"], q10=fc["quantiles"]["0.1"],
                               q90=fc["quantiles"]["0.9"], title=title)
    if not pack.ok:
        raise RuntimeError(f"artefact build failed: {pack.stderr[-500:]}")
    return {
        "forecast": fc,
        "run_id": pack.run_id,
        "manifest": f"{pack.workdir}/manifest.json",
        "files": [{"path": o.path, "sha256": o.sha256, "bytes": o.bytes} for o in pack.outputs],
    }


def observe(result: dict) -> dict:
    from prophet_agent.tools.timesfm_tool import _compact
    return {"forecast": _compact(result["forecast"]), "files": [f["path"] for f in result["files"]]}


def forecast_report_tool() -> Tool:
    return Tool("forecast_report", _run, idempotent=True, observe=observe)


FORECAST_REPORT_SPEC = {
    "name": "forecast_report",
    "title": "Forecast pack",
    "activity": "Building the chart, workbook and report",
    "description": "Forecast ONE series with TimesFM 3 and produce the deliverables a person opens: a chart (PNG, SVG), a "
                   "spreadsheet (XLSX), a report (DOCX) and a PDF, listed in a sha256 manifest. Use it for the final "
                   "forecast the person asked for, after any checking is done.",
    "args": {"values": VALUES_HELP, "source": SOURCE_HELP, "horizon": "integer, 1 to 512", "title": "short title for the chart and report"},
}
