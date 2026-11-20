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
from prophet_agent.tools.timesfm_tool import LICENCE_NOTE, _run as timesfm_run

MAX_TITLE = 120
ALLOWED_ARGS = {"values", "horizon", "title"}


def _run(args: dict, key: str) -> dict[str, Any]:
    extra = sorted(set(args) - ALLOWED_ARGS)
    if extra:
        raise ValueError(f"unknown arguments {extra}; the series goes in 'values' (list of numbers); "
                         "allowed: values, horizon, title")
    title = str(args.get("title", "Forecast"))[:MAX_TITLE] or "Forecast"
    fc = timesfm_run({"values": args.get("values"), "horizon": args.get("horizon")}, key)
    values = [float(v) for v in args["values"]]
    point = fc["point"]
    q10 = fc["quantiles"]["0.1"]
    q90 = fc["quantiles"]["0.9"]
    pack = build_forecast_pack(history=values, point=point, q10=q10, q90=q90, title=title, note=LICENCE_NOTE)
    if not pack.ok:
        raise RuntimeError(f"artefact build failed: {pack.stderr[-500:]}")
    return {
        "forecast": fc,
        "run_id": pack.run_id,
        "manifest": f"{pack.workdir}/manifest.json",
        "files": [{"path": o.path, "sha256": o.sha256, "bytes": o.bytes} for o in pack.outputs],
    }


def forecast_report_tool() -> Tool:
    return Tool("forecast_report", _run, idempotent=True)


FORECAST_REPORT_SPEC = {
    "name": "forecast_report",
    "title": "Forecast pack",
    "activity": "Building the chart, workbook and report",
    "description": "Forecast a series with TimesFM 3 and produce a chart (PNG, SVG), spreadsheet (XLSX), report "
                   "(DOCX) and PDF, listed in a sha256 manifest. Research use only.",
    "args": {"values": "list of numbers, oldest first", "horizon": "integer, 1 to 512", "title": "short title"},
}
