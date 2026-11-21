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
from prophet_agent.forecast.control import CONTROL_ARGS, CONTROL_HELP
from prophet_agent.runtime.runner import Tool
from prophet_agent.tools.series_io import SOURCE_HELP, VALUES_HELP, resolve_series
from prophet_agent.tools.timesfm_tool import _run as timesfm_run

MAX_TITLE = 120
ALLOWED_ARGS = {"values", "source", "horizon", "title"} | CONTROL_ARGS


def _method_note(config: dict) -> str:
    """One line in the document that says what produced the numbers, so the file is honest on its own."""
    c = config.get("control")
    if not c:
        return "TimesFM 3, default settings, no external drivers (research use only)."
    parts = []
    if c.get("ignored_points"):
        parts.append(f"{c['ignored_points']} history points ignored")
    if c.get("future_covariates"):
        parts.append("known-ahead drivers: " + ", ".join(c["future_covariates"]))
    if c.get("past_covariates"):
        parts.append("past drivers: " + ", ".join(c["past_covariates"]))
    if c.get("context_points"):
        parts.append(f"last {c['context_points']} points only")
    if c.get("znorm"):
        parts.append("standardised")
    if c.get("symmetric_averaging"):
        parts.append("symmetric averaging")
    return "TimesFM 3 (research use only); " + "; ".join(parts) + "."


def _run(args: dict, key: str) -> dict[str, Any]:
    extra = sorted(set(args) - ALLOWED_ARGS)
    if extra:
        raise ValueError(f"unknown arguments {extra}; allowed: values or source, horizon, title, "
                         f"and the control options {sorted(CONTROL_ARGS)}")
    named = resolve_series(args)
    if len(named) != 1:
        raise ValueError(f"the pack is for one series; choose one with source.name from {[n for n, _ in named][:20]}")
    title = str(args.get("title", "Forecast"))[:MAX_TITLE] or "Forecast"
    fc = timesfm_run({k: v for k, v in args.items() if k != "title"}, key)
    if "series" in fc:
        fc = next(iter(fc["series"].values()))
    values = [float(v) for v in named[0][1]]
    pack = build_forecast_pack(history=values, point=fc["point"], q10=fc["quantiles"]["0.1"],
                               q90=fc["quantiles"]["0.9"], title=title, note=_method_note(fc["config"]))
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
                   "forecast the person asked for, after any checking is done. It takes the same control arguments as "
                   "timesfm_forecast: pass exactly the configuration that won your backtest, or the pack will show the default forecast.",
    "args": {"values": VALUES_HELP, "source": SOURCE_HELP, "horizon": "integer, 1 to 512", "title": "short title for the chart and report",
             **CONTROL_HELP},
}
