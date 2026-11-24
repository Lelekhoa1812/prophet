"""Tools the goal loop can call. Each is a runtime Tool with a stated idempotency."""
from dataclasses import replace
from pathlib import Path

from .data_tools import BACKTEST_SPEC, INSPECT_DATA_SPEC, backtest_tool, inspect_data_tool
from .calibrate_tool import CALIBRATE_SPEC, calibrate_tool
from .panel_tool import PANEL_BACKTEST_SPEC, panel_backtest_tool
from .reconcile_tool import RECONCILE_FORECASTS_SPEC, reconcile_forecasts_tool
from .lab_tool import LAB_TOOL_SPEC
from .lab_tool import lab_tool as make_lab_tool
from .drivers_tool import (BUILD_DRIVERS_SPEC, DETECT_ANOMALIES_SPEC, build_drivers_tool, detect_anomalies_tool)
from .notebook_tools import SAVE_TABLE_SPEC, NotebookBinding, save_table_tool
from .opendata_tools import OpenDataContext, open_data_tools
from .research_tools import tools as research_tool_pairs
from .report_tool import FORECAST_REPORT_SPEC, forecast_report_tool
from .sandbox_tool import SANDBOX_PYTHON_SPEC, sandbox_python_tool
from .skill_tool import ANALYST_SKILL_SPEC, skill_tool
from .timesfm_tool import QUANTILE_LEVELS, TIMESFM_TOOL_SPEC, timesfm_tool


def spec_doc(spec: dict) -> str:
    """One planner-facing block built from a tool spec, so the planner sees the exact argument names."""
    args = "; ".join(f"{name}: {desc}" for name, desc in spec["args"].items())
    return f"{spec['description']} Arguments (use these names only): {args}."


def default_tools(datasets: dict[str, Path] | None = None, binding: NotebookBinding | None = None, *,
                  opendata: OpenDataContext | None = None) -> dict:
    """The tool set for one mission. ``datasets`` maps the person's file names to paths; with none, the data tools are
    left out, so the planner is never offered something it cannot use. ``binding`` is set for a notebook turn only,
    which is the only place ``save_table`` is offered. ``opendata`` is the mission's dataset store and attach hook; it
    enables the download tools and makes the sandbox read files downloaded during the run."""
    specs = {
        "timesfm_forecast": (timesfm_tool(), TIMESFM_TOOL_SPEC),
        "backtest_series": (backtest_tool(), BACKTEST_SPEC),
        "forecast_report": (forecast_report_tool(), FORECAST_REPORT_SPEC),
        "sandbox_python": (sandbox_python_tool(datasets), SANDBOX_PYTHON_SPEC),
        "build_drivers": (build_drivers_tool(), BUILD_DRIVERS_SPEC),
        "detect_anomalies": (detect_anomalies_tool(), DETECT_ANOMALIES_SPEC),
        "analyst_skill": (skill_tool(), ANALYST_SKILL_SPEC),
        "control_lab": (make_lab_tool(), LAB_TOOL_SPEC),
        "calibrate_intervals": (calibrate_tool(), CALIBRATE_SPEC),
        "panel_backtest": (panel_backtest_tool(), PANEL_BACKTEST_SPEC),
        "reconcile_forecasts": (reconcile_forecasts_tool(), RECONCILE_FORECASTS_SPEC),
    }
    for tool, spec in research_tool_pairs():
        specs[tool.name] = (tool, spec)
    if datasets:
        specs["inspect_data"] = (inspect_data_tool(datasets), INSPECT_DATA_SPEC)
    if binding is not None:
        specs["save_table"] = (save_table_tool(binding), SAVE_TABLE_SPEC)
    specs.update(open_data_tools(datasets, opendata))
    download = specs.get("download_web_data")
    fetch = specs.get("web_fetch")
    if download is not None and fetch is not None and opendata is not None and opendata.store is not None:
        # SciRes re-run 2026-11-24 was told to call download_web_data and pasted the preview instead. Landing the file
        # inside the fetch puts the full bytes on the mission before the planner writes a cell.
        tool, spec = fetch
        specs["web_fetch"] = (replace(tool, fn=_fetch_lands_the_file(tool.fn, download[0].fn)), spec)
    return {name: replace(tool, doc=spec_doc(spec), title=spec["title"], activity=spec["activity"])
            for name, (tool, spec) in specs.items()}


def _fetch_lands_the_file(fetch_fn, download_fn):
    """When a fetch is a data table, store the full file and add its sandbox path to the fetch result."""
    def run(args, key):
        out = fetch_fn(args, key)
        if not isinstance(out, dict) or not out.get("data_file") or not isinstance(args, dict):
            return out
        url = args.get("url")
        if not isinstance(url, str) or not url.strip():
            return out
        try:
            landed = download_fn({"url": url.strip()}, key)
        except Exception as exc:
            out["download_error"] = str(exc)[:300]
            return out
        for name in ("sandbox_file", "row_count", "dataset_id", "columns", "first_rows", "name", "last_row"):
            if name in landed:
                out[name] = landed[name]
        if out.get("sandbox_file"):
            out["note"] = ((out.get("note") or "") + f" The full file is mounted at {out['sandbox_file']} "
                           f"({out.get('row_count')} rows). Read that file in sandbox_python.").strip()
        return out
    return run


def tool_catalogue() -> dict:
    """Every tool with its spec, for documentation and the /api/tools listing. The listing binding has no notebook."""
    return default_tools({"example.csv": Path("example.csv")}, binding=NotebookBinding(None, "", ""),
                         opendata=OpenDataContext(store=None, attach=None))


__all__ = ["ANALYST_SKILL_SPEC", "LAB_TOOL_SPEC", "BACKTEST_SPEC", "BUILD_DRIVERS_SPEC", "DETECT_ANOMALIES_SPEC", "FORECAST_REPORT_SPEC", "INSPECT_DATA_SPEC", "NotebookBinding",
           "QUANTILE_LEVELS", "SAVE_TABLE_SPEC", "SANDBOX_PYTHON_SPEC", "TIMESFM_TOOL_SPEC", "default_tools",
           "build_drivers_tool", "detect_anomalies_tool", "forecast_report_tool", "sandbox_python_tool", "save_table_tool", "timesfm_tool", "tool_catalogue"]
