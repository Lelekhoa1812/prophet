"""Tools the goal loop can call. Each is a runtime Tool with a stated idempotency."""
from dataclasses import replace
from pathlib import Path

from .data_tools import BACKTEST_SPEC, INSPECT_DATA_SPEC, backtest_tool, inspect_data_tool
from .calibrate_tool import CALIBRATE_SPEC, calibrate_tool
from .chart_tools import chart_tool_pair
from .panel_tool import PANEL_BACKTEST_SPEC, panel_backtest_tool
from .reconcile_tool import RECONCILE_FORECASTS_SPEC, reconcile_forecasts_tool
from .lab_tool import LAB_TOOL_SPEC
from .lab_tool import lab_tool as make_lab_tool
from .drivers_tool import (BUILD_DRIVERS_SPEC, DETECT_ANOMALIES_SPEC, build_drivers_tool, detect_anomalies_tool)
from .notebook_tools import SAVE_TABLE_SPEC, NotebookBinding, save_table_tool
from .opendata_tools import (DOWNLOAD_OPEN_DATA_SPEC, DOWNLOAD_WEB_DATA_SPEC, LIST_SOURCES_SPEC, MARKET_PRICES_SPEC,
                             LiveMounts, OpenDataContext, download_market_prices_tool, download_open_data_tool,
                             download_web_data_tool, list_sources_tool, live_sandbox_tool)
from .research_tools import tools as research_tool_pairs
from .source_table_tools import source_table_tools
from .report_tool import FORECAST_REPORT_SPEC, forecast_report_tool
from .sports_tool import SIMULATE_SEASON_SPEC, simulate_season_tool
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
        "simulate_season": (simulate_season_tool(datasets), SIMULATE_SEASON_SPEC),
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
    # One LiveMounts for the whole mission: the download tools, the source tables, the chart tool and the sandbox all
    # mount into it, so a chart series landed in this run is read by the next sandbox_python call.
    live = LiveMounts(datasets)
    specs.update(_open_data_tools(opendata, live))
    specs.update(chart_tool_pair(datasets, opendata, live))
    download = specs.get("download_web_data")
    if download is not None and opendata is not None and opendata.store is not None:
        # SciRes re-run 2026-11-24 was told to call download_web_data and pasted the preview instead. Landing the file
        # inside the fetch puts the full bytes on the mission before the planner writes a cell. web_research meets the
        # same files, so it lands them too.
        for name in ("web_fetch", "web_research"):
            tool, spec = specs[name]
            specs[name] = (replace(tool, fn=_fetch_lands_the_file(tool.fn, download[0].fn)), spec)
    return {name: replace(tool, doc=spec_doc(spec), title=spec["title"], activity=spec["activity"])
            for name, (tool, spec) in specs.items()}


def _open_data_tools(context: OpenDataContext | None, live: LiveMounts) -> dict[str, tuple]:
    """The open-data tools around one shared LiveMounts. Same pairs as opendata_tools.open_data_tools, which makes its own
    LiveMounts and so cannot share the mission's mounts with the chart tool; the parts are the same public functions."""
    pairs = {"list_open_data_sources": (list_sources_tool(), LIST_SOURCES_SPEC)}
    if context is None:
        return pairs
    pairs["sandbox_python"] = (live_sandbox_tool(live), SANDBOX_PYTHON_SPEC)
    pairs["download_open_data"] = (download_open_data_tool(context, live), DOWNLOAD_OPEN_DATA_SPEC)
    pairs["download_web_data"] = (download_web_data_tool(context, live), DOWNLOAD_WEB_DATA_SPEC)
    pairs["download_market_prices"] = (download_market_prices_tool(context, live), MARKET_PRICES_SPEC)
    pairs.update(source_table_tools(context, live))
    return pairs


def _fetch_lands_the_file(fetch_fn, download_fn):
    """When a fetch or a research round finds a data table, store the full file and name its sandbox path."""
    def run(args, key):
        out = fetch_fn(args, key)
        if not isinstance(out, dict) or not isinstance(args, dict):
            return out
        urls = []
        if out.get("data_file") and isinstance(args.get("url"), str):
            urls.append(args["url"].strip())
        for item in out.get("data_files") or []:
            if isinstance(item, str) and item.strip():
                urls.append(item.strip())
        seen = []
        for url in urls:
            if url not in seen:
                seen.append(url)
        landed_rows = []
        for url in seen:
            try:
                landed = download_fn({"url": url}, key)
            except Exception as exc:
                landed_rows.append({"url": url, "download_error": str(exc)[:300]})
                continue
            row = {"url": url}
            for name in ("sandbox_file", "row_count", "dataset_id", "columns", "first_rows", "name", "last_row"):
                if name in landed:
                    row[name] = landed[name]
            landed_rows.append(row)
        if not landed_rows:
            return out
        if out.get("data_file") and len(landed_rows) == 1:
            row = landed_rows[0]
            out.update({k: v for k, v in row.items() if k != "url"})
            if row.get("download_error"):
                out["download_error"] = row["download_error"]
            elif row.get("sandbox_file"):
                out["note"] = ((out.get("note") or "") + f" The full file is mounted at {row['sandbox_file']} "
                               f"({row.get('row_count')} rows). Read that file in sandbox_python.").strip()
            return out
        out["landed_files"] = landed_rows
        by_url = {row["url"]: row for row in landed_rows}
        for source in out.get("sources") or []:
            row = by_url.get(source.get("url"))
            if not row:
                continue
            source.update({k: v for k, v in row.items() if k != "url"})
        return out
    return run


def tool_catalogue() -> dict:
    """Every tool with its spec, for documentation and the /api/tools listing. The listing binding has no notebook."""
    return default_tools({"example.csv": Path("example.csv")}, binding=NotebookBinding(None, "", ""),
                         opendata=OpenDataContext(store=None, attach=None))


__all__ = ["ANALYST_SKILL_SPEC", "LAB_TOOL_SPEC", "BACKTEST_SPEC", "BUILD_DRIVERS_SPEC", "DETECT_ANOMALIES_SPEC", "FORECAST_REPORT_SPEC", "INSPECT_DATA_SPEC", "NotebookBinding",
           "QUANTILE_LEVELS", "SAVE_TABLE_SPEC", "SANDBOX_PYTHON_SPEC", "TIMESFM_TOOL_SPEC", "default_tools",
           "build_drivers_tool", "detect_anomalies_tool", "forecast_report_tool", "sandbox_python_tool", "save_table_tool", "timesfm_tool", "tool_catalogue"]
