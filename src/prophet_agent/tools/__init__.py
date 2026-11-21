"""Tools the goal loop can call. Each is a runtime Tool with a stated idempotency."""
from dataclasses import replace
from pathlib import Path

from .data_tools import BACKTEST_SPEC, INSPECT_DATA_SPEC, backtest_tool, inspect_data_tool
from .drivers_tool import (BUILD_DRIVERS_SPEC, DETECT_ANOMALIES_SPEC, build_drivers_tool, detect_anomalies_tool)
from .notebook_tools import SAVE_TABLE_SPEC, NotebookBinding, save_table_tool
from .report_tool import FORECAST_REPORT_SPEC, forecast_report_tool
from .sandbox_tool import SANDBOX_PYTHON_SPEC, sandbox_python_tool
from .timesfm_tool import QUANTILE_LEVELS, TIMESFM_TOOL_SPEC, timesfm_tool


def spec_doc(spec: dict) -> str:
    """One planner-facing block built from a tool spec, so the planner sees the exact argument names."""
    args = "; ".join(f"{name}: {desc}" for name, desc in spec["args"].items())
    return f"{spec['description']} Arguments (use these names only): {args}."


def default_tools(datasets: dict[str, Path] | None = None, binding: NotebookBinding | None = None) -> dict:
    """The tool set for one mission. ``datasets`` maps the person's file names to paths; with none, the data tools are
    left out, so the planner is never offered something it cannot use. ``binding`` is set for a notebook turn only,
    which is the only place ``save_table`` is offered."""
    specs = {
        "timesfm_forecast": (timesfm_tool(), TIMESFM_TOOL_SPEC),
        "backtest_series": (backtest_tool(), BACKTEST_SPEC),
        "forecast_report": (forecast_report_tool(), FORECAST_REPORT_SPEC),
        "sandbox_python": (sandbox_python_tool(datasets), SANDBOX_PYTHON_SPEC),
        "build_drivers": (build_drivers_tool(), BUILD_DRIVERS_SPEC),
        "detect_anomalies": (detect_anomalies_tool(), DETECT_ANOMALIES_SPEC),
    }
    if datasets:
        specs["inspect_data"] = (inspect_data_tool(datasets), INSPECT_DATA_SPEC)
    if binding is not None:
        specs["save_table"] = (save_table_tool(binding), SAVE_TABLE_SPEC)
    return {name: replace(tool, doc=spec_doc(spec), title=spec["title"], activity=spec["activity"])
            for name, (tool, spec) in specs.items()}


def tool_catalogue() -> dict:
    """Every tool with its spec, for documentation and the /api/tools listing. The listing binding has no notebook."""
    return default_tools({"example.csv": Path("example.csv")}, binding=NotebookBinding(None, "", ""))


__all__ = ["BACKTEST_SPEC", "BUILD_DRIVERS_SPEC", "DETECT_ANOMALIES_SPEC", "FORECAST_REPORT_SPEC", "INSPECT_DATA_SPEC", "NotebookBinding",
           "QUANTILE_LEVELS", "SAVE_TABLE_SPEC", "SANDBOX_PYTHON_SPEC", "TIMESFM_TOOL_SPEC", "default_tools",
           "build_drivers_tool", "detect_anomalies_tool", "forecast_report_tool", "sandbox_python_tool", "save_table_tool", "timesfm_tool", "tool_catalogue"]
