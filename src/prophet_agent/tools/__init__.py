"""Tools the goal loop can call. Each is a runtime Tool with a stated idempotency."""
from dataclasses import replace

from .report_tool import FORECAST_REPORT_SPEC, forecast_report_tool
from .sandbox_tool import SANDBOX_PYTHON_SPEC, sandbox_python_tool
from .timesfm_tool import LICENCE_NOTE, QUANTILE_LEVELS, TIMESFM_TOOL_SPEC, timesfm_tool


def spec_doc(spec: dict) -> str:
    """One planner-facing line built from a tool spec, so the planner sees the exact argument names."""
    args = "; ".join(f"{name}: {desc}" for name, desc in spec["args"].items())
    return f"{spec['description']} Arguments (use these names only): {args}."


def default_tools() -> dict:
    specs = {
        "timesfm_forecast": (timesfm_tool(), TIMESFM_TOOL_SPEC),
        "forecast_report": (forecast_report_tool(), FORECAST_REPORT_SPEC),
        "sandbox_python": (sandbox_python_tool(), SANDBOX_PYTHON_SPEC),
    }
    return {name: replace(tool, doc=spec_doc(spec)) for name, (tool, spec) in specs.items()}


__all__ = ["FORECAST_REPORT_SPEC", "LICENCE_NOTE", "QUANTILE_LEVELS", "SANDBOX_PYTHON_SPEC", "TIMESFM_TOOL_SPEC",
           "default_tools", "forecast_report_tool", "sandbox_python_tool", "timesfm_tool"]
